#!/usr/bin/env python3
"""In-container evaluation script for RASB benchmarking.

Runs inside a Docker container at /benchmark/. Iterates over synthetic
samples, invokes the callable, evaluates outputs against expected results
(exact match or LLM judge committee), and writes structured results.
"""

import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Ensure the working directory is on sys.path so judge.py can find lm.py
sys.path.insert(0, os.getcwd())

Path("results").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/evaluate.log", mode="w"),
    ],
)
log = logging.getLogger("evaluate")

from dotenv import load_dotenv

load_dotenv(".env")

log.info("Working directory: %s", os.getcwd())
log.info("Python: %s", sys.version)
log.info("TARGET_MODEL: %s", os.environ.get("TARGET_MODEL", "(not set)"))
log.info("MODEL_ANTHROPIC: %s", os.environ.get("MODEL_ANTHROPIC", "(not set)"))

from callable import call
from tools import TOOLS, execute_tool
from judge import judge_requirements

# Try to import apply_inputs from harness for assembled_user_message support
_harness_apply_inputs = None
try:
    from harness import apply_inputs as _harness_apply_inputs
    log.info("Imported apply_inputs from harness.py")
except (ImportError, AttributeError):
    pass

log.info("Imports OK. TOOLS defined: %d", len(TOOLS))
if TOOLS:
    for tname, tspec in TOOLS.items():
        params = list(tspec.get("parameters", {}).keys())
        has_func = callable(tspec.get("function"))
        log.info("  Tool '%s': params=%s, function=%s", tname, params,
                 "OK" if has_func else "MISSING")


# ---------------------------------------------------------------------------
# Helpers (mirrored from harness.py to avoid import issues across envs)
# ---------------------------------------------------------------------------

def apply_placeholders(template: str, fields: dict) -> str:
    result = template
    for key, value in fields.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(placeholder, str(value))
    return result


def _build_prompts(input_mode, system_template, user_template, fields):
    """Build system/user prompts from fields using input_mode routing."""
    # Try harness.apply_inputs first (handles assembled_user_message and custom logic)
    if _harness_apply_inputs is not None:
        try:
            result = _harness_apply_inputs(system_template, user_template, fields, input_mode)
            if isinstance(result, tuple) and len(result) == 2:
                sys_out, usr_out = result
                # Only accept plain text prompts; reject multimodal content blocks
                if isinstance(sys_out, str) and isinstance(usr_out, str):
                    return result
                log.warning("Harness returned non-string prompt content (multimodal?), falling through to built-in")
        except Exception:
            pass  # Fall through to built-in handling

    if input_mode == "placeholder_system":
        return apply_placeholders(system_template, fields), user_template
    elif input_mode == "placeholder_user":
        return system_template, apply_placeholders(user_template, fields)
    elif input_mode == "placeholder_both":
        return apply_placeholders(system_template, fields), apply_placeholders(user_template, fields)
    elif input_mode == "direct_user_message":
        return system_template, str(list(fields.values())[0]) if fields else ""
    else:
        return apply_placeholders(system_template, fields), apply_placeholders(user_template, fields)


def _extract_full_response(messages):
    """Extract complete response including tool call activity.

    For non-tool envs: returns last assistant text (existing behavior).
    For tool envs: concatenates assistant text + tool call summaries + tool results.
    """
    has_tool_calls = any(m.get("tool_calls") for m in messages if m.get("role") == "assistant")

    if not has_tool_calls:
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                return msg.get("content", "")
        return ""

    parts = []
    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            content = msg.get("content", "")
            if content:
                parts.append(content)
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", tc)
                name = fn.get("name", "") if isinstance(fn, dict) else tc.get("name", "")
                parts.append(f"[Called tool: {name}]")
        elif role == "tool":
            content = msg.get("content", "")
            name = msg.get("name", "tool")
            parts.append(f"[{name} result]: {content}")
    return "\n".join(parts)


def parse_json_output(text: str):
    """Extract JSON from response text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Code blocks
    for match in re.findall(r"```(?:json)?\s*([\s\S]*?)```", text):
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    # Bare JSON object
    for match in re.findall(r"\{[\s\S]*\}", text):
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    return None


def parse_regex_extraction(text: str, pattern):
    """Extract output using a regex pattern from parsing_details."""
    if isinstance(pattern, dict):
        # Multi-pattern extraction: {name: regex_pattern, ...}
        # Only process keys that look like patterns (end with _pattern) or are strings
        results = {}
        for name, pat in pattern.items():
            # Skip non-pattern keys (e.g., intensity_type, intensity_range)
            if not isinstance(pat, str):
                continue
            try:
                matches = re.findall(pat, text)
                results[name] = matches[0] if len(matches) == 1 else matches if matches else None
            except re.error:
                # Invalid regex pattern, skip it
                results[name] = None
        return results
    if not isinstance(pattern, str):
        return None
    matches = re.findall(pattern, text)
    if matches:
        return matches[0] if len(matches) == 1 else matches
    return None


def extract_tool_call_result(messages: list[dict], parsing_details: str):
    """Extract output from tool call arguments.

    Looks for the last tool call matching the description in parsing_details
    and returns its arguments (or a specific field).
    """
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", tc)
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            if args:
                return args
    return None


def _get_written_files() -> dict[str, str]:
    """Collect files written by tools during the session.

    Checks for get_virtual_filesystem() in tools.py first (in-memory writes),
    then falls back to scanning the results directory for any written artifacts.
    """
    try:
        from tools import get_virtual_filesystem
        return get_virtual_filesystem()
    except (ImportError, AttributeError):
        pass
    return {}


def process_tool_calls(messages: list[dict], max_iterations: int = 10) -> list[dict]:
    if not TOOLS:
        return messages
    iterations = 0
    while iterations < max_iterations:
        last = messages[-1] if messages else None
        if not last or not last.get("tool_calls"):
            break
        for tc in last["tool_calls"]:
            # Support both OpenAI format {"function": {"name": ..., "arguments": ...}}
            # and flat format {"name": ..., "arguments": ...}
            if "function" in tc:
                name = tc["function"]["name"]
                args_raw = tc["function"].get("arguments", "{}")
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            else:
                name = tc["name"]
                args = tc.get("arguments", {})
            try:
                result = execute_tool(name, args)
                messages.append({"role": "tool", "name": name, "content": str(result),
                                 "tool_call_id": tc.get("id", "")})
            except Exception as e:
                messages.append({"role": "tool", "name": name, "content": f"Error: {e}",
                                 "tool_call_id": tc.get("id", "")})
        try:
            messages.extend(call(messages))
        except Exception as e:
            messages.append({"role": "assistant", "content": f"Error: {e}"})
            break
        iterations += 1
    return messages


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

def _normalize_for_exact(value) -> str:
    """Normalize a value for exact comparison."""
    if isinstance(value, str):
        # Try parsing as JSON for canonical form
        try:
            parsed = json.loads(value)
            return json.dumps(parsed, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        except (json.JSONDecodeError, TypeError):
            return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return str(value).strip()


def evaluate_exact(model_output: str, parsed_output, expected_value) -> dict:
    """Evaluate an exact-match judgment."""
    # Try matching against parsed output first (for json_parse environments)
    if parsed_output is not None:
        norm_parsed = _normalize_for_exact(parsed_output)
        norm_expected = _normalize_for_exact(expected_value)
        if norm_parsed == norm_expected:
            return {"passed": True, "comparison": "parsed_output_match"}

    # Fall back to raw output comparison
    norm_raw = _normalize_for_exact(model_output)
    norm_expected = _normalize_for_exact(expected_value)
    passed = norm_raw == norm_expected
    return {"passed": passed, "comparison": "raw_output_match" if passed else "no_match",
            "expected_normalized": norm_expected, "actual_normalized": norm_raw}


def evaluate_requirements(
    system_prompt: str, user_prompt: str,
    model_output: str, requirements: list[str],
) -> dict:
    """Evaluate a requirements-based judgment using LLM judge committee."""
    result = judge_requirements(system_prompt, user_prompt, model_output, requirements)
    return {"passed": result["overall_pass"], **result}


# ---------------------------------------------------------------------------
# Incremental results writing
# ---------------------------------------------------------------------------

def _write_incremental_results(env_id: str, sample_results: list[dict]) -> None:
    """Write current results to disk so timeout preserves progress."""
    total = len(sample_results)
    passed = sum(1 for s in sample_results if s["passed"])
    errors = sum(1 for s in sample_results if s.get("error"))
    failed = total - passed

    results = {
        "env_id": env_id,
        "pass_rate": passed / total if total else 0,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "total": total,
        "samples": sample_results,
    }

    Path("results").mkdir(exist_ok=True)
    Path("results/results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n"
    )


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation():
    """Run evaluation on all synthetic samples."""
    # Load config
    metadata = json.loads(Path("metadata.json").read_text())
    env_id = metadata.get("id", "unknown")
    input_mode = metadata.get("input_mode", "placeholder_system")
    max_turns = metadata.get("max_turns", 1)
    output_schema = metadata.get("output_schema", {})
    parsing_method = output_schema.get("parsing", "face_value")

    # Load prompts
    system_template = Path("prompt_system.txt").read_text()
    user_template = Path("prompt_user.txt").read_text() if Path("prompt_user.txt").exists() else ""

    # Collect synthetic input files (only those with expected_output)
    inputs_dir = Path("inputs")
    synth_files = sorted(inputs_dir.glob("synth_*.json"))
    if not synth_files:
        print("No synthetic sample files found.")
        results = {"env_id": env_id, "pass_rate": 0, "passed": 0, "failed": 0,
                   "errors": 0, "total": 0, "samples": []}
        Path("results").mkdir(exist_ok=True)
        Path("results/results.json").write_text(json.dumps(results, indent=2))
        return

    # Load existing results for resumption (skip already-evaluated, error-free samples)
    existing_results: dict[str, dict] = {}
    results_file = Path("results/results.json")
    if results_file.exists():
        try:
            cached = json.loads(results_file.read_text())
            for sample in cached.get("samples", []):
                fname = sample.get("input_file")
                if fname and not sample.get("error"):
                    existing_results[fname] = sample
            if existing_results:
                log.info("Resuming: found %d existing error-free results", len(existing_results))
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Could not load existing results for resumption: %s", e)

    sample_results = []

    for sf in synth_files:
        # Skip if already evaluated without error
        if sf.name in existing_results:
            log.info("--- Skipping: %s (already evaluated) ---", sf.name)
            sample_results.append(existing_results[sf.name])
            continue

        log.info("--- Evaluating: %s ---", sf.name)
        start = time.time()
        sample_data = json.loads(sf.read_text())
        fields = sample_data.get("fields", {})
        expected = sample_data.get("expected_output")
        log.debug("Fields: %s", json.dumps(fields, ensure_ascii=False)[:500])
        log.debug("Expected judgment: %s", expected.get("judgment") if expected else "none")

        if not expected:
            print(f"  Skipping {sf.name} (no expected_output)")
            continue

        try:
            sys_prompt, usr_prompt = _build_prompts(input_mode, system_template, user_template, fields)

            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": usr_prompt},
            ]

            # Invoke callable
            response_msgs = call(messages)
            messages.extend(response_msgs)
            messages = process_tool_calls(messages, max_iterations=max_turns * 2)

            final_response = _extract_full_response(messages)

            # Append written file contents so the judge can see them
            written_files = _get_written_files()
            if written_files:
                file_section = "\n\n--- Written Files ---\n"
                for fpath, fcontent in written_files.items():
                    file_section += f"\n=== {fpath} ===\n{fcontent}\n"
                final_response += file_section

            # Parse output
            parsed_output = None
            if parsing_method == "json_parse":
                parsed_output = parse_json_output(final_response)
            elif parsing_method == "regex_extraction":
                pattern = output_schema.get("parsing_details", "")
                parsed_output = parse_regex_extraction(final_response, pattern)
            elif parsing_method == "tool_call_result":
                parsed_output = extract_tool_call_result(messages, output_schema.get("parsing_details", ""))
                if parsed_output is not None:
                    final_response += f"\n\n[Tool call result]: {json.dumps(parsed_output, ensure_ascii=False)}"
            else:
                parsed_output = final_response

            # Evaluate
            judgment_type = expected.get("judgment")
            if judgment_type == "exact":
                judgment = evaluate_exact(final_response, parsed_output, expected["value"])
            elif judgment_type == "requirements":
                judgment = evaluate_requirements(
                    sys_prompt, usr_prompt, final_response, expected["requirements"],
                )
            else:
                judgment = {"passed": False, "error": f"Unknown judgment type: {judgment_type}"}

            duration_ms = int((time.time() - start) * 1000)
            sample_results.append({
                "input_file": sf.name,
                "passed": judgment["passed"],
                "judgment_type": judgment_type,
                "judgment_details": judgment,
                "model_output": final_response,
                "parsed_output": parsed_output if isinstance(parsed_output, (dict, list, str, type(None))) else str(parsed_output),
                "messages": messages,
                "error": None,
                "duration_ms": duration_ms,
            })
            status = "PASS" if judgment["passed"] else "FAIL"
            log.info("%s -> %s (%dms)", sf.name, status, duration_ms)
            if not judgment["passed"] and judgment_type == "requirements":
                failed_reqs = [r for r in judgment.get("requirements", []) if not r.get("pass")]
                for fr in failed_reqs:
                    log.debug("  FAILED req: %s", fr.get("text", "")[:100])

        except Exception as exc:
            duration_ms = int((time.time() - start) * 1000)
            sample_results.append({
                "input_file": sf.name,
                "passed": False,
                "judgment_type": expected.get("judgment", "unknown"),
                "judgment_details": {},
                "model_output": None,
                "parsed_output": None,
                "messages": messages if 'messages' in locals() else [],
                "error": str(exc),
                "duration_ms": duration_ms,
            })
            log.error("%s -> ERROR: %s (%dms)", sf.name, exc, duration_ms)

        # Write incremental results after each sample (so timeout preserves progress)
        _write_incremental_results(env_id, sample_results)

    # Compute stats
    total = len(sample_results)
    passed = sum(1 for s in sample_results if s["passed"])
    errors = sum(1 for s in sample_results if s["error"])
    failed = total - passed

    results = {
        "env_id": env_id,
        "pass_rate": passed / total if total else 0,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "total": total,
        "samples": sample_results,
    }

    Path("results").mkdir(exist_ok=True)
    Path("results/results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n"
    )

    print(f"\n{'='*50}")
    print(f"EVALUATION: {env_id}")
    print(f"{'='*50}")
    print(f"Total: {total}  Passed: {passed}  Failed: {failed}  Errors: {errors}")
    print(f"Pass rate: {results['pass_rate']:.1%}")
    print(f"{'='*50}")


def _check_one_sample(sf, input_mode, system_template, user_template,
                      max_attempts, max_turns, parsing_method, output_schema=None):
    """Check solvability of a single sample. Returns a result dict."""
    log.info("--- Solvability: %s (up to %d attempts) ---", sf.name, max_attempts)
    output_schema = output_schema or {}
    sample_data = json.loads(sf.read_text())
    fields = sample_data.get("fields", {})
    expected = sample_data["expected_output"]

    sys_prompt, usr_prompt = _build_prompts(input_mode, system_template, user_template, fields)

    solved = False
    solution = None
    last_judgment = None
    last_attempt = 0

    for attempt in range(1, max_attempts + 1):
        last_attempt = attempt
        start = time.time()
        try:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": usr_prompt},
            ]

            response_msgs = call(messages)
            messages.extend(response_msgs)
            messages = process_tool_calls(messages, max_iterations=max_turns * 2)

            final_response = _extract_full_response(messages)

            # Append written file contents so the judge can see them
            written_files = _get_written_files()
            if written_files:
                file_section = "\n\n--- Written Files ---\n"
                for fpath, fcontent in written_files.items():
                    file_section += f"\n=== {fpath} ===\n{fcontent}\n"
                final_response += file_section

            parsed_output = None
            if parsing_method == "json_parse":
                parsed_output = parse_json_output(final_response)
            elif parsing_method == "regex_extraction":
                pattern = output_schema.get("parsing_details", "")
                parsed_output = parse_regex_extraction(final_response, pattern)
            elif parsing_method == "tool_call_result":
                parsed_output = extract_tool_call_result(messages, output_schema.get("parsing_details", ""))
                if parsed_output is not None:
                    final_response += f"\n\n[Tool call result]: {json.dumps(parsed_output, ensure_ascii=False)}"
            else:
                parsed_output = final_response

            judgment_type = expected.get("judgment")
            if judgment_type == "exact":
                judgment = evaluate_exact(final_response, parsed_output, expected["value"])
            elif judgment_type == "requirements":
                judgment = evaluate_requirements(
                    sys_prompt, usr_prompt, final_response, expected["requirements"],
                )
            else:
                judgment = {"passed": False, "error": f"Unknown judgment: {judgment_type}"}

            last_judgment = judgment
            duration_ms = int((time.time() - start) * 1000)

            if judgment["passed"]:
                log.info("%s attempt %d -> PASS (%dms)", sf.name, attempt, duration_ms)
                solved = True
                solution = {
                    "model_output": final_response,
                    "parsed_output": parsed_output if isinstance(parsed_output, (dict, list, str, type(None))) else str(parsed_output),
                    "attempt_number": attempt,
                    "total_attempts": attempt,
                }
                break
            else:
                log.info("%s attempt %d -> FAIL (%dms)", sf.name, attempt, duration_ms)

        except Exception as exc:
            log.error("%s attempt %d -> ERROR: %s", sf.name, attempt, exc)
            last_judgment = {"passed": False, "error": str(exc)}

    result = {
        "input_file": sf.name,
        "solvable": solved,
        "total_attempts": last_attempt if not solved else solution["attempt_number"],
    }
    if solved:
        result["solution"] = solution
    else:
        result["last_judgment"] = last_judgment
    return result


def run_solvability():
    """Run solvability checking: retry each sample up to K times, record first success."""
    max_attempts = int(os.environ.get("RASB_MAX_ATTEMPTS", "5"))
    sample_filter = os.environ.get("RASB_SAMPLES", "all")
    workers = int(os.environ.get("RASB_SOLVABILITY_WORKERS", "4"))

    # Load config
    metadata = json.loads(Path("metadata.json").read_text())
    env_id = metadata.get("id", "unknown")
    input_mode = metadata.get("input_mode", "placeholder_system")
    max_turns = metadata.get("max_turns", 1)
    output_schema = metadata.get("output_schema", {})
    parsing_method = output_schema.get("parsing", "face_value")

    system_template = Path("prompt_system.txt").read_text()
    user_template = Path("prompt_user.txt").read_text() if Path("prompt_user.txt").exists() else ""

    # Collect target samples
    inputs_dir = Path("inputs")
    if sample_filter == "all":
        target_files = sorted(inputs_dir.glob("*.json"))
    else:
        names = [n.strip() for n in sample_filter.split(",") if n.strip()]
        target_files = [inputs_dir / n for n in names if (inputs_dir / n).exists()]

    target_files = [f for f in target_files
                    if json.loads(f.read_text()).get("expected_output")]

    log.info("Solvability check: %d samples, max_attempts=%d, workers=%d",
             len(target_files), max_attempts, workers)

    sample_results = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _check_one_sample, sf, input_mode, system_template,
                user_template, max_attempts, max_turns, parsing_method, output_schema,
            ): sf
            for sf in target_files
        }
        for future in as_completed(futures):
            sf = futures[future]
            try:
                result = future.result()
                sample_results.append(result)
                status = "SOLVABLE" if result["solvable"] else "unsolvable"
                log.info("%s -> %s (%d attempts)", sf.name, status, result["total_attempts"])
            except Exception as exc:
                log.error("%s -> EXCEPTION: %s", sf.name, exc)
                sample_results.append({
                    "input_file": sf.name, "solvable": False,
                    "total_attempts": max_attempts,
                    "last_judgment": {"passed": False, "error": str(exc)},
                })

    Path("results").mkdir(exist_ok=True)
    Path("results/solvability.json").write_text(
        json.dumps({"env_id": env_id, "samples": sample_results},
                   indent=2, ensure_ascii=False) + "\n"
    )

    solvable_count = sum(1 for s in sample_results if s["solvable"])
    log.info("Solvability: %d/%d samples solvable", solvable_count, len(sample_results))


if __name__ == "__main__":
    mode = os.environ.get("RASB_MODE", "benchmark")
    if mode == "solvability":
        run_solvability()
    else:
        run_evaluation()
