"""LLM judge for evaluating requirements-based expected outputs.

Uses AnthropicLM from lm.py with Claude Opus 4.5 as a committee of judges
to determine whether model outputs satisfy a list of requirements.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# lm.py lives in the same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lm import AnthropicLM, LMQueryError

log = logging.getLogger("pap.judge")

# Default judge model - should be a reliable, high-quality model
DEFAULT_JUDGE_MODEL = "azure/anthropic/claude-opus-4-5"


def resolve_judge_model() -> str:
    """Resolve the model to use for judging.

    Uses JUDGE_MODEL env var if set, otherwise defaults to Claude Opus 4.5.
    This is intentionally separate from TARGET_MODEL (the model being tested)
    to ensure judging works reliably regardless of which model is being benchmarked.
    """
    model = os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    if model.startswith("nvidia_nim/"):
        model = model[len("nvidia_nim/"):]
    return model

JUDGE_SYSTEM_PROMPT = """\
You are an expert judge evaluating whether an AI model's output satisfies a set of requirements.

You will be given:
1. The prompt that was given to the model (system + user messages)
2. The model's actual output
3. A list of requirements that the output should satisfy

For each requirement, determine whether the output satisfies it. Be strict but fair:
- A requirement is satisfied only if the output clearly and unambiguously meets it
- Partial satisfaction counts as not satisfied
- Interpret requirements literally unless context makes the intent obvious

Respond with a JSON object (no markdown fences, just raw JSON):
{
  "evaluations": [
    {"requirement": "<the requirement text>", "satisfied": true/false, "reasoning": "<brief explanation>"},
    ...
  ]
}
"""


def _build_judge_prompt(
    prompt_system: str,
    prompt_user: str,
    model_output: str,
    requirements: list[str],
) -> str:
    """Build the user message for a single judge invocation."""
    reqs_formatted = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(requirements))
    return f"""\
## Prompt Given to Model

**System:**
{prompt_system}

**User:**
{prompt_user}

## Model Output
{model_output}

## Requirements to Evaluate
{reqs_formatted}

Evaluate each requirement and respond with the JSON object."""


def _parse_judge_response(response_text: str, requirements: list[str]) -> list[dict]:
    """Parse a judge's JSON response into a list of evaluations."""
    # Try to extract JSON from the response
    text = response_text.strip()
    # Handle markdown fences
    if "```" in text:
        match = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        log.warning("Failed to parse judge response as JSON")
        # Return all requirements as unsatisfied
        return [{"requirement": r, "satisfied": False, "reasoning": "Judge response unparseable"} for r in requirements]

    # Handle both {"evaluations": [...]} and raw list [...] formats
    if isinstance(data, list):
        evals = data
    else:
        evals = data.get("evaluations", [])
    if len(evals) != len(requirements):
        log.warning("Judge returned %d evaluations but %d requirements given", len(evals), len(requirements))

    return evals


def _run_single_judge(
    judge: AnthropicLM,
    prompt_system: str,
    prompt_user: str,
    model_output: str,
    requirements: list[str],
) -> list[dict]:
    """Run a single judge instance and return its evaluations."""
    user_msg = _build_judge_prompt(prompt_system, prompt_user, model_output, requirements)
    try:
        response = judge.query_messages([
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ], max_tokens=4096)
        return _parse_judge_response(response, requirements)
    except LMQueryError as exc:
        log.error("Judge query failed: %s", exc)
        return [{"requirement": r, "satisfied": False, "reasoning": f"Judge error: {exc}"} for r in requirements]


def judge_requirements(
    prompt_system: str,
    prompt_user: str,
    model_output: str,
    requirements: list[str],
    committee_size: int = 3,
) -> dict:
    """Judge whether model_output satisfies requirements using an LLM committee.

    Args:
        prompt_system: The system prompt that was given to the model.
        prompt_user: The user message that was given to the model.
        model_output: The model's actual output text.
        requirements: List of requirement strings to evaluate.
        committee_size: Number of parallel judge instances (default 3).

    Returns:
        {
            "requirements": [
                {
                    "text": "the requirement",
                    "pass": True/False,  # majority vote
                    "votes": [True, False, True],  # individual judge votes
                    "reasoning": ["...", "...", "..."],  # judge reasoning
                },
                ...
            ],
            "overall_pass": True/False,  # all requirements pass
            "pass_count": int,  # number of requirements that passed
            "total_count": int,  # total requirements
        }
    """
    if not requirements:
        return {"requirements": [], "overall_pass": True, "pass_count": 0, "total_count": 0}

    judge = AnthropicLM(model=resolve_judge_model(), max_tokens=4096)

    # Run committee in parallel
    all_evals: list[list[dict]] = []
    with ThreadPoolExecutor(max_workers=committee_size) as pool:
        futures = [
            pool.submit(
                _run_single_judge, judge,
                prompt_system, prompt_user, model_output, requirements,
            )
            for _ in range(committee_size)
        ]
        for fut in as_completed(futures):
            all_evals.append(fut.result())

    # Aggregate by majority vote
    result_reqs = []
    for i, req_text in enumerate(requirements):
        votes = []
        reasoning = []
        for judge_evals in all_evals:
            if i < len(judge_evals):
                votes.append(bool(judge_evals[i].get("satisfied", False)))
                reasoning.append(judge_evals[i].get("reasoning", ""))
            else:
                votes.append(False)
                reasoning.append("No evaluation returned")

        pass_count = sum(votes)
        passes = pass_count > committee_size / 2  # majority

        result_reqs.append({
            "text": req_text,
            "pass": passes,
            "votes": votes,
            "reasoning": reasoning,
        })

    passed = sum(1 for r in result_reqs if r["pass"])
    return {
        "requirements": result_reqs,
        "overall_pass": passed == len(requirements),
        "pass_count": passed,
        "total_count": len(requirements),
    }
