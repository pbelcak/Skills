# RASB (Real Agent Scaffolds Bench)

## Overview

RASB evaluates how well LLMs follow complex agent scaffolding rather than solving problems autonomously. It tests models as "bearers of scaffolds" - following extensive system prompts, using tools correctly, and producing outputs in specified formats.

The benchmark uses bi-annual snapshots:

- **26H1**: First 2026 snapshot with 193 environments and 5,731 verified synthetic test samples from 63 real agent repositories

## Architecture

The Skills integration follows the original RASB benchmark architecture for reproducibility:

1. **Base image**: Built from each environment's Dockerfile
2. **Overlay image**: Adds evaluation infrastructure (evaluate.py, judge.py, lm.py, callable)
3. **Container execution**: Runs `evaluate.py` which processes all `synth_*.json` samples
4. **Results collection**: Reads `results/results.json` from the container

The container logic is identical to the original RASB benchmark. Only orchestration and results aggregation are adapted for Skills.

## Benchmark Definition

- Dataset: [`nemo_skills/dataset/rasb-26h1/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/rasb-26h1/__init__.py)
- Generation module: [`nemo_skills/inference/eval/rasb.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/inference/eval/rasb.py)
- Container files: [`nemo_skills/inference/eval/rasb_container/`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/inference/eval/rasb_container/)
- Evaluator: [`nemo_skills/evaluation/evaluator/rasb.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/evaluation/evaluator/rasb.py)

## Environment Types

RASB covers six environment types based on the primary task:

| Type | Samples | Description |
|------|---------|-------------|
| generation | 2,463 | Content generation (text, code, structured data) |
| evaluation | 1,075 | Judging, scoring, or comparing outputs |
| retrieval | 860 | Finding relevant information from context |
| extraction | 720 | Extracting structured data from unstructured input |
| coding | 330 | Code generation with execution validation |
| codebase | 283 | Working with multi-file code repositories |

## Data Preparation

RASB requires Docker for evaluation. Each environment contains a Dockerfile that builds the execution container.

### Prerequisites

1. Docker must be installed and running
2. Access to RASB 26H1 environment data (contact the RASB maintainers or refer to the RASB technical report for data access)

### Setup

Once you have obtained the RASB 26H1 data, link the environments to the dataset directory:

```bash
ln -s /path/to/rasb-26h1/26h1 nemo_skills/dataset/rasb-26h1/26h1
```

Prepare the benchmark data:

```bash
ns prepare_data rasb-26h1
```

Or specify a custom data source:

```bash
ns prepare_data rasb-26h1 --data_source=/path/to/rasb-26h1/26h1
```

This creates `test.jsonl` with pointers to each environment and input file.

## Evaluation

### Quickstart Example

```bash
ns eval \
  --benchmarks rasb-26h1 \
  --server_type openai \
  --model azure/anthropic/claude-opus-4-5 \
  --server_address https://inference-api.nvidia.com \
  --output_dir /workspace/rasb-eval
```

### Supported Endpoints and Models

RASB supports multiple API types and model providers:

| Server Type | Compatible Models | Example Endpoint |
|-------------|-------------------|------------------|
| `openai` | GPT-4, GPT-4o, Claude (via proxy), Gemini (via proxy) | OpenAI-compatible APIs |
| `anthropic` | Claude models | Anthropic SDK-compatible APIs |

Supported endpoint types:
- OpenAI-compatible APIs (completions and responses endpoints)
- Anthropic SDK-compatible APIs
- Local model servers with compatible APIs

### Controlling Evaluation Scope

The `++max_samples` parameter controls how many samples (and thus environments) to evaluate. Samples are ordered by environment in test.jsonl, with each environment containing ~30 samples (range: 20-30).

| max_samples | Environments | Use case |
|-------------|--------------|----------|
| `30` | ~1 | Quick smoke test |
| `300` | ~10 | Development testing |
| `3000` | ~100 | Partial benchmark |
| `-1` | all 193 | Full benchmark (5,731 samples) |

**Formula**: To evaluate N environments, set `max_samples` to approximately `N × 30`.

### Docker Configuration

RASB runs each environment in an isolated Docker container (processing all samples for that environment). Key configuration options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `++docker_timeout` | 1800 | Timeout per environment in seconds (30 min) |
| `++docker_memory_limit` | 4g | Memory limit per container |
| `++max_concurrent_containers` | 2 | Parallel container execution |
| `++keep_containers` | False | Keep containers for debugging |
| `++rebuild_images` | False | Force rebuild Docker images |
| `++docker_build_timeout` | 600 | Image build timeout (10 min) |

### Example with Docker Settings

```bash
ns eval \
  --benchmarks rasb-26h1 \
  --server_type openai \
  --model gpt-4o \
  --output_dir /workspace/rasb-eval \
  ++docker_timeout=3600 \
  ++max_concurrent_containers=4 \
  ++docker_memory_limit=8g
```

## Judgment Types

RASB uses two judgment types (evaluated inside the container):

| Type | Description |
|------|-------------|
| `exact` | Exact match comparison (normalized JSON or string) |
| `requirements` | LLM judge committee evaluates against requirements list |

The requirements-based judgment uses a committee of LLM judges for open-ended tasks where exact matching isn't appropriate.

## Output Parsing Strategies

RASB uses four output parsing strategies defined per environment:

| Strategy | Description |
|----------|-------------|
| `face_value` | Use model output as-is |
| `json_parse` | Parse output as JSON (handles markdown fences) |
| `regex_extraction` | Extract output using regex patterns from metadata |
| `tool_call_result` | Extract from tool call arguments |

## Benchmark Results

Results from the RASB 26H1 technical report (pass rates in %):

| Model | Overall | Mean | Std | Median | Q1 | Q3 |
|-------|---------|------|-----|--------|----|----|
| Claude 4.5 Opus | 85.8 | 85.5 | 17.2 | 90.0 | 80.0 | 96.7 |
| Claude 4.6 Opus | 83.0 | 82.8 | 19.1 | 90.0 | 76.7 | 96.7 |
| Claude 4.6 Sonnet | 78.4 | 78.0 | 23.5 | 86.7 | 70.0 | 96.0 |
| Claude 4.7 Opus | 78.6 | 78.3 | 23.1 | 86.7 | 66.7 | 96.7 |
| Claude 4.5 Sonnet | 77.0 | 76.7 | 22.2 | 83.3 | 66.7 | 93.3 |
| Claude 4.5 Haiku | 71.1 | 70.9 | 24.9 | 76.7 | 60.0 | 90.0 |
| Gemini 3 Flash | 71.1 | 70.8 | 23.7 | 76.7 | 56.7 | 90.0 |
| GPT-5.1 | 70.9 | 70.6 | 28.0 | 76.7 | 60.0 | 90.0 |
| GPT-5.3 Codex | 70.4 | 70.0 | 28.1 | 76.7 | 56.7 | 92.0 |
| Gemini 3.1 Pro | 70.4 | 70.1 | 25.1 | 73.3 | 52.0 | 93.3 |
| GPT-5.3 | 68.9 | 68.4 | 29.0 | 76.7 | 53.3 | 90.0 |
| Gemini 2.5 Flash | 66.7 | 66.4 | 24.8 | 70.0 | 46.7 | 84.0 |
| Gemini 3.1 Flash Lite | 64.8 | 64.5 | 27.7 | 66.7 | 46.7 | 86.7 |
| Nemotron Super | 63.3 | 62.9 | 25.9 | 63.3 | 43.3 | 83.3 |
| Nemotron Nano | 55.6 | 55.2 | 28.0 | 53.3 | 33.3 | 80.0 |

Overall is the sample-weighted pass rate across all 5,731 samples. Mean, Std, Median, Q1, and Q3 are computed over per-environment pass rates (193 environments).

## Metrics

RASB reports pass rates aggregated by:

- **Overall**: Total correct / total samples
- **By environment type**: Pass rate per type (generation, evaluation, etc.)
- **By output parsing**: Pass rate by parsing strategy
- **By judgment type**: Pass rate for exact vs requirements judgments
- **By tool usage**: Pass rate for samples with/without tools
- **By repository**: Pass rate per source repository

### Aggregate Statistics

RASB also computes statistics across all environments:

| Metric | Description |
|--------|-------------|
| `mean_pass_rate` | Average pass rate across environments |
| `median_pass_rate` | Median pass rate (50th percentile) |
| `q1_pass_rate` | First quartile (25th percentile) |
| `q3_pass_rate` | Third quartile (75th percentile) |
| `std_pass_rate` | Standard deviation of pass rates |
| `overall_pass_rate` | Total correct / total samples |
| `num_environments` | Number of environments evaluated |

The mean treats all environments equally regardless of sample count, while the overall rate weights by sample count. The quartiles help identify performance distribution across environments.

### Example Metrics Output

```json
{
  "pass@1": {
    "pass_rate": 72.5,
    "pass_rate_generation": 78.2,
    "pass_rate_evaluation": 65.1,
    "pass_rate_retrieval": 71.8,
    "pass_rate_judgment_exact": 85.3,
    "pass_rate_judgment_requirements": 58.7,
    "pass_rate_with_tools": 68.4,
    "pass_rate_no_tools": 74.1,
    "errors": 15,
    "container_errors": 2,
    "mean_pass_rate": 71.3,
    "median_pass_rate": 73.5,
    "q1_pass_rate": 58.2,
    "q3_pass_rate": 85.0,
    "std_pass_rate": 18.7,
    "overall_pass_rate": 72.5,
    "num_environments": 193
  }
}
```

## Execution Flow

1. **Group samples**: Samples grouped by environment
2. **Build base image**: Docker image built from environment's Dockerfile (cached)
3. **Build overlay image**: Adds evaluate.py, judge.py, lm.py, callable, .env
4. **Run container**: Container executes `evaluate.py` processing all `synth_*.json` files
5. **Collect results**: Results read from `results/results.json`
6. **Map to samples**: Container results mapped back to individual sample entries

## Troubleshooting

### Container Build Failures

Check the environment's Dockerfile and requirements:

```bash
cd nemo_skills/dataset/rasb-26h1/26h1/<environment>/
docker build -t test .
```

### Debugging Container Execution

Enable `keep_containers` to inspect failed containers:

```bash
ns eval ... ++keep_containers=True
```

Then inspect the container logs:

```bash
# Check the results directory
ls /workspace/rasb-eval/environments/<env_id>/
cat /workspace/rasb-eval/environments/<env_id>/container_stdout.log
cat /workspace/rasb-eval/environments/<env_id>/results.json
```

### Network Issues

By default, containers use `host` network mode to access the LLM server. If your server is not on localhost, ensure the container can reach it.

### API Credentials

The container needs API credentials. Set the appropriate environment variable before running:

```bash
# For OpenAI
export OPENAI_API_KEY=your-openai-key

# For Anthropic
export ANTHROPIC_API_KEY=your-anthropic-key

# For NVIDIA Inference API
export NVAPI_KEY=your-nvidia-key
```

Ensure the endpoint URL and model name match your API provider. Connection errors inside Docker containers often indicate mismatched endpoint/model configuration.
