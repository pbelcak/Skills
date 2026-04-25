# RASB-26H1 Benchmark

## Overview

RASB (Real Agent Scaffolds Bench) 26H1 is the first bi-annual snapshot containing 193 environments with 5,731 verified synthetic test samples from 63 real agent repositories.

## Setup

### 1. Link Environment Data

The RASB environments must be symlinked from the agentcompiler repository:

```bash
ln -s /path/to/rasb-26h1/26h1 nemo_skills/dataset/rasb-26h1/26h1
```

### 2. Prepare Data

```bash
ns prepare_data rasb-26h1
```

This creates `test.jsonl` with pointers to each environment.

## Running Evaluation

### NVIDIA Inference API (Ouickstart Example)

Use the NVIDIA Inference API endpoint for Anthropic models:

```bash
ns eval \
  --benchmarks rasb-26h1 \
  --output_dir /path/to/output \
  --model azure/anthropic/claude-opus-4-5 \
  --server_type openai \
  --server_address https://inference-api.nvidia.com \
  ++max_samples=150 \
  ++docker_timeout=3600 \
  ++max_concurrent_containers=2
```

### Environment Variables

Set the API key before running:

```bash
export NVAPI_KEY=your-nvidia-api-key
```

### Controlling Evaluation Scope

The `++max_samples` parameter controls how many samples to evaluate. Samples are ordered by environment in test.jsonl, with each environment containing ~30 samples.

| max_samples | Environments | Use case |
|-------------|--------------|----------|
| `30` | ~1 | Quick smoke test |
| `300` | ~10 | Development testing |
| `3000` | ~100 | Partial benchmark |
| `-1` | all 193 | Full benchmark (5,731 samples) |

**Formula**: To evaluate N environments, set `max_samples` to approximately `N × 30`.

### Configuration Reference

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--server_address` | `https://inference-api.nvidia.com` | NVIDIA Inference API endpoint |
| `--model` | `azure/anthropic/claude-opus-4-5` | Claude Opus via NVIDIA NIM |
| `++docker_timeout` | `3600` | 1 hour timeout per environment |
| `++max_concurrent_containers` | `2` | Parallel container execution |
| `++max_samples` | `-1` | All samples, or `N × 30` for N envs |

## Common Issues

### Connection Errors Inside Docker Containers

If you see `"Error querying ... Connection error."` for all samples in an environment:

1. **Wrong endpoint**: Ensure you're using the correct endpoint address, e.g., `https://inference-api.nvidia.com`

2. **Wrong model name**: Use the correct full model path string, e.g., `azure/anthropic/claude-opus-4-5`

3. **Network mode**: Containers use `--network host` by default. Ensure Docker can reach external APIs.

4. **API key**: Ensure `NVAPI_KEY` (or other applicable API key environment variable) is set and valid.


## Metrics

RASB reports aggregate statistics across environments:

| Metric | Description |
|--------|-------------|
| `pass_rate` | Overall pass rate (sample-weighted) |
| `mean_pass_rate` | Average pass rate across environments |
| `median_pass_rate` | Median pass rate (50th percentile) |
| `q1_pass_rate` | First quartile (25th percentile) |
| `q3_pass_rate` | Third quartile (75th percentile) |
| `std_pass_rate` | Standard deviation |
| `num_environments` | Number of environments evaluated |

## Files

- `__init__.py` - Benchmark registration
- `prepare.py` - Data preparation script
- `test.jsonl` - Generated test samples (after `ns prepare_data`)
- `26h1/` - Symlink to environment folders (must be created manually)
