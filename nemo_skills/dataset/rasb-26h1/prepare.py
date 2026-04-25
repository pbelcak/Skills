# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Prepare RASB 26H1 benchmark data.

This script creates a pointer JSONL that references environment folders
rather than converting them. Each entry contains:
- environment_path: relative path to the environment folder
- input_file: name of the input JSON file
- metadata fields for filtering and metrics aggregation

The actual environment folders (Dockerfile, tools.py, prompts, inputs)
remain as-is, following the pattern used by SWE-bench (container references)
and MMMU-Pro (image paths).

Usage:
    ns prepare_data rasb-26h1

    Or with custom data location:
    ns prepare_data rasb-26h1 --data_source=/path/to/rasb-26h1/26h1
"""

import argparse
import json
import os
import urllib.request
from pathlib import Path


# Default location for RASB 26H1 data
# Users can override this with --data_source argument
DEFAULT_DATA_URL = None  # TODO: Add HuggingFace or other hosted URL when available


def load_metadata(env_path: Path) -> dict:
    """Load and validate environment metadata."""
    metadata_file = env_path / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"No metadata.json in {env_path}")

    with open(metadata_file, "r", encoding="utf-8") as f:
        return json.load(f)


def get_input_files(env_path: Path) -> list[Path]:
    """Get verified synthetic test inputs from the environment's inputs directory.

    Only synth_*.json files are used for benchmarking (matching the original
    RASB benchmark behavior). Example files and candidate files are excluded.
    """
    inputs_dir = env_path / "inputs"
    if not inputs_dir.exists():
        return []
    return sorted(inputs_dir.glob("synth_*.json"))


def create_entry(
    env_path: Path,
    input_file: Path,
    metadata: dict,
    base_dir: Path,
) -> dict:
    """Create a single JSONL entry for an environment/input pair."""
    # Relative path from the dataset directory
    rel_env_path = env_path.relative_to(base_dir)

    # Extract output parsing strategy
    output_schema = metadata.get("output_schema", {})
    output_parsing = output_schema.get("parsing", "face_value")

    return {
        # Pointers to environment and input (like image_path in MMMU-Pro)
        "environment_path": str(rel_env_path),
        "input_file": input_file.name,
        # Metadata for filtering and metrics
        "task_id": metadata.get("id", env_path.name),
        "environment_type": metadata.get("environment_type", "unknown"),
        "interaction_mode": metadata.get("interaction_mode", "single_shot"),
        "output_parsing": output_parsing,
        "max_turns": metadata.get("max_turns", 1),
        "has_tools": bool(metadata.get("tools", [])),
        # Additional metadata for analysis
        "repo": metadata.get("repo", ""),
        "input_mode": metadata.get("input_mode", ""),
    }


def save_data(split: str = "test", data_source: str | None = None):
    """Prepare RASB 26H1 data by creating pointer JSONL."""
    data_dir = Path(__file__).absolute().parent

    # Determine where environments are located
    if data_source:
        environments_dir = Path(data_source)
    else:
        environments_dir = data_dir / "26h1"

    if not environments_dir.exists():
        # Try to download if URL is configured
        if DEFAULT_DATA_URL:
            print(f"Downloading RASB 26H1 data from {DEFAULT_DATA_URL}...")
            # TODO: Implement download logic when hosting is available
            raise NotImplementedError("Automatic download not yet implemented")

        raise FileNotFoundError(
            f"Environment directory not found: {environments_dir}\n\n"
            "Please provide the path to RASB 26H1 environments using:\n"
            "  ns prepare_data rasb-26h1 --data_source=/path/to/rasb-26h1/26h1\n\n"
            "Or create a symlink:\n"
            f"  ln -s /path/to/rasb-26h1/26h1 {data_dir / '26h1'}"
        )

    output_file = data_dir / f"{split}.jsonl"

    entries = []
    env_count = 0
    input_count = 0
    skipped_envs = []

    print(f"Scanning environments in {environments_dir}...")

    for env_path in sorted(environments_dir.iterdir()):
        if not env_path.is_dir():
            continue

        # Skip hidden directories
        if env_path.name.startswith("."):
            continue

        try:
            metadata = load_metadata(env_path)
        except FileNotFoundError as e:
            skipped_envs.append((env_path.name, str(e)))
            continue

        input_files = get_input_files(env_path)
        if not input_files:
            skipped_envs.append((env_path.name, "No input files found"))
            continue

        env_count += 1

        for input_file in input_files:
            entry = create_entry(env_path, input_file, metadata, data_dir)
            entries.append(entry)
            input_count += 1

    # Write output
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    # Summary
    print(f"\nRASB 26H1 Preparation Summary")
    print(f"{'=' * 40}")
    print(f"Environments processed: {env_count}")
    print(f"Total samples: {input_count}")
    print(f"Output file: {output_file}")

    if skipped_envs:
        print(f"\nSkipped {len(skipped_envs)} environments:")
        for name, reason in skipped_envs[:5]:
            print(f"  - {name}: {reason}")
        if len(skipped_envs) > 5:
            print(f"  ... and {len(skipped_envs) - 5} more")

    # Environment type breakdown
    type_counts = {}
    for entry in entries:
        env_type = entry["environment_type"]
        type_counts[env_type] = type_counts.get(env_type, 0) + 1

    print(f"\nSamples by environment type:")
    for env_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {env_type}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare RASB 26H1 benchmark data"
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("test",),
        help="Dataset split to prepare (default: test)",
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default=None,
        help="Path to RASB 26H1 environments directory (default: ./26h1)",
    )
    args = parser.parse_args()

    save_data(args.split, args.data_source)
