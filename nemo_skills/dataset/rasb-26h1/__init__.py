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
RASB 26H1 benchmark dataset module.

RASB (Real Agent Scaffolds Bench) 26H1 is the first bi-annual snapshot
containing 193 environments with 5,731 verified synthetic test samples
from 63 real agent repositories.

Environment types: generation (83), evaluation (36), retrieval (29),
extraction (24), coding (11), codebase (10).
"""

# Environment folders are stored alongside the JSONL (like images in MMMU-Pro)
REQUIRES_DATA_DIR = True

# Custom generation module handles Docker container orchestration
GENERATION_MODULE = "nemo_skills.inference.eval.rasb"

# Custom metrics for RASB-specific aggregation by environment type
METRICS_TYPE = "rasb"

# Default generation arguments with custom evaluator
GENERATION_ARGS = "++eval_type=rasb"

# Default evaluation split
EVAL_SPLIT = "test"
