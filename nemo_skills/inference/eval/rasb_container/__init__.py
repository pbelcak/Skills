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
RASB container files.

This package contains the evaluation infrastructure that gets injected into
RASB Docker containers:
- evaluate.py: Main evaluation orchestrator (runs synth_*.json samples)
- judge.py: Requirements-based LLM committee judging
- lm.py: Model wrapper classes for API calls

These files are copied from the original RASB benchmark to ensure
reproducibility of evaluation results.
"""

from pathlib import Path

CONTAINER_FILES_DIR = Path(__file__).parent
