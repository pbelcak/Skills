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
RASB evaluator module.

This evaluator passes through results from the Docker container evaluation.
The actual evaluation (output parsing, exact match, requirements judging)
is done by evaluate.py inside the container for reproducibility.

The evaluator here simply extracts the evaluation results from rasb_result
and formats them for Skills metrics aggregation.
"""

import logging
from typing import Any, Dict

from nemo_skills.evaluation.evaluator.base import BaseEvaluator
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


class RasbEvaluator(BaseEvaluator):
    """
    Evaluator for RASB benchmark.

    Evaluation is performed inside Docker containers by evaluate.py
    (matching the original RASB benchmark for reproducibility).
    This evaluator extracts and formats those results for Skills.

    The rasb_result field from generation contains:
    - passed: bool - Whether the sample passed evaluation
    - judgment_type: str - "exact" or "requirements"
    - judgment_details: dict - Details of the judgment
    - model_output: str - Raw model output
    - parsed_output: Any - Parsed output (JSON, regex match, etc.)
    - error: str | None - Error message if evaluation failed
    """

    def __init__(self, config: Dict[str, Any], num_parallel_requests: int = 10):
        super().__init__(config, num_parallel_requests)

    async def eval_single(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract evaluation results from the container output.

        The is_correct field is already set by the generation task
        based on the container's evaluate.py results.
        """
        rasb_result = data_point.get("rasb_result", {})

        # The container sets 'passed' field
        is_correct = rasb_result.get("passed", False)

        # Extract additional evaluation details
        judgment_type = rasb_result.get("judgment_type", "unknown")
        judgment_details = rasb_result.get("judgment_details", {})
        parsed_output = rasb_result.get("parsed_output")
        error = rasb_result.get("error")

        result = {
            "is_correct": is_correct,
            "judgment_type": judgment_type,
        }

        # Include parsed output if available
        if parsed_output is not None:
            result["parsed_output"] = parsed_output

        # Include error if present
        if error:
            result["error"] = error
            result["is_correct"] = False

        # Include judgment details for debugging
        if judgment_details:
            # For requirements-based judgments, include requirement-level results
            if judgment_type == "requirements" and "requirements" in judgment_details:
                result["requirements_passed"] = sum(
                    1 for r in judgment_details["requirements"] if r.get("pass")
                )
                result["requirements_total"] = len(judgment_details["requirements"])

        return result

    def supports_single_eval(self) -> bool:
        """RASB evaluation is done inside containers, not during generation."""
        return False
