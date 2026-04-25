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
RASB metrics module.

Computes pass rates aggregated by:
- Overall pass rate
- Environment type (generation, evaluation, retrieval, extraction, coding, codebase)
- Output parsing strategy (face_value, json_parse, regex_extraction, tool_call_result)
- Judgment type (exact, requirements)
- Source repository

Also computes aggregate statistics across environments:
- Mean pass rate
- Median pass rate
- First quartile (Q1) pass rate
- Third quartile (Q3) pass rate
- Standard deviation
"""

import logging
import math
from collections import defaultdict
from typing import Any

from nemo_skills.evaluation.metrics.base import BaseMetrics
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


class RasbMetrics(BaseMetrics):
    """
    Metrics class for RASB benchmark.

    Provides aggregated pass rates across multiple dimensions:
    - Overall pass rate
    - Pass rate by environment type
    - Pass rate by output parsing strategy
    - Pass rate by judgment type (exact vs requirements)
    - Pass rate by tool usage
    - Pass rate by source repository
    """

    def __init__(self, compute_no_answer: bool = True):
        super().__init__(compute_no_answer=compute_no_answer)

    def reset(self):
        """Reset all counters."""
        self.total = 0
        self.correct = 0
        self.avg_tokens = 0
        self.max_k = 1
        self.min_start_time = float("inf")
        self.max_end_time = float("-inf")
        self.eval_dict = {"pass@1": {}}
        self.all_scores = defaultdict(list)

        # RASB-specific counters
        self.by_env_type = defaultdict(lambda: {"total": 0, "correct": 0})
        self.by_output_parsing = defaultdict(lambda: {"total": 0, "correct": 0})
        self.by_judgment_type = defaultdict(lambda: {"total": 0, "correct": 0})
        self.by_interaction_mode = defaultdict(lambda: {"total": 0, "correct": 0})
        self.by_repo = defaultdict(lambda: {"total": 0, "correct": 0})
        self.by_has_tools = {True: {"total": 0, "correct": 0}, False: {"total": 0, "correct": 0}}

        # Error tracking
        self.errors = 0
        self.container_errors = 0

        # Per-environment tracking for aggregate statistics
        self.by_env_id = defaultdict(lambda: {"total": 0, "correct": 0})

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        """Return score dictionary for a prediction."""
        return {"is_correct": prediction.get("is_correct", False)}

    def get_incorrect_sample(self, prediction: dict) -> dict:
        """Return a modified prediction marked as incorrect."""
        return {**prediction, "is_correct": False}

    def update(self, predictions: list[dict]):
        """
        Update metrics with a list of predictions for a single sample.

        For RASB, we typically have 1 prediction per sample.
        """
        super().update(predictions)

        # Use first prediction for aggregation
        pred = predictions[0]
        rasb_result = pred.get("rasb_result", {})

        # Overall metrics - use is_correct which is set from rasb_result["passed"]
        is_correct = pred.get("is_correct", False)
        self.correct += int(is_correct)

        # Token counting
        if "num_generated_tokens" in pred:
            self.avg_tokens += pred["num_generated_tokens"]

        # Timing
        if "_generation_start_time" in pred:
            self.min_start_time = min(self.min_start_time, pred["_generation_start_time"])
        if "_generation_end_time" in pred:
            self.max_end_time = max(self.max_end_time, pred["_generation_end_time"])

        # By environment type
        env_type = pred.get("environment_type", "unknown")
        self.by_env_type[env_type]["total"] += 1
        self.by_env_type[env_type]["correct"] += int(is_correct)

        # By output parsing (from metadata in input)
        output_parsing = pred.get("output_parsing", "unknown")
        self.by_output_parsing[output_parsing]["total"] += 1
        self.by_output_parsing[output_parsing]["correct"] += int(is_correct)

        # By judgment type (from rasb_result)
        judgment_type = rasb_result.get("judgment_type", "unknown")
        self.by_judgment_type[judgment_type]["total"] += 1
        self.by_judgment_type[judgment_type]["correct"] += int(is_correct)

        # By interaction mode
        interaction_mode = pred.get("interaction_mode", "unknown")
        self.by_interaction_mode[interaction_mode]["total"] += 1
        self.by_interaction_mode[interaction_mode]["correct"] += int(is_correct)

        # By repository
        repo = pred.get("repo", "unknown")
        self.by_repo[repo]["total"] += 1
        self.by_repo[repo]["correct"] += int(is_correct)

        # By tool usage
        has_tools = pred.get("has_tools", False)
        self.by_has_tools[has_tools]["total"] += 1
        self.by_has_tools[has_tools]["correct"] += int(is_correct)

        # Per-environment tracking (for aggregate stats)
        env_id = pred.get("env_id", "unknown")
        self.by_env_id[env_id]["total"] += 1
        self.by_env_id[env_id]["correct"] += int(is_correct)

        # Error tracking
        if rasb_result.get("error"):
            self.errors += 1
        if rasb_result.get("status") in ("container_error", "base_build_failed", "bench_build_failed"):
            self.container_errors += 1

        # Store score for pass@k computation
        self.all_scores["is_correct"].append([is_correct])

        # Update pass@1 metric
        self.eval_dict["pass@1"]["is_correct"] = self.correct

    def _compute_rate(self, counter: dict) -> float:
        """Compute pass rate as percentage."""
        if counter["total"] == 0:
            return 0.0
        return 100.0 * counter["correct"] / counter["total"]

    def _compute_percentile(self, sorted_values: list, percentile: float) -> float:
        """Compute percentile from sorted list of values."""
        if not sorted_values:
            return 0.0
        n = len(sorted_values)
        idx = (n - 1) * percentile
        lower = int(idx)
        upper = lower + 1
        if upper >= n:
            return sorted_values[-1]
        weight = idx - lower
        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

    def _compute_aggregate_stats(self) -> dict[str, float]:
        """
        Compute aggregate statistics across all environments.

        Returns:
            Dictionary with mean, median, Q1, Q3, std, and overall pass rates.
        """
        # Collect per-environment pass rates
        pass_rates = []
        for env_id, counter in self.by_env_id.items():
            if counter["total"] > 0:
                rate = 100.0 * counter["correct"] / counter["total"]
                pass_rates.append(rate)

        if not pass_rates:
            return {
                "mean_pass_rate": 0.0,
                "median_pass_rate": 0.0,
                "q1_pass_rate": 0.0,
                "q3_pass_rate": 0.0,
                "std_pass_rate": 0.0,
                "overall_pass_rate": 0.0,
                "num_environments": 0,
            }

        # Sort for percentile calculations
        sorted_rates = sorted(pass_rates)
        n = len(sorted_rates)

        # Mean
        mean = sum(pass_rates) / n

        # Standard deviation
        variance = sum((r - mean) ** 2 for r in pass_rates) / n
        std = math.sqrt(variance)

        # Median (Q2)
        median = self._compute_percentile(sorted_rates, 0.5)

        # First quartile (Q1)
        q1 = self._compute_percentile(sorted_rates, 0.25)

        # Third quartile (Q3)
        q3 = self._compute_percentile(sorted_rates, 0.75)

        # Overall pass rate (total correct / total samples across all envs)
        total_samples = sum(c["total"] for c in self.by_env_id.values())
        total_correct = sum(c["correct"] for c in self.by_env_id.values())
        overall = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0

        return {
            "mean_pass_rate": mean,
            "median_pass_rate": median,
            "q1_pass_rate": q1,
            "q3_pass_rate": q3,
            "std_pass_rate": std,
            "overall_pass_rate": overall,
            "num_environments": n,
        }

    def get_metrics(self) -> dict[str, Any]:
        """
        Get all computed metrics.

        Returns:
            Dictionary with metrics organized by aggregation mode.
        """
        metrics = super().get_metrics()

        # Add overall pass rate
        if "pass@1" in metrics:
            metrics["pass@1"]["pass_rate"] = (
                100.0 * self.correct / self.total if self.total > 0 else 0.0
            )

        # Add pass rates by environment type
        env_type_metrics = {}
        for env_type, counter in sorted(self.by_env_type.items()):
            env_type_metrics[f"pass_rate_{env_type}"] = self._compute_rate(counter)
            env_type_metrics[f"count_{env_type}"] = counter["total"]

        if "pass@1" in metrics:
            metrics["pass@1"].update(env_type_metrics)

        # Add pass rates by output parsing
        parsing_metrics = {}
        for parsing, counter in sorted(self.by_output_parsing.items()):
            parsing_metrics[f"pass_rate_parsing_{parsing}"] = self._compute_rate(counter)

        if "pass@1" in metrics:
            metrics["pass@1"].update(parsing_metrics)

        # Add pass rates by judgment type
        judgment_metrics = {}
        for jtype, counter in sorted(self.by_judgment_type.items()):
            judgment_metrics[f"pass_rate_judgment_{jtype}"] = self._compute_rate(counter)
            judgment_metrics[f"count_judgment_{jtype}"] = counter["total"]

        if "pass@1" in metrics:
            metrics["pass@1"].update(judgment_metrics)

        # Add pass rates by tool usage
        if "pass@1" in metrics:
            tools_counter = self.by_has_tools[True]
            no_tools_counter = self.by_has_tools[False]
            metrics["pass@1"]["pass_rate_with_tools"] = self._compute_rate(tools_counter)
            metrics["pass@1"]["pass_rate_no_tools"] = self._compute_rate(no_tools_counter)
            metrics["pass@1"]["count_with_tools"] = tools_counter["total"]
            metrics["pass@1"]["count_no_tools"] = no_tools_counter["total"]

        # Add error counts
        if "pass@1" in metrics:
            metrics["pass@1"]["errors"] = self.errors
            metrics["pass@1"]["container_errors"] = self.container_errors

        # Add aggregate statistics across environments
        if "pass@1" in metrics:
            aggregate_stats = self._compute_aggregate_stats()
            metrics["pass@1"].update(aggregate_stats)

        return metrics

    def get_breakdown_by_repo(self) -> dict[str, dict]:
        """
        Get detailed breakdown by source repository.

        Returns:
            Dictionary mapping repo name to pass rate and count.
        """
        breakdown = {}
        for repo, counter in sorted(self.by_repo.items(), key=lambda x: -x[1]["total"]):
            breakdown[repo] = {
                "pass_rate": self._compute_rate(counter),
                "total": counter["total"],
                "correct": counter["correct"],
            }
        return breakdown

    def get_breakdown_by_type(self) -> dict[str, dict]:
        """
        Get detailed breakdown by environment type.

        Returns:
            Dictionary mapping environment type to pass rate and count.
        """
        breakdown = {}
        for env_type, counter in sorted(self.by_env_type.items()):
            breakdown[env_type] = {
                "pass_rate": self._compute_rate(counter),
                "total": counter["total"],
                "correct": counter["correct"],
            }
        return breakdown

    def get_breakdown_by_env_id(self) -> dict[str, dict]:
        """
        Get detailed breakdown by environment ID.

        Returns:
            Dictionary mapping environment ID to pass rate and count,
            sorted by pass rate (ascending) for easy identification of
            challenging environments.
        """
        breakdown = {}
        for env_id, counter in self.by_env_id.items():
            breakdown[env_id] = {
                "pass_rate": self._compute_rate(counter),
                "total": counter["total"],
                "correct": counter["correct"],
            }
        # Sort by pass rate (ascending) to show hardest environments first
        return dict(sorted(breakdown.items(), key=lambda x: x[1]["pass_rate"]))
