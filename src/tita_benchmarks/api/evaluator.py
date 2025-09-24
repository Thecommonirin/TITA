import json
import os
from typing import Dict
import numpy as np


class APIBenchmarkEvaluator:
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.results_data = {}
        self.comparison_stats = {}

    def load_results(self, model_name: str, results_file: str):
        if not os.path.exists(results_file):
            return
        items = []
        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        self.results_data[model_name] = items

    def stats(self, model_name: str) -> Dict:
        if model_name not in self.results_data:
            return {}
        results = self.results_data[model_name]
        ok = [r for r in results if r.get("success", False)]
        ko = [r for r in results if not r.get("success", False)]
        if not ok:
            return {
                "model_name": model_name,
                "total_questions": len(results),
                "successful_questions": 0,
                "failed_questions": len(ko),
                "success_rate": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "avg_response_time": 0.0,
                "avg_input_tokens_per_question": 0.0,
                "avg_output_tokens_per_question": 0.0,
                "total_cost_estimate": 0.0,
                "response_times": [],
            }
        tin = sum(r["input_tokens"] for r in ok)
        tout = sum(r["output_tokens"] for r in ok)
        tt = sum(r["total_tokens"] for r in ok)
        avg_t = float(np.mean([r["response_time"] for r in ok]))
        avg_in = tin / len(ok)
        avg_out = tout / len(ok)
        input_cost_per_1k = 0.002
        output_cost_per_1k = 0.006
        cost = tin / 1000 * input_cost_per_1k + tout / 1000 * output_cost_per_1k
        return {
            "model_name": model_name,
            "total_questions": len(results),
            "successful_questions": len(ok),
            "failed_questions": len(ko),
            "success_rate": len(ok) / len(results) * 100 if results else 0.0,
            "total_input_tokens": tin,
            "total_output_tokens": tout,
            "total_tokens": tt,
            "avg_response_time": avg_t,
            "avg_input_tokens_per_question": avg_in,
            "avg_output_tokens_per_question": avg_out,
            "total_cost_estimate": cost,
            "response_times": [r["response_time"] for r in ok],
        }


