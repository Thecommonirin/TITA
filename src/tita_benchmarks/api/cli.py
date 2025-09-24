import argparse
import os

from .benchmark import APIModelBenchmark, ModelConfig
from .evaluator import APIBenchmarkEvaluator


def main():
    parser = argparse.ArgumentParser(prog="tita-bench", description="API-based benchmarking for VLMs")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sample = sub.add_parser("sample", help="create sample questions file")
    p_sample.add_argument("--out", required=True)
    p_sample.add_argument("--num", type=int, default=10)

    p_run = sub.add_parser("run", help="run benchmark")
    p_run.add_argument("--api-key", required=True)
    p_run.add_argument("--model", required=True, choices=["deepseek-vl2", "qwen2.5-vl"])
    p_run.add_argument("--questions", required=True)
    p_run.add_argument("--images", default=None)
    p_run.add_argument("--out", required=True)
    p_run.add_argument("--max", type=int, default=None)

    p_eval = sub.add_parser("eval", help="evaluate results directory")
    p_eval.add_argument("--dir", required=True)

    args = parser.parse_args()

    if args.cmd == "sample":
        APIModelBenchmark.create_sample(args.out, args.num)
        return

    if args.cmd == "run":
        configs = {
            "deepseek-vl2": ModelConfig(name="DeepSeek-VL2", model_id="deepseek-ai/deepseek-vl2", api_key=args.api_key),
            "qwen2.5-vl": ModelConfig(name="Qwen2.5-VL-7B-Instruct", model_id="Qwen/Qwen2.5-VL-72B-Instruct", api_key=args.api_key),
        }
        cfg = configs[args.model]
        bench = APIModelBenchmark(cfg)
        bench.run(args.questions, args.images, args.out, args.max)
        return

    if args.cmd == "eval":
        ev = APIBenchmarkEvaluator(args.dir)
        # try load two files if exist
        for name in os.listdir(args.dir):
            if name.endswith("_results.jsonl"):
                model_name = name.replace("_results.jsonl", "")
                ev.load_results(model_name, os.path.join(args.dir, name))
        # write a simple text report
        report = []
        for model in ev.results_data.keys():
            s = ev.stats(model)
            report.append(f"== {model} ==")
            report.append(f"success_rate: {s['success_rate']:.2f}%")
            report.append(f"avg_response_time: {s['avg_response_time']:.2f}s")
            report.append(f"avg_input_tokens: {s['avg_input_tokens_per_question']:.1f}")
            report.append(f"avg_output_tokens: {s['avg_output_tokens_per_question']:.1f}")
            report.append("")
        out_path = os.path.join(args.dir, "benchmark_report.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report))
        print(f"report saved to {out_path}")


if __name__ == "__main__":
    main()


