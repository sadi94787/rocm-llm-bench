"""CLI entry point for ROCm LLM Benchmark Tool."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from .gpu import get_system_info, print_system_info
from .models import load_model, get_model_info
from .benchmark import BenchmarkConfig, run_full_benchmark
from .scaling import benchmark_scaling, print_scaling_report
from .report import generate_html_report


def main():
    parser = argparse.ArgumentParser(
        prog="rocm-bench",
        description="ROCm LLM Benchmark Tool — measure inference performance on AMD GPUs",
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max new tokens to generate (default: 128)")
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark runs (default: 5)")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations (default: 2)")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt text")
    parser.add_argument("--mode", choices=["single", "multi", "compare", "sweep"],
                        default="single", help="Benchmark mode (default: single)")
    parser.add_argument("--gpu", type=str, default=None, help="Target GPU type (mi250, mi300x)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--html", action="store_true", help="Generate HTML report")
    parser.add_argument("--info", action="store_true", help="Print system info and exit")
    args = parser.parse_args()

    if args.info:
        print_system_info()
        return

    print(f"\n{'='*60}")
    print(f"ROCm LLM Benchmark Tool v0.2.0")
    print(f"{'='*60}")
    print_system_info()
    print(f"Model:  {args.model}")
    print(f"Mode:   {args.mode}")
    print(f"{'='*60}\n")

    config = BenchmarkConfig(
        max_new_tokens=args.max_tokens,
        num_runs=args.runs,
        warmup=args.warmup,
        prompt=args.prompt or "Explain the theory of relativity in simple terms.",
    )

    model, tokenizer = load_model(args.model)
    device_info = get_system_info()
    model_info = get_model_info(model)

    # Run benchmarks
    result = run_full_benchmark(model, tokenizer, config)
    result["model"] = args.model
    result["timestamp"] = datetime.now().isoformat()
    result["device"] = device_info
    result["model_info"] = model_info

    # Scaling benchmark
    if args.mode == "multi":
        print("Running multi-GPU scaling benchmark...")
        scaling = benchmark_scaling(model, tokenizer, config)
        result["scaling"] = scaling
        print_scaling_report(scaling)

    # Summary
    s = result["summary"]
    print(f"\n{'='*60}")
    print(f"Results Summary")
    print(f"{'='*60}")
    print(f"Avg tokens/sec:  {s['avg_tokens_per_sec']:.2f}")
    print(f"Avg peak memory: {s['avg_peak_memory_gb']:.3f} GB")
    print(f"Prefill tps:     {s['prefill_tps']:.2f}")

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    model_short = args.model.split("/")[-1].replace("-", "_").lower()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.output:
        json_path = args.output
    else:
        json_path = str(results_dir / f"{model_short}_{ts}.json")

    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"JSON report:     {json_path}")

    if args.html:
        html_path = str(results_dir / f"{model_short}_{ts}.html")
        generate_html_report(result, html_path)
        print(f"HTML report:     {html_path}")

    print(f"{'='*60}\n")
    return result


if __name__ == "__main__":
    main()
