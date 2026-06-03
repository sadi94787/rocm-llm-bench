"""Internal CLI implementation for rocm-bench."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def run_single_benchmark(args):
    """Run single-GPU benchmark."""
    import torch
    from rocm_bench.benchmark import benchmark_generation, benchmark_prefill, aggregate_results
    from rocm_bench.gpu import get_gpu_info
    from rocm_bench.models import load_model_simple
    from rocm_bench.report import generate_html_report

    gpu_info = get_gpu_info()
    print("\n" + "=" * 60)
    print("ROCm LLM Benchmark Tool")
    print("=" * 60)
    print("GPU: {}".format(gpu_info.gpus[0].name if gpu_info.gpus else "N/A"))
    print("ROCm: {}".format(gpu_info.rocm_version or "N/A"))
    print("PyTorch: {}".format(gpu_info.torch_version))
    print("Model: {}".format(args.model))
    print("=" * 60 + "\n")

    model, tokenizer = load_model_simple(args.model)
    prompt = args.prompt or "Explain the theory of relativity in simple terms."

    print("Running generation benchmark...")
    gen_results = benchmark_generation(
        model, tokenizer, prompt,
        max_new_tokens=args.max_tokens,
        num_runs=args.runs,
    )

    print("Running prefill benchmark...")
    prefill_results = benchmark_prefill(model, tokenizer, prompt, num_runs=args.runs)

    summary = aggregate_results(gen_results, prefill_results)

    report = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "device": gpu_info.to_dict(),
        "config": {
            "max_new_tokens": args.max_tokens,
            "num_runs": args.runs,
            "warmup": 2,
        },
        "generation": [
            {
                "run": r.run,
                "generated_tokens": r.generated_tokens,
                "total_time_s": r.total_time_s,
                "tokens_per_sec": r.tokens_per_sec,
                "memory_used_gb": r.memory_used_gb,
                "peak_memory_gb": r.peak_memory_gb,
            }
            for r in gen_results
        ],
        "prefill": {
            "input_tokens": prefill_results.input_tokens,
            "avg_prefill_time_ms": prefill_results.avg_prefill_time_ms,
            "prefill_tokens_per_sec": prefill_results.prefill_tokens_per_sec,
            "min_ms": prefill_results.min_ms,
            "max_ms": prefill_results.max_ms,
        },
        "summary": summary,
    }

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    model_short = args.model.split("/")[-1].replace("-", "_").lower()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = results_dir / "{}_{}.json".format(model_short, ts)
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    html_path = results_dir / "{}_{}.html".format(model_short, ts)
    generate_html_report(report, str(html_path))

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print("Avg tokens/sec:  {:.2f}".format(summary["avg_tokens_per_sec"]))
    print("Avg peak memory: {:.3f} GB".format(summary["avg_peak_memory_gb"]))
    print("Prefill tps:     {:.2f}".format(prefill_results.prefill_tokens_per_sec))
    print("Results saved:   {}".format(json_path))
    print("HTML report:     {}".format(html_path))
    print("=" * 60 + "\n")

    return report


def run_scaling(args):
    """Run multi-GPU scaling benchmark."""
    from rocm_bench.scaling import run_scaling_benchmark, print_scaling_report
    from rocm_bench.report import generate_html_report

    report = run_scaling_benchmark(
        model_name=args.model,
        prompt=args.prompt or "Explain the theory of relativity in simple terms.",
        max_new_tokens=args.max_tokens,
        num_runs=args.runs,
    )

    print(print_scaling_report(report))

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    model_short = args.model.split("/")[-1].replace("-", "_").lower()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    benchmark_data = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "device": report.gpu_info or {},
        "config": {"max_new_tokens": args.max_tokens, "num_runs": args.runs, "warmup": 2},
        "generation": [],
        "prefill": {},
        "summary": {},
    }

    scaling_data = {
        "results": [
            {
                "num_gpus": r.num_gpus,
                "tokens_per_sec": r.tokens_per_sec,
                "latency_s": r.latency_s,
                "peak_memory_per_gpu_gb": r.peak_memory_per_gpu_gb,
                "speedup_ratio": r.speedup_ratio,
                "efficiency_pct": r.efficiency_pct,
            }
            for r in report.results
        ]
    }

    html_path = results_dir / "{}_scaling_{}.html".format(model_short, ts)
    generate_html_report(benchmark_data, str(html_path), scaling_data=scaling_data)
    print("Scaling report saved to: {}".format(html_path))


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ROCm LLM Benchmark Tool - inference performance on AMD GPUs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  rocm-bench --model meta-llama/Llama-3-8B
  rocm-bench --model meta-llama/Llama-3-8B --mode scaling --runs 10
        """,
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max new tokens (default: 128)")
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark runs (default: 5)")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt")
    parser.add_argument(
        "--mode", choices=["single", "scaling"], default="single",
        help="Benchmark mode (default: single)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--json", action="store_true", help="Output JSON to stdout")

    args = parser.parse_args()

    if args.output:
        Path(args.output).mkdir(parents=True, exist_ok=True)

    if args.mode == "single":
        report = run_single_benchmark(args)
        if args.json:
            print(json.dumps(report, indent=2))
    elif args.mode == "scaling":
        run_scaling(args)
