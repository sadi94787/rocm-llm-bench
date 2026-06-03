#!/usr/bin/env python3
"""ROCm LLM Benchmark Tool — inference performance on AMD GPUs."""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ROCm detection
ROCM_AVAILABLE = torch.version.hip is not None
DEVICE = "cuda"  # ROCm uses the same 'cuda' device type in PyTorch


def get_gpu_info():
    """Get AMD GPU information via ROCm."""
    info = {
        "rocm_available": ROCM_AVAILABLE,
        "rocm_version": torch.version.hip if ROCM_AVAILABLE else None,
        "torch_version": torch.__version__,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": [],
    }
    for i in range(info["gpu_count"]):
        props = torch.cuda.get_device_properties(i)
        info["gpus"].append({
            "name": props.name,
            "total_memory_gb": round(props.total_mem / 1e9, 2),
            "major": props.major,
            "minor": props.minor,
        })
    return info


def load_model(model_name: str, dtype=torch.float16):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def benchmark_generation(model, tokenizer, prompt: str, max_new_tokens: int = 128,
                         num_runs: int = 5, warmup: int = 2):
    """Benchmark text generation."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=10, do_sample=False)
    torch.cuda.synchronize()

    results = []
    for run in range(num_runs):
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated() / 1e9

        start = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        torch.cuda.synchronize()
        end = time.perf_counter()

        mem_after = torch.cuda.memory_allocated() / 1e9
        generated = output[0][inputs["input_ids"].shape[1]:]
        num_tokens = len(generated)
        total_time = end - start
        tokens_per_sec = num_tokens / total_time

        results.append({
            "run": run + 1,
            "generated_tokens": num_tokens,
            "total_time_s": round(total_time, 4),
            "tokens_per_sec": round(tokens_per_sec, 2),
            "memory_used_gb": round(mem_after - mem_before, 3),
            "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        })

    return results


def benchmark_prefill(model, tokenizer, prompt: str, num_runs: int = 5, warmup: int = 2):
    """Benchmark prefill (prompt processing) latency."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    for _ in range(warmup):
        with torch.no_grad():
            model(**inputs)
    torch.cuda.synchronize()

    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            model(**inputs)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times)
    return {
        "input_tokens": input_len,
        "avg_prefill_time_ms": round(avg_time * 1000, 2),
        "prefill_tokens_per_sec": round(input_len / avg_time, 2),
        "min_ms": round(min(times) * 1000, 2),
        "max_ms": round(max(times) * 1000, 2),
    }


def run_benchmark(args):
    """Run full benchmark suite."""
    gpu_info = get_gpu_info()
    print(f"\n{'='*60}")
    print(f"ROCm LLM Benchmark Tool")
    print(f"{'='*60}")
    print(f"GPU: {gpu_info['gpus'][0]['name'] if gpu_info['gpus'] else 'N/A'}")
    print(f"ROCm: {gpu_info['rocm_version'] or 'N/A'}")
    print(f"PyTorch: {gpu_info['torch_version']}")
    print(f"Model: {args.model}")
    print(f"{'='*60}\n")

    model, tokenizer = load_model(args.model, dtype=torch.float16)

    prompt = args.prompt or "Explain the theory of relativity in simple terms."

    print("Running generation benchmark...")
    gen_results = benchmark_generation(
        model, tokenizer, prompt,
        max_new_tokens=args.max_tokens,
        num_runs=args.runs,
    )

    print("Running prefill benchmark...")
    prefill_results = benchmark_prefill(model, tokenizer, prompt, num_runs=args.runs)

    # Aggregate
    avg_tps = sum(r["tokens_per_sec"] for r in gen_results) / len(gen_results)
    avg_mem = sum(r["peak_memory_gb"] for r in gen_results) / len(gen_results)

    report = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "device": gpu_info,
        "config": {
            "max_new_tokens": args.max_tokens,
            "num_runs": args.runs,
            "warmup": 2,
        },
        "generation": gen_results,
        "prefill": prefill_results,
        "summary": {
            "avg_tokens_per_sec": round(avg_tps, 2),
            "avg_peak_memory_gb": round(avg_mem, 3),
            "prefill_tps": prefill_results["prefill_tokens_per_sec"],
        },
    }

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    model_short = args.model.split("/")[-1].replace("-", "_").lower()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = results_dir / f"{model_short}_{ts}.json"

    with open(filename, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results Summary")
    print(f"{'='*60}")
    print(f"Avg tokens/sec:  {avg_tps:.2f}")
    print(f"Avg peak memory: {avg_mem:.3f} GB")
    print(f"Prefill tps:     {prefill_results['prefill_tokens_per_sec']:.2f}")
    print(f"Results saved:   {filename}")
    print(f"{'='*60}\n")

    return report


def main():
    parser = argparse.ArgumentParser(description="ROCm LLM Benchmark Tool")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max new tokens")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt")
    parser.add_argument("--gpu", type=str, default=None, help="Target GPU (mi250, mi300x)")
    parser.add_argument("--mode", choices=["single", "multi", "compare", "sweep"],
                        default="single", help="Benchmark mode")
    args = parser.parse_args()

    run_benchmark(args)


if __name__ == "__main__":
    main()
