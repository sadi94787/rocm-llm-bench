"""Multi-GPU scaling benchmarks for AMD ROCm systems."""

import time
import json
import torch
from dataclasses import dataclass, asdict
from .benchmark import BenchmarkConfig


@dataclass
class ScalingResult:
    gpu_count: int
    tokens_per_sec: float
    peak_memory_gb: float
    speedup_ratio: float
    efficiency_pct: float
    total_time_s: float


def benchmark_scaling(model, tokenizer, config: BenchmarkConfig,
                      gpu_counts: list = None):
    """Benchmark inference across different GPU counts.

    Measures speedup ratio and parallel efficiency for multi-GPU
    tensor parallel inference on AMD Instinct GPUs.
    """
    available_gpus = torch.cuda.device_count()
    if gpu_counts is None:
        gpu_counts = [2 ** i for i in range(available_gpus.bit_length())
                      if 2 ** i <= available_gpus]
        if not gpu_counts:
            gpu_counts = [1]

    inputs = tokenizer(config.prompt, return_tensors="pt")
    base_tps = None
    results = []

    for num_gpus in gpu_counts:
        if num_gpus > available_gpus:
            continue

        device_map = _get_device_map(num_gpus)

        # Warmup
        for _ in range(config.warmup):
            with torch.no_grad():
                input_ids = inputs["input_ids"].to(f"cuda:{0}")
                model.generate(input_ids, max_new_tokens=10, do_sample=False)
        torch.cuda.synchronize()

        # Benchmark
        total_tokens = 0
        total_time = 0
        for _ in range(config.num_runs):
            input_ids = inputs["input_ids"].to(f"cuda:{0}")
            torch.cuda.synchronize()

            start = time.perf_counter()
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=False,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            generated = output[0][inputs["input_ids"].shape[1]:]
            total_tokens += len(generated)
            total_time += elapsed

        avg_tps = total_tokens / total_time
        peak_mem = torch.cuda.max_memory_allocated() / 1e9

        if base_tps is None:
            base_tps = avg_tps

        speedup = avg_tps / base_tps if base_tps else 1.0
        efficiency = (speedup / num_gpus) * 100

        results.append(ScalingResult(
            gpu_count=num_gpus,
            tokens_per_sec=round(avg_tps, 2),
            peak_memory_gb=round(peak_mem, 3),
            speedup_ratio=round(speedup, 2),
            efficiency_pct=round(efficiency, 1),
            total_time_s=round(total_time, 2),
        ))

    return [asdict(r) for r in results]


def _get_device_map(num_gpus: int):
    """Generate device map for multi-GPU inference."""
    if num_gpus <= 1:
        return "auto"
    return {i: f"cuda:{i}" for i in range(num_gpus)}


def print_scaling_report(results: list):
    """Print formatted scaling results."""
    print(f"\n{'='*70}")
    print(f"Multi-GPU Scaling Results")
    print(f"{'='*70}")
    print(f"{'GPUs':>4} | {'tok/s':>8} | {'Speedup':>7} | {'Efficiency':>10} | {'Memory':>8}")
    print(f"{'-'*4}-+-{'-'*8}-+-{'-'*7}-+-{'-'*10}-+-{'-'*8}")
    for r in results:
        print(f"{r['gpu_count']:>4} | {r['tokens_per_sec']:>8.2f} | "
              f"{r['speedup_ratio']:>6.2f}x | {r['efficiency_pct']:>9.1f}% | "
              f"{r['peak_memory_gb']:>6.2f}GB")
    print(f"{'='*70}\n")
