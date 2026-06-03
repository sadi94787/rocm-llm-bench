"""Core benchmarking logic for LLM inference on AMD GPUs."""

import time
import torch
from dataclasses import dataclass, asdict


@dataclass
class GenerationResult:
    run: int
    generated_tokens: int
    total_time_s: float
    tokens_per_sec: float
    memory_used_gb: float
    peak_memory_gb: float


@dataclass
class PrefillResult:
    input_tokens: int
    avg_prefill_time_ms: float
    prefill_tokens_per_sec: float
    min_ms: float
    max_ms: float


@dataclass
class BenchmarkConfig:
    max_new_tokens: int = 128
    num_runs: int = 5
    warmup: int = 2
    prompt: str = "Explain the theory of relativity in simple terms."


def benchmark_generation(model, tokenizer, config: BenchmarkConfig):
    """Benchmark text generation throughput and latency."""
    inputs = tokenizer(config.prompt, return_tensors="pt").to(model.device)

    # Warmup
    for _ in range(config.warmup):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=10, do_sample=False)
    torch.cuda.synchronize()

    results = []
    for run in range(config.num_runs):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=config.max_new_tokens, do_sample=False
            )
        torch.cuda.synchronize()
        end = time.perf_counter()

        generated = output[0][inputs["input_ids"].shape[1]:]
        num_tokens = len(generated)
        total_time = end - start

        results.append(GenerationResult(
            run=run + 1,
            generated_tokens=num_tokens,
            total_time_s=round(total_time, 4),
            tokens_per_sec=round(num_tokens / total_time, 2),
            memory_used_gb=round(torch.cuda.memory_allocated() / 1e9, 3),
            peak_memory_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3),
        ))

    return results


def benchmark_prefill(model, tokenizer, config: BenchmarkConfig):
    """Benchmark prefill (prompt processing) latency."""
    inputs = tokenizer(config.prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    for _ in range(config.warmup):
        with torch.no_grad():
            model(**inputs)
    torch.cuda.synchronize()

    times = []
    for _ in range(config.num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            model(**inputs)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times)
    return PrefillResult(
        input_tokens=input_len,
        avg_prefill_time_ms=round(avg_time * 1000, 2),
        prefill_tokens_per_sec=round(input_len / avg_time, 2),
        min_ms=round(min(times) * 1000, 2),
        max_ms=round(max(times) * 1000, 2),
    )


def run_full_benchmark(model, tokenizer, config: BenchmarkConfig):
    """Run complete benchmark suite (generation + prefill)."""
    gen_results = benchmark_generation(model, tokenizer, config)
    prefill_result = benchmark_prefill(model, tokenizer, config)

    avg_tps = sum(r.tokens_per_sec for r in gen_results) / len(gen_results)
    avg_mem = sum(r.peak_memory_gb for r in gen_results) / len(gen_results)

    return {
        "generation": [asdict(r) for r in gen_results],
        "prefill": asdict(prefill_result),
        "summary": {
            "avg_tokens_per_sec": round(avg_tps, 2),
            "avg_peak_memory_gb": round(avg_mem, 3),
            "prefill_tps": prefill_result.prefill_tokens_per_sec,
        },
    }
