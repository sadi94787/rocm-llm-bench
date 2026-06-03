#!/usr/bin/env python3
"""Quick-run entry point for ROCm LLM Benchmark Tool.

Usage:
    python bench.py --model meta-llama/Llama-3-8B
    python bench.py --model meta-llama/Llama-3-8B --mode multi --html
"""

from src.rocm_bench.cli import main

if __name__ == "__main__":
    main()
