# ROCm LLM Benchmark Tool

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ROCm 6.0+](https://img.shields.io/badge/ROCm-6.0+-red.svg)](https://rocm.docs.amd.com/)

A comprehensive benchmarking toolkit for measuring LLM inference performance on AMD Instinct GPUs using ROCm.

## Features

- **Generation Benchmark** — tokens/sec, latency, memory usage
- **Prefill Benchmark** — prompt processing speed
- **Multi-GPU Scaling** — speedup ratio and parallel efficiency across 1-8 GPUs
- **HTML Reports** — beautiful reports with embedded charts
- **JSON Export** — machine-readable results for CI/comparison
- **CLI Interface** — `rocm-bench` command after install

## Architecture

```
rocm-llm-bench/
├── src/rocm_bench/
│   ├── __init__.py        # Package init
│   ├── cli.py             # CLI entry point
│   ├── benchmark.py       # Core benchmarking logic
│   ├── models.py          # Model loading utilities
│   ├── gpu.py             # GPU detection & info
│   ├── scaling.py         # Multi-GPU scaling benchmarks
│   └── report.py          # HTML report generator
├── tests/
│   ├── test_benchmark.py  # Benchmark unit tests
│   └── test_report.py     # Report generation tests
├── .github/workflows/
│   └── ci.yml             # GitHub Actions CI
├── bench.py               # Quick-run script
├── pyproject.toml         # Package configuration
└── requirements.txt       # Dependencies
```

## Quick Start

### Prerequisites

- AMD Instinct GPU (MI250, MI250X, MI300X recommended)
- ROCm 6.0+ installed ([installation guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/))
- Python 3.10+

### Install

```bash
git clone https://github.com/sadi94787/rocm-llm-bench.git
cd rocm-llm-bench
pip install -e .
```

### Run

```bash
# Single GPU benchmark
rocm-bench --model meta-llama/Llama-3-8B

# Multi-GPU scaling
rocm-bench --model meta-llama/Llama-3-8B --mode multi

# With HTML report
rocm-bench --model meta-llama/Llama-3-8B --html

# Quick run (no install)
python bench.py --model meta-llama/Llama-3-8B
```

## Benchmark Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `single` | Single GPU inference | Baseline performance |
| `multi` | Multi-GPU tensor parallel | Scaling efficiency |
| `compare` | Compare against CUDA baselines | ROCm vs CUDA analysis |
| `sweep` | Batch size / seq length sweep | Capacity planning |

## Example Results

### Generation Performance (MI300X 192GB)

| Model | Tokens/sec | Peak Memory | Prefill tok/s |
|-------|-----------|-------------|---------------|
| Llama-3-8B (FP16) | 58.3 | 16.2 GB | 1,842 |
| Llama-3-70B (FP16) | 12.4 | 138.7 GB | 412 |
| Mistral-7B (FP16) | 64.1 | 14.8 GB | 2,105 |
| Phi-3-Mini (FP16) | 89.2 | 7.4 GB | 3,218 |

### Multi-GPU Scaling (Llama-3-70B on MI300X)

| GPUs | Tokens/sec | Speedup | Efficiency |
|------|-----------|---------|------------|
| 1 | 12.4 | 1.00x | 100% |
| 2 | 23.1 | 1.86x | 93% |
| 4 | 42.8 | 3.45x | 86% |
| 8 | 78.5 | 6.33x | 79% |

## Supported Models

Any HuggingFace Transformers model compatible with ROCm:

- **Meta** — Llama 3 (8B, 70B, 405B)
- **Mistral** — Mistral 7B, Mixtral 8x7B
- **Microsoft** — Phi-3 Mini, Medium
- **Qwen** — Qwen2 7B/72B
- **Google** — Gemma 2
- Any model using `AutoModelForCausalLM`

## Output

Results are saved to `results/` directory:

```
results/
├── llama3_8b_20250101_120000.json    # Raw benchmark data
├── llama3_8b_20250101_120000.html    # Visual report (with --html)
└── comparison_chart.png              # Comparison plots
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
flake8 src/ tests/
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{rocm_llm_bench,
  title = {ROCm LLM Benchmark Tool},
  author = {sadi94787},
  year = {2025},
  url = {https://github.com/sadi94787/rocm-llm-bench}
}
```
