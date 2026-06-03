# ROCm LLM Benchmark Tool

A lightweight benchmarking toolkit for measuring LLM inference performance on AMD GPUs using ROCm.

## Features

- Benchmark popular LLM models (Llama, Mistral, Phi) on AMD Instinct GPUs
- Measure tokens/sec, latency (TTFT, inter-token), and memory usage
- Compare ROCm vs CUDA performance profiles
- Support for multiple quantization formats (FP16, INT8, INT4)
- HTML report generation with charts
- Multi-GPU scaling benchmarks

## Requirements

- AMD Instinct GPU (MI250, MI300X recommended)
- ROCm 6.0+
- PyTorch 2.3+ (ROCm build)
- Python 3.10+

## Quick Start

```bash
pip install -r requirements.txt
python bench.py --model meta-llama/Llama-3-8B --gpu mi300x
```

## Benchmark Modes

| Mode | Description |
|------|-------------|
| `single` | Single GPU inference benchmark |
| `multi` | Multi-GPU tensor parallel benchmark |
| `compare` | Compare against published CUDA benchmarks |
| `sweep` | Sweep across batch sizes and sequence lengths |

## Output

Results are saved to `results/` as JSON and optional HTML reports.

```
results/
├── llama3_8b_mi300x_20250101.json
├── report.html
└── comparison_chart.png
```

## Supported Models

- Meta Llama 3 (8B, 70B)
- Mistral 7B / Mixtral 8x7B
- Phi-3 Mini/Medium
- Qwen2 7B/72B
- Any HuggingFace model compatible with ROCm

## License

MIT
