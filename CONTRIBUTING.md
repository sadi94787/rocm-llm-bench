# Contributing to ROCm LLM Benchmark Tool

Thank you for your interest in contributing!

## Getting Started

```bash
git clone https://github.com/sadi94787/rocm-llm-bench.git
cd rocm-llm-bench
pip install -e ".[dev]"
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Linting

```bash
flake8 src/ tests/ --max-line-length=120
```

### Adding a New Benchmark

1. Add benchmark function in `src/rocm_bench/benchmark.py`
2. Add CLI argument in `src/rocm_bench/cli.py`
3. Add unit tests in `tests/`
4. Update README if needed

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Code Style

- Follow PEP 8
- Use type hints where possible
- Add docstrings to all public functions
- Keep functions focused and testable

## Reporting Issues

Please use GitHub Issues with:
- Clear description
- Steps to reproduce
- Expected vs actual behavior
- GPU model and ROCm version
