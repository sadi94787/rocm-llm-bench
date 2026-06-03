"""Unit tests for benchmark module."""

import pytest
from unittest.mock import MagicMock, patch
from rocm_bench.benchmark import (
    BenchmarkConfig, GenerationResult, PrefillResult,
    benchmark_generation, benchmark_prefill, run_full_benchmark,
)


class TestBenchmarkConfig:
    def test_default_config(self):
        config = BenchmarkConfig()
        assert config.max_new_tokens == 128
        assert config.num_runs == 5
        assert config.warmup == 2

    def test_custom_config(self):
        config = BenchmarkConfig(max_new_tokens=256, num_runs=10)
        assert config.max_new_tokens == 256
        assert config.num_runs == 10


class TestGenerationResult:
    def test_creation(self):
        r = GenerationResult(
            run=1, generated_tokens=128, total_time_s=2.5,
            tokens_per_sec=51.2, memory_used_gb=4.5, peak_memory_gb=6.2,
        )
        assert r.run == 1
        assert r.tokens_per_sec == 51.2

    def test_asdict(self):
        from dataclasses import asdict
        r = GenerationResult(
            run=1, generated_tokens=128, total_time_s=2.5,
            tokens_per_sec=51.2, memory_used_gb=4.5, peak_memory_gb=6.2,
        )
        d = asdict(r)
        assert "tokens_per_sec" in d
        assert d["run"] == 1


class TestBenchmarkGeneration:
    def test_benchmark_runs(self):
        """Test that benchmark runs correct number of iterations."""
        mock_model = MagicMock()
        mock_model.device = "cpu"

        # Mock generate to return tensor with some generated tokens
        import torch
        mock_output = torch.zeros(1, 20)  # 20 tokens generated
        mock_model.generate.return_value = mock_output

        mock_tokenizer = MagicMock()
        mock_input = {"input_ids": torch.zeros(1, 10, dtype=torch.long)}
        mock_tokenizer.return_value = mock_input

        config = BenchmarkConfig(num_runs=2, warmup=1, max_new_tokens=10)

        with patch("torch.cuda.synchronize"), \
             patch("torch.cuda.reset_peak_memory_stats"), \
             patch("torch.cuda.memory_allocated", return_value=1e9), \
             patch("torch.cuda.max_memory_allocated", return_value=2e9):
            results = benchmark_generation(mock_model, mock_tokenizer, config)

        assert len(results) == 2
        assert all(isinstance(r, GenerationResult) for r in results)
        assert results[0].run == 1
        assert results[1].run == 2


class TestBenchmarkPrefill:
    def test_prefill_returns_result(self):
        import torch
        mock_model = MagicMock()
        mock_model.return_value = MagicMock()

        mock_tokenizer = MagicMock()
        mock_input = {"input_ids": torch.zeros(1, 10, dtype=torch.long)}
        mock_tokenizer.return_value = mock_input

        config = BenchmarkConfig(num_runs=2, warmup=1)

        with patch("torch.cuda.synchronize"):
            result = benchmark_prefill(mock_model, mock_tokenizer, config)

        assert isinstance(result, PrefillResult)
        assert result.input_tokens == 10
        assert result.avg_prefill_time_ms > 0
