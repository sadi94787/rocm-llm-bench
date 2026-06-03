"""Unit tests for report generator."""

import os
import pytest
from rocm_bench.report import generate_html_report, generate_charts


SAMPLE_DATA = {
    "timestamp": "2025-01-01T00:00:00",
    "model": "test-model",
    "device": {
        "rocm_version": "6.0.0",
        "gpus": [{"name": "Test GPU", "total_memory_gb": 32}],
    },
    "generation": [
        {"run": 1, "generated_tokens": 128, "total_time_s": 2.5,
         "tokens_per_sec": 51.2, "memory_used_gb": 4.5, "peak_memory_gb": 6.2},
        {"run": 2, "generated_tokens": 128, "total_time_s": 2.3,
         "tokens_per_sec": 55.7, "memory_used_gb": 4.5, "peak_memory_gb": 6.3},
    ],
    "prefill": {"input_tokens": 50, "avg_prefill_time_ms": 120.5,
                "prefill_tokens_per_sec": 414.9, "min_ms": 110.0, "max_ms": 130.0},
    "summary": {"avg_tokens_per_sec": 53.45, "avg_peak_memory_gb": 6.25, "prefill_tps": 414.9},
}


class TestGenerateCharts:
    def test_generation_chart(self):
        charts = generate_charts(SAMPLE_DATA)
        assert "generation_tps" in charts
        assert len(charts["generation_tps"]) > 100  # base64 string

    def test_memory_chart(self):
        charts = generate_charts(SAMPLE_DATA)
        assert "memory" in charts

    def test_scaling_chart(self):
        data = {**SAMPLE_DATA, "scaling": [
            {"gpu_count": 1, "tokens_per_sec": 50, "speedup_ratio": 1.0,
             "efficiency_pct": 100, "peak_memory_gb": 6},
            {"gpu_count": 2, "tokens_per_sec": 95, "speedup_ratio": 1.9,
             "efficiency_pct": 95, "peak_memory_gb": 6},
        ]}
        charts = generate_charts(data)
        assert "scaling" in charts

    def test_empty_data(self):
        charts = generate_charts({})
        assert charts == {}


class TestGenerateHtmlReport:
    def test_creates_file(self, tmp_path):
        output = str(tmp_path / "report.html")
        result = generate_html_report(SAMPLE_DATA, output)
        assert os.path.exists(result)
        assert result == output

    def test_contains_summary(self, tmp_path):
        output = str(tmp_path / "report.html")
        generate_html_report(SAMPLE_DATA, output)
        with open(output) as f:
            content = f.read()
        assert "53.45" in content  # avg tokens/sec
        assert "test-model" in content

    def test_contains_generation_table(self, tmp_path):
        output = str(tmp_path / "report.html")
        generate_html_report(SAMPLE_DATA, output)
        with open(output) as f:
            content = f.read()
        assert "51.2" in content
        assert "55.7" in content

    def test_creates_parent_dirs(self, tmp_path):
        output = str(tmp_path / "sub" / "dir" / "report.html")
        result = generate_html_report(SAMPLE_DATA, output)
        assert os.path.exists(result)
