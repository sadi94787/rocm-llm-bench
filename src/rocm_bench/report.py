"""HTML report generator with embedded charts."""

import base64
import io
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_charts(benchmark_data: dict) -> dict[str, str]:
    """Generate base64-encoded charts from benchmark data."""
    charts = {}

    # Tokens/sec bar chart
    gen = benchmark_data.get("generation", [])
    if gen:
        fig, ax = plt.subplots(figsize=(8, 4))
        runs = [f"Run {r['run']}" for r in gen]
        tps = [r["tokens_per_sec"] for r in gen]
        colors = ["#e74c3c" if t == min(tps) else "#2ecc71" if t == max(tps) else "#3498db" for t in tps]
        ax.bar(runs, tps, color=colors)
        ax.set_ylabel("Tokens/sec")
        ax.set_title("Generation Throughput")
        ax.axhline(y=sum(tps)/len(tps), color="#e67e22", linestyle="--", label=f"Avg: {sum(tps)/len(tps):.1f}")
        ax.legend()
        charts["generation_tps"] = _fig_to_base64(fig)
        plt.close(fig)

    # Memory usage chart
    if gen:
        fig, ax = plt.subplots(figsize=(8, 4))
        mem = [r["peak_memory_gb"] for r in gen]
        ax.bar(runs, mem, color="#9b59b6")
        ax.set_ylabel("Peak Memory (GB)")
        ax.set_title("GPU Memory Usage")
        charts["memory"] = _fig_to_base64(fig)
        plt.close(fig)

    # Scaling chart (if available)
    scaling = benchmark_data.get("scaling", [])
    if scaling:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        gpus = [s["gpu_count"] for s in scaling]
        tps = [s["tokens_per_sec"] for s in scaling]
        eff = [s["efficiency_pct"] for s in scaling]

        ax1.plot(gpus, tps, "o-", color="#2ecc71", linewidth=2, markersize=8)
        ax1.set_xlabel("GPU Count")
        ax1.set_ylabel("Tokens/sec")
        ax1.set_title("Scaling Performance")
        ax1.grid(True, alpha=0.3)

        ax2.bar([str(g) for g in gpus], eff, color="#3498db")
        ax2.set_xlabel("GPU Count")
        ax2.set_ylabel("Efficiency (%)")
        ax2.set_title("Parallel Efficiency")
        ax2.axhline(y=100, color="#e74c3c", linestyle="--", alpha=0.5)
        ax2.set_ylim(0, 110)

        charts["scaling"] = _fig_to_base64(fig)
        plt.close(fig)

    return charts


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def generate_html_report(benchmark_data: dict, output_path: str = "results/report.html"):
    """Generate full HTML report with charts."""
    charts = generate_charts(benchmark_data)
    summary = benchmark_data.get("summary", {})
    device = benchmark_data.get("device", {})
    model_name = benchmark_data.get("model", "Unknown")
    timestamp = benchmark_data.get("timestamp", datetime.now().isoformat())

    gen_rows = ""
    for r in benchmark_data.get("generation", []):
        gen_rows += f"""
        <tr>
            <td>{r['run']}</td>
            <td>{r['generated_tokens']}</td>
            <td>{r['total_time_s']}s</td>
            <td><strong>{r['tokens_per_sec']}</strong></td>
            <td>{r['peak_memory_gb']} GB</td>
        </tr>"""

    scaling_rows = ""
    for s in benchmark_data.get("scaling", []):
        scaling_rows += f"""
        <tr>
            <td>{s['gpu_count']}</td>
            <td>{s['tokens_per_sec']}</td>
            <td>{s['speedup_ratio']}x</td>
            <td>{s['efficiency_pct']}%</td>
            <td>{s['peak_memory_gb']} GB</td>
        </tr>"""

    gpu_name = device.get("gpus", [{}])[0].get("name", "N/A") if device else "N/A"
    rocm_ver = device.get("rocm_version", "N/A") if device else "N/A"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ROCm LLM Benchmark Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        h1 {{ color: #e74c3c; border-bottom: 3px solid #e74c3c; padding-bottom: 10px; }}
        h2 {{ color: #2c3e50; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #e74c3c; color: white; }}
        tr:hover {{ background: #f9f9f9; }}
        .summary {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0; }}
        .card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
        .card .value {{ font-size: 2em; font-weight: bold; color: #e74c3c; }}
        .card .label {{ color: #666; margin-top: 5px; }}
        .chart {{ text-align: center; margin: 20px 0; }}
        .chart img {{ max-width: 100%; border-radius: 4px; }}
        .meta {{ color: #888; font-size: 0.9em; }}
    </style>
</head>
<body>
<div class="container">
    <h1>ROCm LLM Benchmark Report</h1>
    <p class="meta">Generated: {timestamp} | Model: {model_name}</p>
    <p class="meta">GPU: {gpu_name} | ROCm: {rocm_ver}</p>

    <h2>Summary</h2>
    <div class="summary">
        <div class="card">
            <div class="value">{summary.get('avg_tokens_per_sec', 'N/A')}</div>
            <div class="label">Avg Tokens/sec</div>
        </div>
        <div class="card">
            <div class="value">{summary.get('avg_peak_memory_gb', 'N/A')} GB</div>
            <div class="label">Peak Memory</div>
        </div>
        <div class="card">
            <div class="value">{summary.get('prefill_tps', 'N/A')}</div>
            <div class="label">Prefill tok/s</div>
        </div>
    </div>

    <h2>Generation Benchmark</h2>
    <table>
        <tr><th>Run</th><th>Tokens</th><th>Time</th><th>Tokens/sec</th><th>Peak Memory</th></tr>
        {gen_rows}
    </table>

    {"<h2>Generation Throughput</h2><div class='chart'><img src='data:image/png;base64," + charts.get("generation_tps", "") + "'></div>" if "generation_tps" in charts else ""}

    {"<h2>Memory Usage</h2><div class='chart'><img src='data:image/png;base64," + charts.get("memory", "") + "'></div>" if "memory" in charts else ""}

    {"<h2>Multi-GPU Scaling</h2><table><tr><th>GPUs</th><th>Tokens/sec</th><th>Speedup</th><th>Efficiency</th><th>Memory</th></tr>" + scaling_rows + "</table>" if scaling_rows else ""}

    {"<div class='chart'><img src='data:image/png;base64," + charts.get("scaling", "") + "'></div>" if "scaling" in charts else ""}
</div>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    return output_path
