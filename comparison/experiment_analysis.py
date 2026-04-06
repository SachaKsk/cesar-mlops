"""
Model experiment comparison and analysis.

Compare multiple trained models by:
- Loading experiment run logs
- Computing statistical comparisons
- Generating performance reports
- Identifying best models by various criteria

Usage:
    from comparison.experiment_analysis import analyze_runs, compare_model_pairs
    
    # Analyze all past runs
    report = analyze_runs()
    print(report)
    
    # Compare two specific models
    comparison = compare_model_pairs("v1_baseline", "v2_cleaned")
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import statistics

from training.experiment_log import list_runs, DEFAULT_LOG_FILE


def _parse_metrics(metrics_str: str | dict) -> dict:
    """Parse metrics from string (JSON) or dict."""
    if isinstance(metrics_str, dict):
        return metrics_str
    if isinstance(metrics_str, str):
        try:
            return json.loads(metrics_str)
        except:
            return {}
    return {}


def get_run_by_version(version: str) -> Optional[Dict[str, Any]]:
    """Find a specific run by model version."""
    runs = list_runs()
    for run in runs:
        if run.get("model_version") == version:
            return run
    return None


def analyze_runs() -> Dict[str, Any]:
    """
    Analyze all experiment runs and generate a comprehensive report.
    
    Returns:
        Dictionary with:
        - summary: best models by various criteria
        - metrics_table: all runs with parsed metrics
        - improvements: performance changes over time
    """
    runs = list_runs()
    
    if not runs:
        return {"error": "No runs found in experiment log"}
    
    # Parse metrics for all runs
    runs_with_metrics = []
    for run in runs:
        metrics = _parse_metrics(run.get("metrics", {}))
        runs_with_metrics.append({
            **run,
            "metrics_parsed": metrics,
        })
    
    # Find best models by different criteria
    summary = {}
    
    # Best by test R²
    by_r2 = sorted(
        [r for r in runs_with_metrics if r["metrics_parsed"].get("test_r2")],
        key=lambda x: x["metrics_parsed"]["test_r2"],
        reverse=True,
    )
    if by_r2:
        best_r2 = by_r2[0]
        summary["best_by_r2"] = {
            "version": best_r2["model_version"],
            "test_r2": best_r2["metrics_parsed"]["test_r2"],
            "test_mae": best_r2["metrics_parsed"].get("test_mae", 0),
            "variant": best_r2["notes"],
        }
    
    # Best by test MAE (lowest is better)
    by_mae = sorted(
        [r for r in runs_with_metrics if r["metrics_parsed"].get("test_mae")],
        key=lambda x: x["metrics_parsed"]["test_mae"],
    )
    if by_mae:
        best_mae = by_mae[0]
        summary["best_by_mae"] = {
            "version": best_mae["model_version"],
            "test_r2": best_mae["metrics_parsed"].get("test_r2", 0),
            "test_mae": best_mae["metrics_parsed"]["test_mae"],
            "variant": best_mae["notes"],
        }
    
    # Best by train/test gap (overfitting indicator, lower is better)
    gap_models = []
    for r in runs_with_metrics:
        m = r["metrics_parsed"]
        if m.get("train_r2") is not None and m.get("test_r2") is not None:
            gap = abs(m["train_r2"] - m["test_r2"])
            gap_models.append({"run": r, "gap": gap})
    
    if gap_models:
        gap_models.sort(key=lambda x: x["gap"])
        best_gap = gap_models[0]
        summary["best_generalization"] = {
            "version": best_gap["run"]["model_version"],
            "train_r2": best_gap["run"]["metrics_parsed"]["train_r2"],
            "test_r2": best_gap["run"]["metrics_parsed"]["test_r2"],
            "gap": best_gap["gap"],
            "variant": best_gap["run"]["notes"],
        }
    
    # Improvement over baseline (first run)
    if len(runs_with_metrics) > 1 and by_r2 and by_mae:
        baseline = runs_with_metrics[0]
        baseline_metrics = baseline["metrics_parsed"]
        best_r2_metrics = by_r2[0]["metrics_parsed"]
        best_mae_metrics = by_mae[0]["metrics_parsed"]
        
        if baseline_metrics.get("test_r2"):
            r2_improvement = (
                (best_r2_metrics.get("test_r2", 0) - baseline_metrics["test_r2"]) 
                / abs(baseline_metrics["test_r2"]) * 100
            )
        else:
            r2_improvement = 0
        
        if baseline_metrics.get("test_mae"):
            mae_improvement = (
                (baseline_metrics["test_mae"] - best_mae_metrics.get("test_mae", 0)) 
                / baseline_metrics["test_mae"] * 100
            )
        else:
            mae_improvement = 0
        
        summary["improvement_over_baseline"] = {
            "r2_improvement_pct": r2_improvement,
            "mae_improvement_pct": mae_improvement,
        }
    
    return {
        "summary": summary,
        "num_runs": len(runs),
        "metrics_table": runs_with_metrics,
        "best_by_r2": by_r2,
        "best_by_mae": by_mae,
    }


def compare_model_pairs(version_a: str, version_b: str) -> Dict[str, Any]:
    """
    Compare two specific model versions.
    
    Returns:
        Dictionary with side-by-side metrics and relative improvements.
    """
    run_a = get_run_by_version(version_a)
    run_b = get_run_by_version(version_b)
    
    if not run_a or not run_b:
        return {"error": "One or both versions not found in experiment log"}
    
    metrics_a = _parse_metrics(run_a.get("metrics", {}))
    metrics_b = _parse_metrics(run_b.get("metrics", {}))
    
    comparison = {
        "version_a": version_a,
        "version_b": version_b,
        "variant_a": run_a["notes"],
        "variant_b": run_b["notes"],
        "rows_a": run_a.get("train_rows", "N/A"),
        "rows_b": run_b.get("train_rows", "N/A"),
        "metrics_a": metrics_a,
        "metrics_b": metrics_b,
    }
    
    # Compute relative differences
    differences = {}
    for key in ["test_r2", "test_mae", "train_r2", "train_mae"]:
        val_a = metrics_a.get(key)
        val_b = metrics_b.get(key)
        if val_a is not None and val_b is not None:
            if key.endswith("_mae"):
                # For MAE, lower is better
                diff_pct = (val_b - val_a) / abs(val_a) * 100 if val_a != 0 else 0
                winner = "B" if val_b < val_a else "A"
            else:
                # For R², higher is better
                diff_pct = (val_b - val_a) / abs(val_a) * 100 if val_a != 0 else 0
                winner = "B" if val_b > val_a else "A"
            
            differences[key] = {
                "value_a": val_a,
                "value_b": val_b,
                "diff_pct": diff_pct,
                "winner": winner,
            }
    
    comparison["differences"] = differences
    return comparison


def generate_html_report(output_path: str | Path = "experiment_report.html") -> None:
    """
    Generate an HTML report of all experiment runs and comparisons.
    Opens in browser for easy viewing.
    """
    analysis = analyze_runs()
    
    if "error" in analysis:
        print(f"Cannot generate report: {analysis['error']}")
        return
    
    summary = analysis["summary"]
    runs = analysis["metrics_table"]
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>CESAR Model Experiment Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #0066cc; padding-bottom: 10px; }}
        h2 {{ color: #0066cc; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background: #0066cc; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background: #f9f9f9; }}
        .metric {{ font-weight: bold; color: #0066cc; }}
        .best {{ background: #e8f4f8; }}
        .improvement {{ color: green; font-weight: bold; }}
        .degradation {{ color: red; font-weight: bold; }}
        .summary-card {{ background: #f9f9f9; padding: 15px; margin: 10px 0; border-left: 4px solid #0066cc; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 CESAR Model Experiment Report</h1>
        <p>Total runs analyzed: <strong>{len(runs)}</strong></p>
"""
    
    # Best models summary
    if summary:
        html += "<h2>📊 Best Models</h2>"
        
        if "best_by_r2" in summary:
            best = summary["best_by_r2"]
            html += f"""
        <div class="summary-card">
            <strong>✓ Best R² Score:</strong> {best['variant']}<br>
            Version: <code>{best['version']}</code><br>
            Test R²: <span class="metric">{best['test_r2']:.4f}</span> | Test MAE: €{best['test_mae']:,.0f}
        </div>
"""
        
        if "best_by_mae" in summary:
            best = summary["best_by_mae"]
            html += f"""
        <div class="summary-card">
            <strong>✓ Best MAE (Lowest Error):</strong> {best['variant']}<br>
            Version: <code>{best['version']}</code><br>
            Test R²: <span class="metric">{best['test_r2']:.4f}</span> | Test MAE: €{best['test_mae']:,.0f}
        </div>
"""
        
        if "best_generalization" in summary:
            best = summary["best_generalization"]
            html += f"""
        <div class="summary-card">
            <strong>✓ Best Generalization (Train/Test Gap):</strong> {best['variant']}<br>
            Version: <code>{best['version']}</code><br>
            Train R²: {best['train_r2']:.4f} | Test R²: {best['test_r2']:.4f} | Gap: {best['gap']:.4f}
        </div>
"""
        
        if "improvement_over_baseline" in summary:
            imp = summary["improvement_over_baseline"]
            html += f"""
        <div class="summary-card">
            <strong>📈 Improvement Over Baseline:</strong><br>
            R² Improvement: <span class="{'improvement' if imp['r2_improvement_pct'] > 0 else 'degradation'}">{imp['r2_improvement_pct']:+.2f}%</span><br>
            MAE Improvement: <span class="{'improvement' if imp['mae_improvement_pct'] > 0 else 'degradation'}">{imp['mae_improvement_pct']:+.2f}%</span>
        </div>
"""
    
    # All runs table
    html += """
    <h2>📋 All Experiment Runs</h2>
    <table>
        <tr>
            <th>Variant</th>
            <th>Version</th>
            <th>Rows</th>
            <th>Test R²</th>
            <th>Test MAE</th>
            <th>Train/Test Gap</th>
            <th>Timestamp</th>
        </tr>
"""
    
    for run in sorted(runs, key=lambda x: x.get("timestamp", ""), reverse=True):
        m = run["metrics_parsed"]
        test_r2 = f"{m.get('test_r2', 0):.4f}" if m.get("test_r2") else "N/A"
        test_mae = f"€{m.get('test_mae', 0):,.0f}" if m.get("test_mae") else "N/A"
        
        gap = m.get("test_r2", 0) - m.get("train_r2", 0)
        gap_str = f"{gap:.4f}" if m.get("test_r2") and m.get("train_r2") else "N/A"
        
        html += f"""
        <tr>
            <td>{run['notes']}</td>
            <td><code>{run['model_version']}</code></td>
            <td>{run.get('train_rows', 'N/A')}</td>
            <td>{test_r2}</td>
            <td>{test_mae}</td>
            <td>{gap_str}</td>
            <td>{run.get('timestamp', 'N/A')}</td>
        </tr>
"""
    
    html += """
    </table>
    </div>
</body>
</html>
"""
    
    output_path = Path(output_path)
    output_path.write_text(html, encoding="utf-8")
    print(f"✓ Report generated: {output_path}")
