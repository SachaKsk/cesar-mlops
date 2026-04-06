"""
CLI for experiment tracking and analysis.

Commands:
    cesar experiment-analysis list     - List all past experiment runs
    cesar experiment-analysis compare  - Compare two model versions
    cesar experiment-analysis report   - Generate HTML report
    cesar experiment-analysis best     - Show best model by criterion
"""

from typing import Optional
import typer
from pathlib import Path
import json

from comparison.experiment_analysis import (
    analyze_runs,
    compare_model_pairs,
    generate_html_report,
    get_run_by_version,
)

app = typer.Typer(help="Experiment tracking and comparison")


@app.command()
def list_runs() -> None:
    """List all logged experiment runs."""
    from training.experiment_log import list_runs as log_list_runs
    
    runs = log_list_runs()
    if not runs:
        typer.echo("No experiment runs logged yet.", err=True)
        raise typer.Exit(1)
    
    typer.echo("\n📋 Experiment Runs:")
    typer.echo("-" * 100)
    typer.echo(f"{'Timestamp':<20} {'Version':<20} {'Variant':<30} {'Rows':<8} {'Test R²':<10} {'Test MAE':<12}")
    typer.echo("-" * 100)
    
    for run in sorted(runs, key=lambda x: x.get("timestamp", ""), reverse=True):
        metrics = run.get("metrics", {})
        if isinstance(metrics, str):
            try:
                metrics = json.loads(metrics)
            except:
                metrics = {}
        
        test_r2 = f"{metrics.get('test_r2', 0):.4f}" if metrics.get("test_r2") else "N/A"
        test_mae = f"€{metrics.get('test_mae', 0):,.0f}" if metrics.get("test_mae") else "N/A"
        
        typer.echo(
            f"{run.get('timestamp', ''):<20} "
            f"{run.get('model_version', ''):<20} "
            f"{run.get('notes', ''):<30} "
            f"{run.get('train_rows', ''):<8} "
            f"{test_r2:<10} "
            f"{test_mae:<12}"
        )


@app.command()
def summary() -> None:
    """Show summary of best models."""
    analysis = analyze_runs()
    
    if "error" in analysis:
        typer.echo(f"❌ {analysis['error']}", err=True)
        raise typer.Exit(1)
    
    summary_data = analysis.get("summary", {})
    typer.echo("\n" + "="*80)
    typer.echo("EXPERIMENT SUMMARY".center(80))
    typer.echo("="*80)
    typer.echo(f"\nTotal runs: {analysis['num_runs']}")
    
    if "best_by_r2" in summary_data:
        best = summary_data["best_by_r2"]
        typer.echo(f"\n✓ Best R² Score: {best['variant']}")
        typer.echo(f"  Version: {best['version']}")
        typer.echo(f"  Test R²: {best['test_r2']:.4f}")
        typer.echo(f"  Test MAE: €{best['test_mae']:,.0f}")
    
    if "best_by_mae" in summary_data:
        best = summary_data["best_by_mae"]
        typer.echo(f"\n✓ Best MAE (Lowest Error): {best['variant']}")
        typer.echo(f"  Version: {best['version']}")
        typer.echo(f"  Test R²: {best['test_r2']:.4f}")
        typer.echo(f"  Test MAE: €{best['test_mae']:,.0f}")
    
    if "best_generalization" in summary_data:
        best = summary_data["best_generalization"]
        typer.echo(f"\n✓ Best Generalization: {best['variant']}")
        typer.echo(f"  Version: {best['version']}")
        typer.echo(f"  Train R²: {best['train_r2']:.4f}")
        typer.echo(f"  Test R²: {best['test_r2']:.4f}")
        typer.echo(f"  Gap: {best['gap']:.4f}")
    
    if "improvement_over_baseline" in summary_data:
        imp = summary_data["improvement_over_baseline"]
        typer.echo(f"\n📈 Improvement Over Baseline:")
        typer.echo(f"  R² Change: {imp['r2_improvement_pct']:+.2f}%")
        typer.echo(f"  MAE Change: {imp['mae_improvement_pct']:+.2f}%")


@app.command()
def compare(
    version_a: str = typer.Argument(..., help="First model version"),
    version_b: str = typer.Argument(..., help="Second model version"),
) -> None:
    """Compare two model versions side-by-side."""
    comparison = compare_model_pairs(version_a, version_b)
    
    if "error" in comparison:
        typer.echo(f"❌ {comparison['error']}", err=True)
        raise typer.Exit(1)
    
    typer.echo("\n" + "="*80)
    typer.echo("MODEL COMPARISON".center(80))
    typer.echo("="*80)
    
    typer.echo(f"\nVersion A: {version_a}")
    typer.echo(f"  Variant: {comparison['variant_a']}")
    typer.echo(f"  Rows: {comparison['rows_a']}")
    
    typer.echo(f"\nVersion B: {version_b}")
    typer.echo(f"  Variant: {comparison['variant_b']}")
    typer.echo(f"  Rows: {comparison['rows_b']}")
    
    typer.echo(f"\n{'Metric':<15} {'Version A':<15} {'Version B':<15} {'Difference':<15} {'Winner'}")
    typer.echo("-" * 70)
    
    for metric, values in comparison.get("differences", {}).items():
        diff = values["diff_pct"]
        diff_str = f"{diff:+.2f}%"
        winner = "B" if values["winner"] == "B" else "A"
        typer.echo(
            f"{metric:<15} "
            f"{values['value_a']:<15.4f} "
            f"{values['value_b']:<15.4f} "
            f"{diff_str:<15} "
            f"{winner}"
        )


@app.command()
def report(
    output: Path = typer.Option(
        Path("experiment_report.html"),
        help="Output HTML file path"
    ),
) -> None:
    """Generate an HTML comparison report."""
    typer.echo(f"Generating report...")
    generate_html_report(output)
    typer.echo(f"✓ Report generated: {output}")
    typer.echo(f"  Open in browser to view: {output.resolve()}")


def main() -> None:
    """Entry point for experiment analysis CLI."""
    app()


if __name__ == "__main__":
    main()
