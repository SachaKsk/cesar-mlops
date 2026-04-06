"""
Advanced training script with data enrichment and experiment comparison.

Demonstrates:
- Loading and merging multiple DVF files
- Data cleaning and department balancing
- Synthetic data augmentation
- Training multiple model versions
- Comparing metrics across runs
- Generating a comparison report

Run from repo root: python -m training.scripts.train_with_enrichment
"""

from pathlib import Path
import json
from argparse import ArgumentParser

from training.asset_rating_model.train_and_export import (
    train_on_dataframe,
    export_artifact,
)
from training.data_enrichment import (
    merge_multiple_dvf_files,
    augment_with_synthetic_samples,
    compute_department_statistics,
    clean_dvf_data,
)
from training.model_evaluation import evaluate_model_with_split, evaluate_model
from training.experiment_log import log_run, list_runs

# Where to find data and where to write the model (relative to repo root).
DATA_DIR_NAME = "data"
ARTIFACT_DIR_NAME = "artifact_storage"


def train_and_log_variant(
    df,
    variant_name: str,
    variant_version: str,
    artifact_dir: Path,
    use_augmentation: bool = False,
) -> dict:
    """
    Train a model variant, evaluate it, log results, and return metrics.
    """
    print(f"\n--- Variant: {variant_name} ({len(df)} rows) ---")
    
    if use_augmentation:
        print("  Augmenting with synthetic samples...")
        df = augment_with_synthetic_samples(df, multiplication_factor=1.5)
        print(f"  After augmentation: {len(df)} rows")
    
    print("  Training model...")
    model = train_on_dataframe(df)
    
    print("  Evaluating...")
    split_metrics = evaluate_model_with_split(model, df, test_size=0.2)
    full_metrics = evaluate_model(model, df)
    
    print(f"    Train R²: {split_metrics['train']['r2']:.4f}, Test R²: {split_metrics['test']['r2']:.4f}")
    print(f"    Train MAE: €{split_metrics['train']['mae']:.0f}, Test MAE: €{split_metrics['test']['mae']:.0f}")
    
    # Save model
    model_path, contract_path = export_artifact(model, artifact_dir, model_version=variant_version)
    print(f"    Saved: {model_path.name}")
    
    # Log to experiment tracker
    version = model_path.stem.replace("model_", "")
    metrics = {
        "train_r2": split_metrics['train']['r2'],
        "test_r2": split_metrics['test']['r2'],
        "train_mae": split_metrics['train']['mae'],
        "test_mae": split_metrics['test']['mae'],
        "train_rmse": split_metrics['train']['rmse'],
        "test_rmse": split_metrics['test']['rmse'],
        "full_r2": full_metrics['r2'],
        "full_mae": full_metrics['mae'],
    }
    
    log_run(version, train_rows=len(df), notes=variant_name, metrics=metrics)
    
    return {
        "variant_name": variant_name,
        "version": version,
        "num_rows": len(df),
        "metrics": metrics,
    }


def print_comparison_report(results: list[dict]) -> None:
    """Print a formatted comparison report of all experiment runs."""
    print("\n" + "="*80)
    print("EXPERIMENT COMPARISON REPORT".center(80))
    print("="*80)
    
    # Sort by test R² (descending, higher is better)
    sorted_results = sorted(results, key=lambda x: x["metrics"]["test_r2"], reverse=True)
    
    print(f"\n{'Rank':<6} {'Variant':<25} {'Rows':<8} {'Test R²':<10} {'Test MAE':<12} {'Version':<20}")
    print("-" * 85)
    
    for i, result in enumerate(sorted_results, 1):
        test_r2 = result["metrics"]["test_r2"]
        test_mae = result["metrics"]["test_mae"]
        print(
            f"{i:<6} "
            f"{result['variant_name']:<25} "
            f"{result['num_rows']:<8} "
            f"{test_r2:<10.4f} "
            f"€{test_mae:<11,.0f} "
            f"{result['version']:<20}"
        )
    
    # Highlight best model
    best = sorted_results[0]
    print("\n" + "-" * 85)
    print(f"✓ Best model: {best['variant_name']} (Test R² = {best['metrics']['test_r2']:.4f})")
    print(f"  → Deploy version: {best['version']}")


def main() -> None:
    parser = ArgumentParser(description="Train multiple model variants with enrichment and compare")
    parser.add_argument("--clean", action="store_true", help="Clean data (remove outliers)")
    parser.add_argument("--augment", action="store_true", help="Add synthetic samples to all variants")
    parser.add_argument("--report", action="store_true", help="Print comparison report from past runs")
    args = parser.parse_args()
    
    repo_root = Path(__file__).resolve().parent.parent.parent
    data_dir = repo_root / DATA_DIR_NAME
    artifact_dir = repo_root / ARTIFACT_DIR_NAME
    
    # If report flag, just show past runs
    if args.report:
        runs = list_runs()
        if not runs:
            print("No experiment runs logged yet. Train some models first.")
            return
        
        print("\n" + "="*80)
        print("PAST EXPERIMENT RUNS".center(80))
        print("="*80)
        print(f"\n{'Timestamp':<20} {'Version':<20} {'Rows':<10} {'Notes':<30} {'Test R²':<10}")
        print("-" * 95)
        
        for run in sorted(runs, key=lambda x: x.get("timestamp", ""), reverse=True)[:10]:
            test_r2 = "N/A"
            if run.get("metrics"):
                metrics = run["metrics"]
                if isinstance(metrics, str):
                    try:
                        metrics = json.loads(metrics)
                    except:
                        pass
                if isinstance(metrics, dict):
                    test_r2 = f"{metrics.get('test_r2', 0):.4f}"
            
            print(
                f"{run.get('timestamp', ''):<20} "
                f"{run.get('model_version', ''):<20} "
                f"{run.get('train_rows', ''):<10} "
                f"{run.get('notes', ''):<30} "
                f"{test_r2:<10}"
            )
        return
    
    # Standard training flow: train multiple variants
    print(f"Loading data from {data_dir} ...")
    
    # Load baseline (simple merge)
    baseline_data = merge_multiple_dvf_files(data_dir, clean=False, balance=False)
    
    # Variant 1: Baseline (as-is)
    results = []
    base_result = train_and_log_variant(
        baseline_data,
        "Baseline (raw data)",
        "v1_baseline",
        artifact_dir,
        use_augmentation=False,
    )
    results.append(base_result)
    
    # Variant 2: Cleaned data
    if args.clean:
        print(f"\nLoading and cleaning data...")
        cleaned_data = merge_multiple_dvf_files(data_dir, clean=True, balance=False)
        result = train_and_log_variant(
            cleaned_data,
            "Cleaned (outliers removed)",
            "v2_cleaned",
            artifact_dir,
            use_augmentation=False,
        )
        results.append(result)
    
    # Variant 3: With synthetic augmentation
    if args.augment:
        result = train_and_log_variant(
            baseline_data,
            "Augmented (synthetic samples)",
            "v3_augmented",
            artifact_dir,
            use_augmentation=True,
        )
        results.append(result)
    
    # Variant 4: Cleaned + Augmented
    if args.clean and args.augment:
        result = train_and_log_variant(
            cleaned_data,
            "Cleaned + Augmented",
            "v4_clean_augmented",
            artifact_dir,
            use_augmentation=True,
        )
        results.append(result)
    
    # Print comparison report
    if len(results) > 1:
        print_comparison_report(results)
    
    # Department statistics
    print("\n" + "="*80)
    print("DEPARTMENT STATISTICS (Baseline)".center(80))
    print("="*80)
    stats = compute_department_statistics(baseline_data)
    print(stats.head(10).to_string())
    
    print(f"\nTotal departments: {len(stats)}")
    print(f"\nTo compare runs:\n  python -m training.scripts.train_with_enrichment --report")


if __name__ == "__main__":
    main()
