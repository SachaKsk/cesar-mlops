"""
Train the model using all CSV files in the data/ folder.

- Finds every *.csv in data/
- Checks that each file has the required columns (surface, rooms, department, type, value)
- Checks that all files have the same columns (same schema); if not, raises an error
- Combines all rows and trains one model
- Saves the model and contract to artifact_storage/
- Evaluates on train/test split and logs results to experiment tracker

Run from repo root: python -m training.scripts.train_from_minimal_csv
"""

from pathlib import Path
import json

from training.asset_rating_model.train_and_export import (
    load_all_csvs_from_dir,
    train_on_dataframe,
    export_artifact,
)
from training.model_evaluation import evaluate_model_with_split, evaluate_model
from training.experiment_log import log_run

# Where to find data and where to write the model (relative to repo root).
DATA_DIR_NAME = "data"
ARTIFACT_DIR_NAME = "artifact_storage"


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    data_dir = repo_root / DATA_DIR_NAME
    artifact_dir = repo_root / ARTIFACT_DIR_NAME

    # Load and validate: all CSVs in data/ must have the same columns and the required ones.
    print(f"Loading CSVs from {data_dir} ...")
    combined = load_all_csvs_from_dir(data_dir, separator=";")
    num_rows = len(combined)
    num_files = len(list(data_dir.glob("*.csv")))
    print(f"  Found {num_files} file(s), {num_rows} rows in total.")

    # Train one model on the combined data.
    print("Training model ...")
    model = train_on_dataframe(combined)

    # Evaluate on train/test split
    print("Evaluating model ...")
    split_metrics = evaluate_model_with_split(model, combined, test_size=0.2)
    full_metrics = evaluate_model(model, combined)
    
    print(f"  Train R²: {split_metrics['train']['r2']:.4f}, Test R²: {split_metrics['test']['r2']:.4f}")
    print(f"  Train MAE: €{split_metrics['train']['mae']:.0f}, Test MAE: €{split_metrics['test']['mae']:.0f}")
    
    # Save model and contract with a version name.
    model_path, contract_path = export_artifact(
        model,
        artifact_dir,
        model_version="minimal",
    )
    print(f"  Model:  {model_path}")
    print(f"  Contract: {contract_path}")
    
    # Log to experiment tracker with metrics
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
    
    log_run(
        version,
        train_rows=num_rows,
        notes=f"{num_files} file(s) from {data_dir.name}",
        metrics=metrics,
    )
    print(f"  Logged to experiment tracker.")
    
    print(
        "Set CESAR_MODEL_PATH and CESAR_CONTRACT_PATH to these paths "
        "(or symlink as model_latest.joblib / contract_latest.json)."
    )


if __name__ == "__main__":
    main()

    print(
        "Set CESAR_MODEL_PATH and CESAR_CONTRACT_PATH to these paths "
        "(or symlink as model_latest.joblib / contract_latest.json)."
    )


if __name__ == "__main__":
    main()
