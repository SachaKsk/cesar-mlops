"""
Data enrichment module: augment training data with additional features or synthetic samples.

Includes:
- Data augmentation utilities (add synthetic samples based on real data distribution)
- Feature engineering examples (distance to city center, urbanization level, etc.)
- Data cleaning and validation
- Multi-department data balancing
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def clean_dvf_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean DVF data: remove outliers, validate ranges.
    
    Returns:
        Cleaned DataFrame with:
        - Surface: 10-500 m² (residential focus)
        - Rooms: 1-10 rooms
        - Price: 10k-2M EUR (reasonable range)
    """
    df = df.copy()
    
    # Remove rows with NaN in required columns
    required = ["surface_reelle_bati", "nombre_pieces_principales", "code_departement", "type_local", "valeur_fonciere"]
    df = df.dropna(subset=required)
    
    # Convert to numeric with error handling
    for col in ["surface_reelle_bati", "nombre_pieces_principales", "valeur_fonciere"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df = df.dropna(subset=["surface_reelle_bati", "nombre_pieces_principales", "valeur_fonciere"])
    
    # Filter outliers
    df = df[
        (df["surface_reelle_bati"] > 10) & (df["surface_reelle_bati"] < 500) &
        (df["nombre_pieces_principales"] > 0) & (df["nombre_pieces_principales"] <= 10) &
        (df["valeur_fonciere"] > 10000) & (df["valeur_fonciere"] < 2000000)
    ]
    
    return df.reset_index(drop=True)


def augment_with_synthetic_samples(
    df: pd.DataFrame,
    multiplication_factor: float = 1.5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Create synthetic samples by interpolating between real samples.
    Generates new but realistic data to increase dataset size.
    
    Args:
        df: Original DataFrame
        multiplication_factor: Target ratio of synthetic:real samples (e.g., 1.5 means add 50%)
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame with original + synthetic samples
    """
    np.random.seed(random_state)
    
    num_synthetic = int(len(df) * (multiplication_factor - 1))
    synthetic_samples = []
    
    for _ in range(num_synthetic):
        # Pick two random rows and interpolate
        idx1, idx2 = np.random.choice(len(df), 2, replace=False)
        row1, row2 = df.iloc[idx1], df.iloc[idx2]
        
        alpha = np.random.uniform(0, 1)
        
        synthetic_row = {
            "surface_reelle_bati": alpha * row1["surface_reelle_bati"] + (1 - alpha) * row2["surface_reelle_bati"],
            "nombre_pieces_principales": int(round(alpha * row1["nombre_pieces_principales"] + (1 - alpha) * row2["nombre_pieces_principales"])),
            "code_departement": row1["code_departement"] if np.random.rand() > 0.5 else row2["code_departement"],
            "type_local": row1["type_local"] if np.random.rand() > 0.5 else row2["type_local"],
            "valeur_fonciere": alpha * row1["valeur_fonciere"] + (1 - alpha) * row2["valeur_fonciere"],
        }
        synthetic_samples.append(synthetic_row)
    
    synthetic_df = pd.DataFrame(synthetic_samples)
    return pd.concat([df, synthetic_df], ignore_index=True)


def generate_synthetic_department_data(
    base_df: pd.DataFrame,
    target_department: str,
    num_samples: int = 100,
    price_multiplier: float = 1.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic samples for a specific department, useful when you have
    limited data for certain regions.
    
    Modulates prices based on typical regional multipliers (e.g., Paris is ~2x national average).
    
    Args:
        base_df: Base data to sample from
        target_department: Departement code (e.g., "75" for Paris)
        num_samples: Number of synthetic samples to generate
        price_multiplier: Price adjustment factor (1.0 = no change, 2.0 = double prices)
        random_state: Random seed
    
    Returns:
        DataFrame with synthetic samples for the target department
    """
    np.random.seed(random_state)
    
    # Use base data as template for realistic distributions
    samples = []
    for _ in range(num_samples):
        template_idx = np.random.randint(0, len(base_df))
        template = base_df.iloc[template_idx]
        
        # Add some noise to surface and rooms
        surface = template["surface_reelle_bati"] * np.random.uniform(0.8, 1.2)
        rooms = max(1, int(template["nombre_pieces_principales"] + np.random.normal(0, 0.5)))
        
        # Adjust price based on department multiplier
        price = template["valeur_fonciere"] * price_multiplier * np.random.uniform(0.9, 1.1)
        
        samples.append({
            "surface_reelle_bati": surface,
            "nombre_pieces_principales": rooms,
            "code_departement": target_department,
            "type_local": template["type_local"],
            "valeur_fonciere": price,
        })
    
    return pd.DataFrame(samples)


def balance_by_department(
    df: pd.DataFrame,
    min_samples_per_dept: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Ensure each department has at least min_samples_per_dept samples.
    For departments with fewer samples, create interpolated synthetic data.
    
    Returns:
        Balanced DataFrame
    """
    np.random.seed(random_state)
    
    result_dfs = []
    for dept in df["code_departement"].unique():
        dept_data = df[df["code_departement"] == dept]
        result_dfs.append(dept_data)
        
        if len(dept_data) < min_samples_per_dept:
            # Generate additional synthetic samples for this department
            num_to_generate = min_samples_per_dept - len(dept_data)
            synthetic = generate_synthetic_department_data(
                df,
                target_department=dept,
                num_samples=num_to_generate,
                random_state=random_state + 1,
            )
            result_dfs.append(synthetic)
    
    return pd.concat(result_dfs, ignore_index=True)


def merge_multiple_dvf_files(
    data_dir: Path,
    clean: bool = True,
    balance: bool = False,
    min_samples_per_dept: int = 50,
) -> pd.DataFrame:
    """
    Load and merge multiple DVF CSV files from a directory.
    Optionally clean and balance across departments.
    
    Args:
        data_dir: Directory containing .csv files
        clean: Apply data cleaning (remove outliers)
        balance: Balance samples across departments (synthetic augmentation)
        min_samples_per_dept: Minimum samples per department (if balance=True)
    
    Returns:
        Merged and optionally cleaned/balanced DataFrame
    """
    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, sep=";", low_memory=False)
            print(f"Loaded {csv_file.name}: {len(df)} rows")
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {csv_file.name}: {e}")
    
    if not dfs:
        raise ValueError("No CSV files could be loaded")
    
    merged = pd.concat(dfs, ignore_index=True)
    print(f"Total merged: {len(merged)} rows")
    
    if clean:
        merged = clean_dvf_data(merged)
        print(f"After cleaning: {len(merged)} rows")
    
    if balance:
        merged = balance_by_department(merged, min_samples_per_dept=min_samples_per_dept)
        print(f"After balancing: {len(merged)} rows")
    
    return merged


def compute_department_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics per department: count, avg surface, avg rooms, avg price, price per m².
    Useful to understand data distribution and spot imbalances.
    
    Returns:
        DataFrame with department-level statistics
    """
    stats = df.groupby("code_departement").agg({
        "surface_reelle_bati": ["count", "mean", "std"],
        "nombre_pieces_principales": "mean",
        "valeur_fonciere": ["mean", "min", "max"],
    }).round(2)
    
    # Flatten column names
    stats.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in stats.columns]
    
    # Add price per m²
    stats["price_per_m2"] = (stats["valeur_fonciere_mean"] / stats["surface_reelle_bati_mean"]).round(2)
    
    return stats.sort_values("surface_reelle_bati_count", ascending=False)
