"""
Generate synthetic DVF data for enrichment and testing.

Creates realistic property valuation data for multiple French departments,
useful for:
- Testing data enrichment pipelines
- Training models with broader geographic coverage
- Demonstrating multi-variant experiment comparison

Run from repo root: python -m training.scripts.generate_synthetic_data
"""

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import argparse


# Real approximate price per m² in EUR for different departments (simplified)
DEPT_PRICE_MULTIPLIERS = {
    "75": 8000,   # Paris - very expensive
    "92": 6000,   # Hauts-de-Seine (west of Paris)
    "94": 5000,   # Val-de-Marne (southeast of Paris)
    "77": 3000,   # Seine-et-Marne (outer Paris region)
    "13": 4000,   # Bouches-du-Rhône (Marseille area)
    "69": 3500,   # Rhône (Lyon area)
    "59": 2500,   # Nord (Lille area)
    "31": 3000,   # Haute-Garonne (Toulouse area)
    "33": 3200,   # Gironde (Bordeaux area)
    "44": 3300,   # Loire-Atlantique (Nantes area)
    "35": 3000,   # Ille-et-Vilaine (Rennes area)
    "68": 2800,   # Haut-Rhin (Alsace)
}

# Default multiplier for departments not in the dict
DEFAULT_PRICE_MULTIPLIER = 2500

# Property types distribution
PROPERTY_TYPES = ["Appartement", "Maison", "Dépendance", "Local industriel. commercial ou assimilé"]
TYPE_DISTRIBUTION = [0.45, 0.40, 0.10, 0.05]


def generate_department_data(
    dept_code: str,
    num_samples: int = 200,
    price_multiplier: float | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic DVF data for a single department.
    
    Args:
        dept_code: Department code (e.g., "75" for Paris)
        num_samples: Number of samples to generate
        price_multiplier: EUR per m² (None uses default)
        random_state: Seed for reproducibility
    
    Returns:
        DataFrame with synthetic property data
    """
    np.random.seed(random_state)
    
    if price_multiplier is None:
        price_multiplier = DEPT_PRICE_MULTIPLIERS.get(dept_code, DEFAULT_PRICE_MULTIPLIER)
    
    # Surface distribution: most homes 50-200 m², some smaller (apartments), some larger (houses)
    surfaces = np.random.lognormal(mean=4.2, sigma=0.5, size=num_samples)
    surfaces = np.clip(surfaces, 20, 400).astype(int)
    
    # Rooms: typically 2-4 for apartments, 3-5 for houses
    rooms = np.random.choice(
        [1, 2, 3, 4, 5, 6, 7],
        size=num_samples,
        p=[0.08, 0.25, 0.30, 0.22, 0.10, 0.04, 0.01]
    )
    
    # Property types
    types = np.random.choice(PROPERTY_TYPES, size=num_samples, p=TYPE_DISTRIBUTION)
    
    # Price: base = surface * multiplier, with some variance
    base_prices = surfaces * price_multiplier
    price_noise = np.random.normal(1.0, 0.15, size=num_samples)  # +/- 15% variance
    prices = (base_prices * price_noise).astype(int)
    
    # Ensure prices are reasonable
    prices = np.clip(prices, 30000, None)
    
    df = pd.DataFrame({
        "surface_reelle_bati": surfaces,
        "nombre_pieces_principales": rooms,
        "code_departement": dept_code,
        "type_local": types,
        "valeur_fonciere": prices,
    })
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic DVF data by department")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory (default: data/)"
    )
    parser.add_argument(
        "--departments",
        type=str,
        default="75,92,94,77,13,69",
        help="Comma-separated list of department codes (default: 75,92,94,77,13,69)"
    )
    parser.add_argument(
        "--samples-per-dept",
        type=int,
        default=150,
        help="Samples per department (default: 150)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--add",
        action="store_true",
        help="Add to existing data instead of replacing"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    departments = [d.strip() for d in args.departments.split(",")]
    
    print(f"Generating synthetic DVF data...")
    print(f"  Output: {output_dir}")
    print(f"  Departments: {', '.join(departments)}")
    print(f"  Samples per dept: {args.samples_per_dept}")
    print()
    
    all_dfs = []
    for dept in departments:
        print(f"  Generating for department {dept}...", end="", flush=True)
        df = generate_department_data(
            dept,
            num_samples=args.samples_per_dept,
            random_state=args.seed + hash(dept) % 1000,
        )
        all_dfs.append(df)
        print(f" ✓ ({len(df)} samples)")
    
    merged = pd.concat(all_dfs, ignore_index=True)
    total_samples = len(merged)
    
    # Determine output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"dvf_synthetic_{timestamp}.csv"
    output_path = output_dir / output_filename
    
    # Write CSV with semicolon separator (DVF standard)
    merged.to_csv(output_path, sep=";", index=False, encoding="utf-8")
    
    print(f"\n✓ Wrote {total_samples} synthetic records to {output_path.name}")
    print(f"\nDataset summary:")
    print(f"  Total samples: {total_samples}")
    print(f"  Surface range: {merged['surface_reelle_bati'].min()}-{merged['surface_reelle_bati'].max()} m²")
    print(f"  Rooms range: {merged['nombre_pieces_principales'].min()}-{merged['nombre_pieces_principales'].max()}")
    print(f"  Price range: €{merged['valeur_fonciere'].min():,}-€{merged['valeur_fonciere'].max():,}")
    print(f"\nDepartment breakdown:")
    
    dept_stats = merged.groupby("code_departement").agg({
        "valeur_fonciere": ["count", "mean", "min", "max"],
    }).round(0)
    
    for dept in departments:
        if dept in merged["code_departement"].values:
            dept_data = merged[merged["code_departement"] == dept]
            price_mean = dept_data["valeur_fonciere"].mean()
            price_per_m2 = price_mean / dept_data["surface_reelle_bati"].mean()
            print(f"  {dept}: {len(dept_data)} samples, avg €{price_mean:,.0f}, €{price_per_m2:.0f}/m²")


if __name__ == "__main__":
    main()
