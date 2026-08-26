"""
FairLens - Build Master CSV

Run this once after all eval jobs have finished.
Concatenates all per-model per-domain CSVs into a single master_results.csv.

Usage:
    python build_master.py

For a multi-sheet Excel rollup of per-model metrics (after ``metrics_model.py``), see
``export_metrics_workbook.py``.
"""

import yaml
import pandas as pd
from pathlib import Path


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    config_path = Path(__file__).parent / "metrics_config.yaml"
    cfg         = load_config(str(config_path))
    _pkg = Path(__file__).resolve().parents[2]
    if str(_pkg) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(_pkg))
    from config.load import results_dir as fairlens_results_dir

    results_dir = fairlens_results_dir()

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    frames = []
    missing = []

    # Walk each model directory
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        model_csvs = list(model_dir.glob("*.csv"))

        # Only pick domain CSVs (healthcare/hiring/legal), not the all_domains one
        domain_csvs = [f for f in model_csvs if f.stem in ("healthcare", "hiring", "legal")]

        if not domain_csvs:
            missing.append(model_name)
            print(f"[WARN] No domain CSVs found for model: {model_name}")
            continue

        for csv_path in sorted(domain_csvs):
            try:
                df = pd.read_csv(csv_path)
                frames.append(df)
                print(f"  Loaded {len(df):>6} rows — {model_name}/{csv_path.name}")
            except Exception as e:
                print(f"  [ERROR] Could not read {csv_path}: {e}")

    if not frames:
        print("No CSVs found. Have the eval jobs finished?")
        return

    master_df   = pd.concat(frames, ignore_index=True)
    master_path = results_dir / "master_results.csv"
    master_df.to_csv(master_path, index=False)

    print(f"\nMaster CSV saved: {master_path}")
    print(f"Total rows: {len(master_df)}")

    # Summary per model + domain
    print("\n--- Row counts per model/domain ---")
    counts = master_df.groupby(["model", "domain"]).size().unstack(fill_value=0)
    print(counts.to_string())

    # Score summary (exclude -1 error rows)
    print("\n--- Score summary (valid rows only) ---")
    for col in ["bias_score", "faithfulness_score"]:
        if col in master_df.columns:
            valid = master_df[master_df[col] != -1.0][col]
            errors = (master_df[col] == -1.0).sum()
            print(f"  {col}: mean={valid.mean():.3f}  std={valid.std():.3f}  "
                  f"n={len(valid)}  errors={errors}")

    if missing:
        print(f"\n[WARN] Models with no results: {missing}")


if __name__ == "__main__":
    main()
