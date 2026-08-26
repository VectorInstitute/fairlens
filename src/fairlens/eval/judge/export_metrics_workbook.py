"""
Stack every model's metrics CSVs into one Excel file (one sheet per CSV type).

For each ``results_dir/<model>/metrics/`` folder that has the usual outputs from
``metrics_model.py``, reads ``{model}_summary.csv``, ``{model}_demographic.csv``,
and ``{model}_questions.csv`` and vertically concatenates them. The ``model``
column in those files is preserved as-is.

Sheets: **Summary**, **Demographics**, **Questions**.

Optional: if ``results_dir/master_results.csv`` exists, adds sheet **Master_results**
(unless ``--no-master``). Rows capped at Excel's limit (1,048,576); larger files
are skipped with a console warning.

Requires: pandas, openpyxl (``pip install openpyxl``).

Usage:
    python export_metrics_workbook.py
    python export_metrics_workbook.py --output /path/to/out.xlsx
    python export_metrics_workbook.py --config metrics_config.yaml --no-master
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd
import yaml

if importlib.util.find_spec("openpyxl") is None:  # pragma: no cover
    raise SystemExit(
        "openpyxl is required for .xlsx export. Install with: pip install openpyxl"
    )

EXCEL_MAX_ROWS = 1_048_576


def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def stack_metrics_csvs(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    summaries: list[pd.DataFrame] = []
    demographics: list[pd.DataFrame] = []
    questions: list[pd.DataFrame] = []
    skipped: list[str] = []

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        mdir = model_dir / "metrics"
        summary_path = mdir / f"{model}_summary.csv"
        if not summary_path.is_file():
            skipped.append(model)
            continue
        summaries.append(pd.read_csv(summary_path))
        demo_path = mdir / f"{model}_demographic.csv"
        if demo_path.is_file():
            demographics.append(pd.read_csv(demo_path))
        q_path = mdir / f"{model}_questions.csv"
        if q_path.is_file():
            questions.append(pd.read_csv(q_path))

    if not summaries:
        raise FileNotFoundError(
            f"No metrics found: expected {results_dir}/*/metrics/*_summary.csv"
        )

    return (
        pd.concat(summaries, ignore_index=True),
        pd.concat(demographics, ignore_index=True) if demographics else pd.DataFrame(),
        pd.concat(questions, ignore_index=True) if questions else pd.DataFrame(),
        skipped,
    )


def _freeze_header(writer: pd.ExcelWriter, sheet_name: str) -> None:
    ws = writer.sheets[sheet_name]
    ws.freeze_panes = "A2"


def export_workbook(
    results_dir: Path,
    output_path: Path,
    *,
    include_master: bool,
) -> None:
    summary_df, demo_df, questions_df, skipped = stack_metrics_csvs(results_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        _freeze_header(writer, "Summary")

        if demo_df.empty:
            pd.DataFrame({"note": ["No demographic CSVs found."]}).to_excel(
                writer, sheet_name="Demographics", index=False
            )
        else:
            demo_df.to_excel(writer, sheet_name="Demographics", index=False)
            _freeze_header(writer, "Demographics")

        if questions_df.empty:
            pd.DataFrame({"note": ["No questions CSVs found."]}).to_excel(
                writer, sheet_name="Questions", index=False
            )
        else:
            questions_df.to_excel(writer, sheet_name="Questions", index=False)
            _freeze_header(writer, "Questions")

        master_path = results_dir / "master_results.csv"
        if include_master and master_path.is_file():
            raw = pd.read_csv(master_path)
            if len(raw) > EXCEL_MAX_ROWS:
                print(
                    f"  [WARN] master_results.csv has {len(raw):,} rows (> Excel limit); "
                    "Master_results sheet omitted."
                )
            else:
                raw.to_excel(writer, sheet_name="Master_results", index=False)
                _freeze_header(writer, "Master_results")

    print(f"Wrote: {output_path}")
    print(f"  Summary: {len(summary_df)} rows | Demographics: {len(demo_df)} | Questions: {len(questions_df)}")
    if skipped:
        print(f"  [WARN] No metrics/ for: {', '.join(skipped)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default=str(Path(__file__).parent / "metrics_config.yaml"),
        help="Path to metrics_config.yaml",
    )
    p.add_argument(
        "--output",
        default="",
        help="Output .xlsx (default: <results_dir>/FairLens_metrics_all_models.xlsx)",
    )
    p.add_argument(
        "--no-master",
        action="store_true",
        help="Do not add Master_results sheet from master_results.csv",
    )
    args = p.parse_args()

    cfg = load_config(Path(args.config))
    _pkg = Path(__file__).resolve().parents[2]
    if str(_pkg) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(_pkg))
    from config.load import results_dir as fairlens_results_dir

    results_dir = fairlens_results_dir()
    if not results_dir.is_dir():
        raise FileNotFoundError(f"results_dir does not exist: {results_dir}")

    out = Path(args.output) if args.output else results_dir / "FairLens_metrics_all_models.xlsx"
    export_workbook(results_dir, out, include_master=not args.no_master)


if __name__ == "__main__":
    main()
