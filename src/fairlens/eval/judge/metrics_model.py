"""
metrics_model.py — Produces one combined JSON + CSVs per model.

Reads all domain CSVs, computes domain-level and overall metrics,
and writes a single {model}_metrics.json with this shape:

{
  "model": "CogVLM2-llama3-chat-19B",
  "domains": {
    "healthcare": {
      "bias_mean": 0.05,
      "total_pairs": 900,
      "unique_images": 100,
      "unique_questions": 9,
      "demographics": {
        "race":       { "White": { "bias_mean": 0.03, "bias_std": 0.09, "n": 310 }, ... },
        "gender":     { "Male": {...}, "Female": {...} },
        "age_bucket": { "18-29": {...}, ... }
      },
      "per_question": {
        "Does this person appear to be in pain? ...": {
          "bias_mean": 0.02, "bias_std": 0.08,
          "n": 100,
          "demographics": { "race": {...}, "gender": {...}, "age_bucket": {...} }
        },
        ...
      }
    },
    "hiring": { ... },
    "legal":  { ... },
    "overall": {
      "bias_mean": 0.07,
      "total_pairs": 2700,
      "demographics": {
        "race":       { "White": {...}, ... },
        "gender":     { ... },
        "age_bucket": { ... }
      }
      // no per_question — questions are domain-specific
    }
  }
}

CSVs (for analysis / dashboards):
  {model}_summary.csv      — one row per domain + overall
  {model}_questions.csv    — one row per domain × question
  {model}_demographic.csv  — one row per domain × dimension × group
                             (plus "overall" as a domain value)
"""

import csv
import json
from pathlib import Path
from collections import defaultdict

from metrics_domain import (
    compute_domain, demo_stats, _stats, DIMS
)

DOMAINS = ["healthcare", "hiring", "legal"]

# ── Overall rollup ─────────────────────────────────────────────────────────────
def _compute_overall(all_rows: list[dict]) -> dict:
    """
    overall.bias_mean    = mean of all bias scores (pooled across domains)
    overall.demographics = computed from ALL raw rows across all domains
                           (exact, not approximated)
    No per_question — questions are domain-specific.
    """
    # Mean + std from raw rows directly (exact — all scores pooled)
    all_bias = [r["bias_score"] for r in all_rows]
    b = _stats(all_bias)

    return {
        "bias_mean":    b["mean"],
        "bias_std":     b["std"],
        "total_pairs":  len(all_rows),
        "demographics": demo_stats(all_rows),
    }

# ── CSV writers ────────────────────────────────────────────────────────────────
def _write_summary_csv(path: Path, model: str, domains_out: dict):
    """One row per domain + overall."""
    fields = ["model", "domain",
              "bias_mean", "bias_std",
              "total_pairs", "unique_images", "unique_questions"]
    rows = []
    for domain, d in domains_out.items():
        rows.append({
            "model":            model,
            "domain":           domain,
            "bias_mean":        d["bias_mean"],
            "bias_std":         d.get("bias_std"),
            "total_pairs":      d["total_pairs"],
            "unique_images":    d.get("unique_images"),
            "unique_questions": d.get("unique_questions"),
        })
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def _write_questions_csv(path: Path, model: str, domains_out: dict):
    """One row per domain × question (overall excluded — no per_question there)."""
    fields = ["model", "domain", "question",
              "bias_mean", "bias_std", "n"]
    rows = []
    for domain, d in domains_out.items():
        if "per_question" not in d:
            continue
        for q, s in d["per_question"].items():
            rows.append({
                "model": model, "domain": domain, "question": q,
                "bias_mean": s["bias_mean"],
                "bias_std":  s["bias_std"],
                "n":         s["n"],
            })
    rows.sort(key=lambda r: r["bias_mean"] or 0, reverse=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def _write_demographic_csv(path: Path, model: str, domains_out: dict):
    """One row per domain × dimension × group (overall included as domain='overall')."""
    fields = ["model", "domain", "dimension", "group",
              "bias_mean", "bias_std", "n"]
    rows = []
    for domain, d in domains_out.items():
        for dim in DIMS:
            for grp, s in d["demographics"][dim].items():
                rows.append({
                    "model": model, "domain": domain,
                    "dimension": dim, "group": grp, **s,
                })
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

# ── Public API ─────────────────────────────────────────────────────────────────
def compute_model_metrics(
    model: str,
    results_dir: Path,
    domains: list[str] | None = None,
    write_files: bool = True,
) -> dict:
    """
    Reads all domain CSVs for this model, computes domain + overall metrics,
    returns and optionally writes the combined result.
    """
    if domains is None:
        domains = DOMAINS

    model_dir  = results_dir / model
    output_dir = model_dir / "metrics"

    domain_data: dict[str, dict] = {}
    all_rows:    list[dict]       = []

    for domain in domains:
        csv_path = model_dir / f"{domain}.csv"
        if not csv_path.exists():
            print(f"  [WARN] {model}/{domain}.csv not found — skipping.")
            continue
        d = compute_domain(csv_path)
        all_rows.extend(d.pop("_rows"))   # collect raw rows, remove from output
        domain_data[domain] = d

    if not domain_data:
        raise ValueError(f"No domain CSVs found for model '{model}' in {results_dir}")

    overall = _compute_overall(all_rows)

    # Build final domains dict: real domains + overall as last key
    domains_out = {**domain_data, "overall": overall}

    result = {
        "model":   model,
        "domains": domains_out,
    }

    if write_files:
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / f"{model}_metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        _write_summary_csv(output_dir / f"{model}_summary.csv", model, domains_out)
        _write_questions_csv(output_dir / f"{model}_questions.csv", model, domains_out)
        _write_demographic_csv(output_dir / f"{model}_demographic.csv", model, domains_out)

        print(f"  Outputs: {output_dir}/")

    return result


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True)
    parser.add_argument("--config", default="metrics_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    _pkg = Path(__file__).resolve().parents[2]
    if str(_pkg) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(_pkg))
    from config.load import results_dir as fairlens_results_dir

    results_dir = fairlens_results_dir()
    domains     = config.get("domains", DOMAINS)

    metrics = compute_model_metrics(
        model=args.model,
        results_dir=results_dir,
        domains=domains,
    )

    print(f"\n{'='*60}")
    print(f"  Model: {args.model}")
    print(f"{'='*60}")
    for domain, d in metrics["domains"].items():
        marker = " ← overall" if domain == "overall" else ""
        print(f"  {domain:<12}  bias={d['bias_mean']}  N={d['total_pairs']}{marker}")

    print(f"\n  Overall demographics (bias_mean / n):")
    overall_demo = metrics["domains"]["overall"]["demographics"]
    for dim in DIMS:
        print(f"  {dim}:")
        for grp, v in sorted(overall_demo[dim].items()):
            print(f"    {grp:<12}  bias={v['bias_mean']}  n={v['n']}")