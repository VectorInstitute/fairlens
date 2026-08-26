"""
metrics_compare.py — Cross-model comparison for FairLens evaluation results.

Compares two or more models. Reads (or computes) each model's metrics,
then builds side-by-side comparison outputs.

Output JSON shape:
{
  "models": ["ModelA", "ModelB"],
  "domains": {
    "healthcare": {
      "ModelA": { "bias_mean": 0.05, "total_pairs": 900 },
      "ModelB": { "bias_mean": 0.08, "total_pairs": 900 }
    },
    "hiring":  { ... },
    "legal":   { ... },
    "overall": { ... }
  },
  "demographics": {
    "race": {
      "White": {
        "overall": {
          "ModelA": { "bias_mean": 0.03, "n": 930 },
          "ModelB": { "bias_mean": 0.07, "n": 930 }
        },
        "healthcare": {
          "ModelA": { "bias_mean": 0.02, "n": 310 },
          "ModelB": { ... }
        },
        ...
      },
      "Black": { ... },
      ...
    },
    "gender":     { ... },
    "age_bucket": { ... }
  },
  "per_question": {
    "healthcare": {
      "Does this person appear to be in pain? ...": {
        "ModelA": { "bias_mean": 0.02, "n": 100 },
        "ModelB": { "bias_mean": 0.05, "n": 100 }
      },
      ...
    },
    "hiring": { ... },
    "legal":  { ... }
  }
}

Outputs (in results/comparison_{models}/):
  comparison.json
  comparison_summary.csv      — one row per model × domain
  comparison_demographic.csv  — one row per model × domain × dimension × group
  comparison_questions.csv    — one row per model × domain × question
"""

import csv
import json
from pathlib import Path

from metrics_model import compute_model_metrics, DOMAINS, DIMS

# ── Builders ───────────────────────────────────────────────────────────────────
def _build_domains(model_results: dict[str, dict]) -> dict:
    """
    { domain: { model: { bias_mean, total_pairs } } }
    Covers all domain keys including "overall".
    """
    all_domain_keys: set[str] = set()
    for mr in model_results.values():
        all_domain_keys.update(mr["domains"].keys())

    result = {}
    for domain in (*DOMAINS, "overall"):
        if domain not in all_domain_keys:
            continue
        result[domain] = {}
        for model, mr in model_results.items():
            d = mr["domains"].get(domain, {})
            result[domain][model] = {
                "bias_mean":   d.get("bias_mean"),
                "total_pairs": d.get("total_pairs"),
            }
    return result

def _build_demographics(model_results: dict[str, dict]) -> dict:
    """
    { dim: { group: { domain: { model: { bias_mean, n } } } } }

    Groups together: "for White subjects, in healthcare, ModelA vs ModelB"
    Includes "overall" as a domain key alongside the real domains.
    """
    result = {}
    for dim in DIMS:
        result[dim] = {}
        # Collect all groups seen across all models and domains
        all_groups: set[str] = set()
        for mr in model_results.values():
            for d in mr["domains"].values():
                all_groups.update(d.get("demographics", {}).get(dim, {}).keys())

        for grp in sorted(all_groups):
            result[dim][grp] = {}
            for domain in (*DOMAINS, "overall"):
                domain_entry = {}
                for model, mr in model_results.items():
                    s = mr["domains"].get(domain, {}).get("demographics", {}).get(dim, {}).get(grp, {})
                    if s:
                        domain_entry[model] = {
                            "bias_mean": s.get("bias_mean"),
                            "n":         s.get("n", 0),
                        }
                if domain_entry:
                    result[dim][grp][domain] = domain_entry
    return result

def _build_per_question(model_results: dict[str, dict]) -> dict:
    """
    { domain: { question: { model: { bias_mean, n } } } }
    """
    result = {}
    all_pairs: set[tuple[str, str]] = set()
    for mr in model_results.values():
        for domain, d in mr["domains"].items():
            for q in d.get("per_question", {}):
                all_pairs.add((domain, q))

    for domain, question in sorted(all_pairs):
        if domain not in result:
            result[domain] = {}
        result[domain][question] = {}
        for model, mr in model_results.items():
            s = mr["domains"].get(domain, {}).get("per_question", {}).get(question, {})
            if s:
                result[domain][question][model] = {
                    "bias_mean": s.get("bias_mean"),
                    "n":         s.get("n", 0),
                }
    return result

# ── CSV writers ────────────────────────────────────────────────────────────────
def _write_summary_csv(path: Path, models: list[str], domains_cmp: dict):
    fields = ["domain", "model", "bias_mean", "total_pairs"]
    rows = []
    for domain, model_vals in domains_cmp.items():
        for model, s in model_vals.items():
            rows.append({"domain": domain, "model": model, **s})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def _write_demographic_csv(path: Path, models: list[str], demo_cmp: dict):
    fields = ["dimension", "group", "domain", "model",
              "bias_mean", "n"]
    rows = []
    for dim in DIMS:
        for grp, domain_vals in demo_cmp[dim].items():
            for domain, model_vals in domain_vals.items():
                for model, s in model_vals.items():
                    rows.append({
                        "dimension": dim, "group": grp,
                        "domain": domain, "model": model, **s,
                    })
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def _write_questions_csv(path: Path, models: list[str], pq_cmp: dict):
    fields = ["domain", "question", "model", "bias_mean", "n"]
    rows = []
    for domain, questions in pq_cmp.items():
        for question, model_vals in questions.items():
            for model, s in model_vals.items():
                rows.append({
                    "domain": domain, "question": question,
                    "model": model, **s,
                })
    rows.sort(key=lambda r: r["bias_mean"] or 0, reverse=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

# ── Public API ─────────────────────────────────────────────────────────────────
def compare_models(
    models: list[str],
    results_dir: Path,
    domains: list[str] | None = None,
    write_files: bool = True,
) -> dict:
    if domains is None:
        domains = DOMAINS
    if len(models) < 2:
        raise ValueError("Need at least 2 models to compare.")

    model_results: dict[str, dict] = {}
    for model in models:
        print(f"  Computing metrics for: {model}")
        model_results[model] = compute_model_metrics(
            model=model,
            results_dir=results_dir,
            domains=domains,
            write_files=write_files,
        )

    domains_cmp  = _build_domains(model_results)
    demo_cmp     = _build_demographics(model_results)
    pq_cmp       = _build_per_question(model_results)

    result = {
        "models":       models,
        "domains":      domains_cmp,
        "demographics": demo_cmp,
        "per_question": pq_cmp,
    }

    if write_files:
        tag     = "_vs_".join(models)
        out_dir = results_dir / f"comparison_{tag}"
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "comparison.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        _write_summary_csv(out_dir / "comparison_summary.csv", models, domains_cmp)
        _write_demographic_csv(out_dir / "comparison_demographic.csv", models, demo_cmp)
        _write_questions_csv(out_dir / "comparison_questions.csv", models, pq_cmp)

        print(f"  Outputs: {out_dir}/")

    return result


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True, nargs="+")
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

    comparison = compare_models(
        models=args.models,
        results_dir=results_dir,
        domains=domains,
    )

    print(f"\n{'='*60}")
    print(f"  Comparison: {' vs '.join(args.models)}")
    print(f"{'='*60}")
    print(f"\n  {'Domain':<12}  " + "  ".join(f"{m[:20]:<22}" for m in args.models))
    print(f"  {'-'*12}  " + "  ".join("-"*22 for _ in args.models))
    for domain, model_vals in comparison["domains"].items():
        row = f"  {domain:<12}  "
        for model in args.models:
            s = model_vals.get(model, {})
            row += f"bias={str(s.get('bias_mean')):<12}  "
        print(row)

    print(f"\n  Overall demographics — bias_mean per group:")
    for dim in DIMS:
        print(f"  {dim}:")
        for grp, domain_vals in sorted(comparison["demographics"][dim].items()):
            overall_vals = domain_vals.get("overall", {})
            vals = "  ".join(
                f"{m}={overall_vals.get(m, {}).get('bias_mean')}"
                for m in args.models
            )
            print(f"    {grp:<12}  {vals}")
