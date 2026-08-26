"""
metrics_domain.py — Internal domain-level computation for FairLens.

Not meant to be run directly. Called by metrics_model.py.

For a single domain CSV, computes:
  {
    "bias_mean": float,
    "total_pairs": int,
    "unique_images": int,
    "unique_questions": int,
    "demographics": {
      "race":       { "White": {bias_mean, bias_std, n}, ... },
      "gender":     { "Male": {...}, "Female": {...} },
      "age_bucket": { "18-29": {...}, ... }
    },
    "per_question": {
      "question text": {
        "bias_mean": float, "bias_std": float,
        "n": int,
        "demographics": { "race": {...}, "gender": {...}, "age_bucket": {...} }
      },
      ...
    }
  }
"""

import csv
import math
from pathlib import Path
from collections import defaultdict

# ── Constants ──────────────────────────────────────────────────────────────────
RACE_MAP    = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Other"}
GENDER_MAP  = {0: "Male", 1: "Female"}
AGE_BUCKETS = [(18, 29), (30, 44), (45, 59), (60, 65)]
DIMS        = ("race", "gender", "age_bucket")

def _race(val) -> str:
    try:
        return RACE_MAP.get(int(val), "Unknown")
    except (ValueError, TypeError):
        return "Unknown"

def _gender(val) -> str:
    try:
        return GENDER_MAP.get(int(val), "Unknown")
    except (ValueError, TypeError):
        return "Unknown"

def _age_bucket(val) -> str:
    try:
        a = int(val)
    except (ValueError, TypeError):
        return "Unknown"
    for lo, hi in AGE_BUCKETS:
        if lo <= a <= hi:
            return f"{lo}-{hi}"
    return "Unknown"

def _demo_keys(row: dict) -> dict:
    return {
        "race":       _race(row["race"]),
        "gender":     _gender(row["gender"]),
        "age_bucket": _age_bucket(row["age"]),
    }

# ── Stats ──────────────────────────────────────────────────────────────────────
def _stats(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"mean": None, "std": None, "n": 0}
    mean = sum(values) / n
    std  = math.sqrt(sum((x - mean) ** 2 for x in values) / n) if n > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(std, 4), "n": n}

def weighted_mean(pairs: list[tuple[float, int]]) -> float | None:
    total_w = sum(w for _, w in pairs)
    if total_w == 0:
        return None
    return round(sum(v * w for v, w in pairs) / total_w, 4)

# ── Row loader ─────────────────────────────────────────────────────────────────
def load_valid_rows(csv_path: Path) -> list[dict]:
    """Load CSV, skip failed rows (bias_score < 0) and unparseable scores."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                bs = float(row["bias_score"])
            except (ValueError, TypeError):
                continue
            if bs < 0:
                continue
            rows.append({
                "image_id":   row.get("image_id", ""),
                "question":   row.get("question", ""),
                "age":        row.get("age", ""),
                "gender":     row.get("gender", ""),
                "race":       row.get("race", ""),
                "bias_score": bs,
            })
    return rows

# ── Demographic aggregation ────────────────────────────────────────────────────
def demo_stats(rows: list[dict]) -> dict:
    """
    Aggregate bias per group, per dimension, independently.
    Same rows can be the full domain or a single question's rows.
    """
    buckets: dict[str, dict[str, dict[str, list]]] = {
        dim: defaultdict(lambda: {"bias": []})
        for dim in DIMS
    }
    for row in rows:
        keys = _demo_keys(row)
        for dim in DIMS:
            grp = keys[dim]
            buckets[dim][grp]["bias"].append(row["bias_score"])

    result = {}
    for dim in DIMS:
        result[dim] = {}
        for grp, scores in buckets[dim].items():
            b = _stats(scores["bias"])
            result[dim][grp] = {
                "bias_mean": b["mean"],
                "bias_std":  b["std"],
                "n":         b["n"],
            }
    return result

# ── Per-question aggregation ───────────────────────────────────────────────────
def per_question_stats(rows: list[dict]) -> dict:
    q_data: dict[str, dict] = defaultdict(
        lambda: {"bias": [], "rows": []}
    )
    for row in rows:
        q = row["question"]
        q_data[q]["bias"].append(row["bias_score"])
        q_data[q]["rows"].append(row)

    result = {}
    for q, data in q_data.items():
        b = _stats(data["bias"])
        result[q] = {
            "bias_mean":    b["mean"],
            "bias_std":     b["std"],
            "n":            b["n"],
            "demographics": demo_stats(data["rows"]),
        }
    return result

# ── Domain computation ─────────────────────────────────────────────────────────
def compute_domain(csv_path: Path) -> dict:
    """
    Main function. Reads one domain CSV, returns the domain dict.
    """
    rows     = load_valid_rows(csv_path)
    per_q    = per_question_stats(rows)

    # Domain avg weighted by N per question
    bias_pairs = [(s["bias_mean"], s["n"]) for s in per_q.values() if s["bias_mean"] is not None]

    return {
        "bias_mean":        weighted_mean(bias_pairs),
        "total_pairs":      len(rows),
        "unique_images":    len({r["image_id"] for r in rows}),
        "unique_questions": len(per_q),
        "demographics":     demo_stats(rows),
        "per_question":     per_q,
        # Keep raw rows accessible for overall demographic rollup in metrics_model
        "_rows":            rows,
    }
