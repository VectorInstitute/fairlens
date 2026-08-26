from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_PKG = Path(__file__).resolve().parents[2]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))
from config.load import match_question_key, soundness_gt

GENDER_MAP = {0: "Male", 1: "Female", "0": "Male", "1": "Female"}
RACE_MAP = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Other"}
GROUND_TRUTH = soundness_gt()


def load_results(file_path: str) -> dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_domain(file_path: str) -> str:
    stem = Path(file_path).name
    for domain in GROUND_TRUTH:
        if stem.startswith(f"{domain}_"):
            return domain
    raise ValueError("Could not infer domain from filename. Pass --domain explicitly.")


def age_group(age: int | float | str) -> str:
    age_value = int(age)
    if age_value < 30:
        return "young"
    if age_value < 60:
        return "middle"
    return "senior"


def answer_text(raw_answer: Any) -> str:
    if isinstance(raw_answer, list):
        if not raw_answer:
            return ""
        raw_answer = raw_answer[0]
    if raw_answer is None:
        return ""
    return str(raw_answer).strip()


def parse_options(question: str) -> list[str]:
    if "Options:" not in question:
        return []
    return [part.strip() for part in question.split("Options:", 1)[1].split("/") if part.strip()]


def clean_answer(answer: str) -> str:
    answer = re.sub(r"\s+", " ", answer.strip())
    return answer.strip(" \t\n\r\"'`.,;:!?")


def canonical_answer(raw_answer: Any, options: list[str]) -> str:
    answer = clean_answer(answer_text(raw_answer))
    if not options:
        return answer

    answer_lower = answer.lower()
    for option in options:
        if answer_lower == option.lower():
            return option

    if len(answer) >= 2:
        prefix_matches = [option for option in options if option.lower().startswith(answer_lower)]
        if len(prefix_matches) == 1:
            return prefix_matches[0]

    if len(answer) == 1:
        letter_matches = [option for option in options if option[:1].lower() == answer_lower]
        if len(letter_matches) == 1:
            return letter_matches[0]

    contained = [
        option
        for option in options
        if re.search(rf"(^|[^\w-]){re.escape(option.lower())}($|[^\w-])", answer_lower)
    ]
    if len(contained) == 1:
        return contained[0]

    return answer


def answers_match(raw_answer: Any, expected: str, question: str) -> bool:
    answer = canonical_answer(raw_answer, parse_options(question))
    return answer.lower() == expected.lower()


def demographic_groups(entry: dict[str, Any]) -> dict[str, str]:
    race = entry.get("race")
    return {
        "gender": GENDER_MAP.get(entry.get("gender"), str(entry.get("gender"))),
        "race": RACE_MAP.get(race, str(race)),
        "age": age_group(entry.get("age")),
    }


def empty_count() -> dict[str, int]:
    return {"correct": 0, "total": 0}


def accuracy(counts: dict[str, int]) -> float:
    if counts["total"] == 0:
        return 0.0
    return counts["correct"] / counts["total"]


def summarize_counts(group_counts: dict[str, dict[str, int]]) -> dict[str, Any]:
    summary = {}
    for group in sorted(group_counts):
        counts = group_counts[group]
        score = accuracy(counts)
        summary[group] = {
            "correct": counts["correct"],
            "total": counts["total"],
            "accuracy": round(score, 4),
            "pct": round(score * 100, 2),
        }
    return summary


def evaluate_soundness(data: dict[str, Any], domain: str) -> dict[str, Any]:
    ground_truth = GROUND_TRUTH[domain]
    overall = {"correct": 0, "total": 0}
    by_question = {question: {"expected": expected, "correct": 0, "total": 0} for question, expected in ground_truth.items()}
    by_demographic = {
        "gender": defaultdict(empty_count),
        "race": defaultdict(empty_count),
        "age": defaultdict(empty_count),
    }

    for entry in data.values():
        groups = demographic_groups(entry)
        for question, raw_answer in entry.get("answers", {}).items():
            key = match_question_key(question, ground_truth)
            if key is None:
                continue

            expected = ground_truth[key]
            correct = answers_match(raw_answer, expected, key)
            overall["total"] += 1
            by_question[key]["total"] += 1
            if correct:
                overall["correct"] += 1
                by_question[key]["correct"] += 1

            for attr, group in groups.items():
                by_demographic[attr][group]["total"] += 1
                if correct:
                    by_demographic[attr][group]["correct"] += 1

    return {
        "domain": domain,
        "ground_truth": ground_truth,
        "overall": {
            "correct": overall["correct"],
            "total": overall["total"],
            "accuracy": round(accuracy(overall), 4),
            "pct": round(accuracy(overall) * 100, 2),
        },
        "by_demographic": {
            attr: summarize_counts(dict(group_counts))
            for attr, group_counts in by_demographic.items()
        },
        "by_question": {
            question: {
                "expected": details["expected"],
                "correct": details["correct"],
                "total": details["total"],
                "accuracy": round(accuracy(details), 4),
                "pct": round(accuracy(details) * 100, 2),
            }
            for question, details in by_question.items()
        },
    }


def print_summary(results: dict[str, Any]) -> None:
    overall = results["overall"]
    print(f"Soundness evaluation ({results['domain']})")
    print(f"Overall: {overall['correct']} / {overall['total']} ({overall['pct']}%)")

    for attr, groups in results["by_demographic"].items():
        print(f"\n--- {attr.upper()} ---")
        for group, values in groups.items():
            print(f"{group}: {values['correct']} / {values['total']} ({values['pct']}%)")

    print("\n--- QUESTION-LEVEL SOUNDNESS ---")
    for question, details in results["by_question"].items():
        short_question = question.split(" Options:", 1)[0]
        print(f"{details['pct']:6.2f}% | expected={details['expected']} | {short_question}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate whether model answers follow logical ground truth."
    )
    parser.add_argument("result_json", help="Path to a model/domain *_results.json file.")
    parser.add_argument(
        "--domain",
        choices=sorted(GROUND_TRUTH),
        help="Domain name. Inferred from filename when omitted.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    domain = args.domain or infer_domain(args.result_json)
    results = evaluate_soundness(load_results(args.result_json), domain)
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_summary(results)


if __name__ == "__main__":
    main()
