"""Load FairLens questions and per-model max_tokens. Stdlib only."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


CONFIG_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = CONFIG_DIR.parent  # src/fairlens
FAIRLENS_ROOT = PACKAGE_DIR.parent.parent  # FairLens repo root


def fairlens_root() -> Path:
    return FAIRLENS_ROOT


def package_dir() -> Path:
    return PACKAGE_DIR


def models_dir() -> Path:
    return PACKAGE_DIR / "models"


def results_dir() -> Path:
    return FAIRLENS_ROOT / "results"


def _norm_apostrophe(text: str) -> str:
    return text.replace("\u2019", "'").replace("\u2018", "'")


@lru_cache(maxsize=1)
def _questions_data() -> dict[str, Any]:
    path = CONFIG_DIR / "questions.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _models_config() -> tuple[int, dict[str, int]]:
    path = CONFIG_DIR / "models.yaml"
    default = 128
    models: dict[str, int] = {}
    section = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith("default_max_tokens:"):
            default = int(line.split(":", 1)[1].strip())
            continue
        if line.strip() == "models:":
            section = "models"
            continue
        if section == "models" and ":" in line:
            key, value = line.strip().split(":", 1)
            models[key.strip()] = int(value.strip())
    return default, models


def system_prompts() -> dict[str, str]:
    return dict(_questions_data()["system_prompts"])


def load_questions() -> tuple[dict[str, str], dict[str, list[str]]]:
    data = _questions_data()
    domain_questions = {
        domain: [item["text"] for item in data[domain]]
        for domain in ("hiring", "legal", "healthcare")
    }
    return system_prompts(), domain_questions


def open_ended_questions(domain: str | None = None) -> dict[str, list[str]] | list[str]:
    data = _questions_data()
    if domain is not None:
        return [item["text"] for item in data[domain] if item["type"] == "open"]
    return {
        d: [item["text"] for item in data[d] if item["type"] == "open"]
        for d in ("hiring", "legal", "healthcare")
    }


def soundness_gt(domain: str | None = None) -> dict[str, str] | dict[str, dict[str, str]]:
    data = _questions_data()

    def for_domain(d: str) -> dict[str, str]:
        return {item["text"]: item["expected"] for item in data[d] if "expected" in item}

    if domain is not None:
        return for_domain(domain)
    return {d: for_domain(d) for d in ("hiring", "legal", "healthcare")}


def parity_outcomes(domain: str | None = None) -> dict[str, dict[str, str]] | dict[str, dict[str, dict[str, str]]]:
    data = _questions_data()

    def for_domain(d: str) -> dict[str, dict[str, str]]:
        return {
            item["text"]: {"target": item["adverse"], "meaning": "adverse"}
            for item in data[d]
            if "adverse" in item
        }

    if domain is not None:
        return for_domain(domain)
    return {d: for_domain(d) for d in ("hiring", "legal", "healthcare")}


def association_questions(domain: str | None = None) -> dict[str, str] | dict[str, dict[str, str]]:
    data = _questions_data()

    def for_domain(d: str) -> dict[str, str]:
        return {item["text"]: item["association"] for item in data[d] if "association" in item}

    if domain is not None:
        return for_domain(domain)
    return {d: for_domain(d) for d in ("hiring", "legal", "healthcare")}


def max_tokens(model_folder: str) -> int:
    default, models = _models_config()
    return models.get(model_folder, default)


def match_question_key(question: str, mapping: dict[str, Any]) -> str | None:
    """Return the mapping key that matches question, ignoring curly vs straight apostrophes."""
    if question in mapping:
        return question
    target = _norm_apostrophe(question)
    for key in mapping:
        if _norm_apostrophe(key) == target:
            return key
    return None


def lookup(mapping: dict[str, Any], question: str, default: Any = None) -> Any:
    key = match_question_key(question, mapping)
    if key is None:
        return default
    return mapping[key]
