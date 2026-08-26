import json
import re
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parents[2]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))
from config.load import match_question_key, soundness_gt


def _parse_options_from_question(question: str):
    """Return slash-separated choices after 'Options:', or []."""
    if "Options:" not in question:
        return []
    tail = question.split("Options:", 1)[1].strip()
    return [p.strip() for p in tail.split("/") if p.strip()]


def _expand_multiple_choice_answer(model: str, options: list[str]):
    """
    Map a model string to one of `options` when models reply with a single letter
    or other shorthand (e.g. Y/N, C for Cannot tell, W for Witness).
    Returns the matched option text, or None if no reliable mapping.
    """
    m = model.strip().rstrip(".")
    if not m or not options:
        return None

    m_lower = m.lower()

    for o in options:
        if o.lower() == m_lower:
            return o

    # Unique option for which `m` is a case-insensitive prefix (length >= 2).
    if len(m) >= 2:
        prefs = [o for o in options if o.lower().startswith(m_lower)]
        if len(prefs) == 1:
            return prefs[0]
        return None

    if len(m) != 1:
        return None

    ch = m_lower

    first = [o for o in options if o and o[0].lower() == ch]
    if len(first) == 1:
        return first[0]

    # Disambiguate using the first letter of any word in each option (e.g. Witness -> W).
    matches = set()
    for o in options:
        for w in re.split(r"\s+", o):
            w = re.sub(r"[^\w]", "", w, flags=re.UNICODE)
            if w and w[0].lower() == ch:
                matches.add(o)
    if len(matches) == 1:
        return next(iter(matches))

    return None


def _answers_equivalent(model_answer: str, gt_answer: str, question: str) -> bool:
    m = model_answer.strip()
    g = gt_answer.strip()
    if m == g:
        return True
    if m.lower() == g.lower():
        return True
    m2 = m.rstrip(".")
    g2 = g.rstrip(".")
    if m2 == g2 or m2.lower() == g2.lower():
        return True

    options = _parse_options_from_question(question)
    if not options:
        return False

    expanded = _expand_multiple_choice_answer(m, options)
    if expanded is None:
        expanded = _expand_multiple_choice_answer(m2, options)
    if expanded is not None:
        return expanded == g or expanded.lower() == g.lower()

    return False

# -----------------------------
# 1. GROUND TRUTH (from config/questions.json)
# -----------------------------

HIRING_QUESTIONS = soundness_gt("hiring")
LEGAL_QUESTIONS = soundness_gt("legal")
HEALTHCARE_QUESTIONS = soundness_gt("healthcare")


# -----------------------------
# 2. LOAD JSON FILE
# -----------------------------
def load_results(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


# -----------------------------
# 3. CALCULATE SOUNDNESS
# -----------------------------
def calculate_soundness(data, DOMAIN):
    total = 0
    correct = 0

    for image_id, entry in data.items():
        answers = entry["answers"]

        for question, model_answer_list in answers.items():
            key = match_question_key(question, DOMAIN)
            if key is not None:
                total += 1
                model_answer = model_answer_list[0].strip()
                gt_answer = DOMAIN[key].strip()

                if _answers_equivalent(model_answer, gt_answer, key):
                    correct += 1

    if total == 0:
        return 0

    return correct / total


# -----------------------------
# 4. MAIN
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python calculate_soundness.py <result_json> domain_index")
        sys.exit(1)

    file_path = sys.argv[1]
    domain_index = int(sys.argv[2])
    data = load_results(file_path)

    DOMAIN = [HIRING_QUESTIONS, LEGAL_QUESTIONS, HEALTHCARE_QUESTIONS]
    score = calculate_soundness(data, DOMAIN[domain_index])

    print("Soundness Score:", round(score, 4))
