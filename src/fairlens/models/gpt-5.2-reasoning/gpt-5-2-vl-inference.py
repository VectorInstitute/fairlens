"""GPT-5.2 vision-language inference. --retry-failed re-runs empty or short open-ended answers."""
import os
import json
import base64
import argparse
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    choices=["cropped", "uncropped"],
    required=True,
    help="Choose which dataset version to use"
)
parser.add_argument(
    "--retry-failed",
    action="store_true",
    help="Re-run only empty or suspiciously short open-ended answers (< 5 words)"
)
args = parser.parse_args()

DATASET_TYPE  = args.dataset
RETRY_FAILED  = args.retry_failed
MODEL_VARIANT = "gpt-5.2"
MODEL_VERSION = "gpt-5.2"

import sys
from pathlib import Path as _Path
_PKG = _Path(__file__).resolve().parents[2]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))
from config.load import fairlens_root, load_questions, max_tokens as _model_max_tokens

SYSTEM_PROMPTS, DOMAIN_QUESTIONS = load_questions()
MAX_TOKENS = _model_max_tokens('gpt-5.2-reasoning')
_ROOT = fairlens_root()
DATASET_PATH = str(_ROOT / "dataset" / "UTKface" / DATASET_TYPE / "all_images")
TEST_CSV = str(_ROOT / "dataset" / "UTKface_test.csv")

RESULTS_DIR = f"results_{DATASET_TYPE}_{MODEL_VARIANT}"
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Running {DATASET_TYPE} dataset using {MODEL_VARIANT}")
print(f"Images path: {DATASET_PATH}")
print(f"Results folder: {RESULTS_DIR}")
if RETRY_FAILED:
    print("Mode: --retry-failed (open-ended: empty or < 5 words | closed-ended: empty only)")

def is_open_ended(question: str) -> bool:
    return "Options:" not in question

def needs_retry(answer: str) -> bool:
    """True if answer is empty or suspiciously short (< 5 words) for open-ended."""
    stripped = answer.strip()
    if not stripped:
        return True
    return len(stripped.split()) < 5

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

client = OpenAI()

def run_inference(image_path: str, questions: list, system_prompt: str) -> dict:
    base64_image = encode_image_to_base64(image_path)
    results = {}
    for q in questions:
        if "Options:" in q:
            instruction = "Please select one of the provided options and output only that option."
        else:
            instruction = "Answer the question as best as you can."

        full_prompt = (
            f"{system_prompt}"
            f"{q}\n\n"
            f"{instruction}"
        )

        response = client.responses.create(
            model=MODEL_VERSION,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        },
                        {
                            "type": "input_text",
                            "text": full_prompt,
                        },
                    ],
                }
            ],
            reasoning={"effort": "low"},
            max_output_tokens=MAX_TOKENS,
        )
        answer = response.output_text.strip()
        results[q] = [answer]
    return results

import pandas as pd

if os.path.exists(TEST_CSV):
    test_df = pd.read_csv(TEST_CSV)
    print(f"Loaded test set: {len(test_df)} images")
else:
    raise FileNotFoundError(f"Evaluation CSV not found: {TEST_CSV}")

for domain, questions in DOMAIN_QUESTIONS.items():
    results_file = os.path.join(RESULTS_DIR, f"{domain}_results.json")

    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            domain_results = json.load(f)
    else:
        domain_results = {}

    if RETRY_FAILED:
        # ── Retry mode: only open-ended questions with empty / < 5-word answers ──
        retry_targets = {}  # { filename: [questions_to_retry] }
        for filename, entry in domain_results.items():
            bad_qs = [
                q for q in questions
                if is_open_ended(q)
                and needs_retry(entry.get("answers", {}).get(q, [""])[0])
                or not is_open_ended(q)
                and not entry.get("answers", {}).get(q, [""])[0].strip()
            ]
            if bad_qs:
                retry_targets[filename] = bad_qs

        if not retry_targets:
            print(f"[{domain}] No failed open-ended answers found, skipping.")
            continue

        total_pairs = sum(len(qs) for qs in retry_targets.values())
        print(f"\n[{domain}] Retrying {total_pairs} open-ended answers across {len(retry_targets)} images...")

        for filename, bad_questions in tqdm(retry_targets.items(), desc=f"retry | {domain}"):
            image_path = os.path.join(DATASET_PATH, str(filename))
            if not os.path.exists(image_path):
                print(f"[WARN] Image not found, skipping: {image_path}")
                continue
            try:
                answers = run_inference(image_path, bad_questions, SYSTEM_PROMPTS[domain])
            except Exception as e:
                print(f"[ERROR] Failed on {filename}: {e}")
                answers = {q: [f"ERROR: {e}"] for q in bad_questions}

            # Merge retried answers back into existing entry
            domain_results[filename]["answers"].update(answers)

            with open(results_file, "w") as f:
                json.dump(domain_results, f, indent=4)

        print(f"=== Retry complete for '{domain}' domain ===")

    else:
        # ── Normal mode ──
        print(f"\nRunning inference on {len(test_df)} images for '{domain}' domain...")
        for row in tqdm(test_df.itertuples(), total=len(test_df)):
            if row.filename in domain_results:
                continue

            image_path = os.path.join(DATASET_PATH, str(row.filename))
            if not os.path.exists(image_path):
                print(f"[WARN] Image not found, skipping: {image_path}")
                continue

            try:
                answers = run_inference(image_path, questions, SYSTEM_PROMPTS[domain])
            except Exception as e:
                print(f"[ERROR] Failed on {row.filename}: {e}")
                answers = {q: [f"ERROR: {e}"] for q in questions}

            domain_results[row.filename] = {
                "path":         image_path,
                "dataset_type": DATASET_TYPE,
                "age":          row.age,
                "gender":       row.gender,
                "race":         row.race,
                "answers":      answers,
            }

            with open(results_file, "w") as f:
                json.dump(domain_results, f, indent=4)

        print(f"=== Completed '{domain}' domain ===")
        print(f"Saved to {results_file}")

print("\n=== All domains completed ===")