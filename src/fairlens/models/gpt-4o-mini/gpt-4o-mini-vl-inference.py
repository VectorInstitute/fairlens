"""GPT-4o-mini vision-language inference."""
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
args = parser.parse_args()

DATASET_TYPE = args.dataset
MODEL_VARIANT = "gpt-4o-mini"
MODEL_VERSION  = "gpt-4o-mini"

import sys
from pathlib import Path as _Path
_PKG = _Path(__file__).resolve().parents[2]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))
from config.load import fairlens_root, load_questions, max_tokens as _model_max_tokens

SYSTEM_PROMPTS, DOMAIN_QUESTIONS = load_questions()
MAX_TOKENS = _model_max_tokens('gpt-4o-mini')
_ROOT = fairlens_root()
DATASET_PATH = str(_ROOT / "dataset" / "UTKface" / DATASET_TYPE / "all_images")
TEST_CSV = str(_ROOT / "dataset" / "UTKface_test.csv")

RESULTS_DIR = f"results_{DATASET_TYPE}_{MODEL_VARIANT}"
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Running {DATASET_TYPE} dataset using {MODEL_VARIANT}")
print(f"Images path: {DATASET_PATH}")
print(f"Results folder: {RESULTS_DIR}")

client = OpenAI()

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

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

        response = client.chat.completions.create(
            model=MODEL_VERSION,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                        {
                            "type": "text",
                            "text": full_prompt,
                        },
                    ],
                }
            ],
            max_tokens=MAX_TOKENS,
        )
        answer = response.choices[0].message.content.strip()
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

    # Resume from previous run if results exist
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            domain_results = json.load(f)
    else:
        domain_results = {}

    print(f"\nRunning inference on {len(test_df)} images for '{domain}' domain...")

    for row in tqdm(test_df.itertuples(), total=len(test_df)):
        if row.filename in domain_results:
            continue  # skip already processed

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

        # Save incrementally after each image
        with open(results_file, "w") as f:
            json.dump(domain_results, f, indent=4)

    print(f"=== Completed '{domain}' domain ===")
    print(f"Saved to {results_file}")

print("\n=== All domains completed ===")