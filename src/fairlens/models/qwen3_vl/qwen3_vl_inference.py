import os
import json
import argparse
from PIL import Image
from tqdm import tqdm
import torch
import pandas as pd
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    choices=["cropped", "uncropped"],
    required=True,
    help="Choose which dataset version to use"
)
args = parser.parse_args()
DATASET_TYPE = args.dataset
MODEL_VARIANT = "Qwen3-VL-8B-Instruct"

import sys
from pathlib import Path as _Path
_PKG = _Path(__file__).resolve().parents[2]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))
from config.load import fairlens_root, load_questions, max_tokens as _model_max_tokens

SYSTEM_PROMPTS, DOMAIN_QUESTIONS = load_questions()
MAX_TOKENS = _model_max_tokens('qwen3_vl')
_ROOT = fairlens_root()
DATASET_PATH = str(_ROOT / "dataset" / "UTKface" / DATASET_TYPE / "all_images")
TEST_CSV = str(_ROOT / "dataset" / "UTKface_test.csv")

MODEL_DIR = "Qwen/Qwen3-VL-8B-Instruct"
RESULTS_DIR = f"results_{DATASET_TYPE}_{MODEL_VARIANT.replace('/', '_')}"
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Running {DATASET_TYPE} dataset using {MODEL_VARIANT}")
print(f"Images path: {DATASET_PATH}")
print(f"Results folder: {RESULTS_DIR}")

print("Loading model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_DIR)

if os.path.exists(TEST_CSV):
    test_df = pd.read_csv(TEST_CSV)
    print(f"Loaded test set: {len(test_df)} images")
else:
    raise FileNotFoundError(f"Evaluation CSV not found: {TEST_CSV}")

def run_inference(image_path, questions, system_prompt):
    image = Image.open(image_path).convert("RGB")
    results = {}
    for q in questions:
        prompt = system_prompt + q
        if "Options:" in q:
            prompt += "\n\nPlease select one of the provided options and output only that."
        else:
            prompt += "\n\nAnswer the question as best as you can."
        
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_TOKENS)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        results[q] = output_text
    return results

for domain, questions in DOMAIN_QUESTIONS.items():
    results_file = os.path.join(RESULTS_DIR, f"{domain}_results.json")
    
    # Load previous results if any
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            domain_results = json.load(f)
    else:
        domain_results = {}
    
    print(f"Running inference on {len(test_df)} images for {domain} domain...")
    
    for row in tqdm(test_df.itertuples(), total=len(test_df)):
        if row.filename in domain_results:
            continue
        
        image_path = os.path.join(DATASET_PATH, str(row.filename))
        answers = run_inference(image_path, questions, SYSTEM_PROMPTS[domain])
        domain_results[row.filename] = {
            "path": image_path,
            "dataset_type": DATASET_TYPE,
            "age": row.age,
            "gender": row.gender,
            "race": row.race,
            "answers": answers
        }
        
        # Save progress incrementally
        with open(results_file, "w") as f:
            json.dump(domain_results, f, indent=4)
    
    print(f"=== Completed {domain} domain ===")
    print(f"Saved to {results_file}")
