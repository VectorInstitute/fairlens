import os
import json
import argparse
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    choices=["cropped", "uncropped"],
    required=True,
    help="Choose which dataset version to use"
)
args = parser.parse_args()

DATASET_TYPE = args.dataset
MODEL_VARIANT = "CogVLM2-llama3-chat-19B"

import sys
from pathlib import Path as _Path
_PKG = _Path(__file__).resolve().parents[2]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))
from config.load import fairlens_root, load_questions, max_tokens as _model_max_tokens

SYSTEM_PROMPTS, DOMAIN_QUESTIONS = load_questions()
MAX_TOKENS = _model_max_tokens('cogvlm')
_ROOT = fairlens_root()
DATASET_PATH = str(_ROOT / "dataset" / "UTKface" / DATASET_TYPE / "all_images")
TEST_CSV = str(_ROOT / "dataset" / "UTKface_test.csv")

MODEL_PATH = "THUDM/cogvlm2-llama3-chat-19B"
RESULTS_DIR = f"results_{DATASET_TYPE}_{MODEL_VARIANT.replace('/', '_')}"

os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Running {DATASET_TYPE} dataset using {MODEL_VARIANT}")
print(f"Images path: {DATASET_PATH}")
print(f"Results folder: {RESULTS_DIR}")

print("Loading model...")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation='eager'
).to(DEVICE).eval()

print(f"Model loaded on {DEVICE} with dtype {torch.bfloat16}")

import pandas as pd

if os.path.exists(TEST_CSV):
    test_df = pd.read_csv(TEST_CSV)
    print(f"Loaded test set: {len(test_df)} images")
else:
    raise FileNotFoundError(f"Evaluation CSV not found: {TEST_CSV}")

def run_inference(image_path, questions, system_prompt):
    image = Image.open(image_path).convert("RGB")
    results = {}
    
    for q in questions:
        prompt = system_prompt + q + "\n\n"
        if "Options:" in q:
            prompt += "Please select one of the provided options and output only that."
        else:
            prompt += "Answer the question as best as you can."
        
        # Build conversation input
        input_by_model = model.build_conversation_input_ids(
            tokenizer,
            query=prompt,
            history=[],
            images=[image],
            template_version='chat'
        )
        
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
            'images': [[input_by_model['images'][0].to(DEVICE).to(torch.bfloat16)]],
        }
        
        gen_kwargs = {
            "max_new_tokens": MAX_TOKENS,
            "pad_token_id": 128002,
        }
        
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0])
            response = response.split("<|end_of_text|>")[0].strip()
        
        results[q] = [response]
    
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

print("\n=== All domains completed ===")