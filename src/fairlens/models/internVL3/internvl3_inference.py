import os
import json
import argparse
import math
from PIL import Image
from tqdm import tqdm

import torch
import pandas as pd
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from transformers import AutoModel, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    choices=["cropped", "uncropped"],
    required=True,
    help="Choose which dataset version to use"
)
args = parser.parse_args()
DATASET_TYPE = args.dataset

import sys
from pathlib import Path as _Path
_PKG = _Path(__file__).resolve().parents[2]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))
from config.load import fairlens_root, load_questions, max_tokens as _model_max_tokens

SYSTEM_PROMPTS, DOMAIN_QUESTIONS = load_questions()
MAX_TOKENS = _model_max_tokens('internVL3')
_ROOT = fairlens_root()
DATASET_PATH = str(_ROOT / "dataset" / "UTKface" / DATASET_TYPE / "all_images")
TEST_CSV = str(_ROOT / "dataset" / "UTKface_test.csv")

MODEL_VARIANT = "InternVL3-9B"
MODEL_PATH = "OpenGVLab/InternVL3-9B"

RESULTS_DIR = f"results_{DATASET_TYPE}_{MODEL_VARIANT}"
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Running {DATASET_TYPE} dataset using {MODEL_VARIANT}")
print(f"Images path: {DATASET_PATH}")
print(f"Results folder: {RESULTS_DIR}")

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_grad_enabled(False)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    return T.Compose([
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ar)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = ratio
        elif diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / orig_h

    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
                 for i in range(1, n + 1)
                 for j in range(1, n + 1)
                 if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1]
    )

    target_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_w, orig_h, image_size
    )

    target_w = image_size * target_ratio[0]
    target_h = image_size * target_ratio[1]
    blocks = target_ratio[0] * target_ratio[1]

    resized = image.resize((target_w, target_h))
    images = []

    for i in range(blocks):
        box = (
            (i % (target_w // image_size)) * image_size,
            (i // (target_w // image_size)) * image_size,
            ((i % (target_w // image_size)) + 1) * image_size,
            ((i // (target_w // image_size)) + 1) * image_size,
        )
        images.append(resized.crop(box))

    return images

def load_image(image_path, input_size=448, max_num=6):
    image = Image.open(image_path).convert("RGB")
    transform = build_transform(input_size)
    tiles = dynamic_preprocess(image, image_size=input_size, max_num=max_num)
    pixel_values = torch.stack([transform(t) for t in tiles])
    return pixel_values

print("Loading InternVL3...")

model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    device_map="cuda"
).eval()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    use_fast=False
)

generation_config = dict(
    max_new_tokens=MAX_TOKENS,
    do_sample=False
)

test_df = pd.read_csv(TEST_CSV)
print(f"Loaded test set: {len(test_df)} images")

def run_inference(image_path, questions, system_prompt):
    pixel_values = load_image(image_path, max_num=6).to(torch.bfloat16).cuda()
    results = {}

    for q in questions:
        prompt = system_prompt + q
        if "Options:" in q:
            prompt += "\n\nPlease select one of the provided options and output only that."
        else:
            prompt += "\n\nAnswer the question as best as you can."

        full_prompt = "<image>\n" + prompt

        with torch.no_grad():
            response = model.chat(
                tokenizer,
                pixel_values,
                full_prompt,
                generation_config
            )

        results[q] = [response]

    return results

for domain, questions in DOMAIN_QUESTIONS.items():
    results_file = os.path.join(RESULTS_DIR, f"{domain}_results.json")

    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            domain_results = json.load(f)
    else:
        domain_results = {}

    print(f"Running inference on {len(test_df)} images for {domain} domain...")

    for row in tqdm(test_df.itertuples(), total=len(test_df)):
        if row.filename in domain_results:
            continue

        image_path = os.path.join(DATASET_PATH, row.filename)

        answers = run_inference(
            image_path,
            questions,
            SYSTEM_PROMPTS[domain]
        )

        domain_results[row.filename] = {
            "path": image_path,
            "dataset_type": DATASET_TYPE,
            "age": row.age,
            "gender": row.gender,
            "race": row.race,
            "answers": answers
        }

        with open(results_file, "w") as f:
            json.dump(domain_results, f, indent=4)

    print(f"=== Completed {domain} domain ===")
    print(f"Saved to {results_file}")
