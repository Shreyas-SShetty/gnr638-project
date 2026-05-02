# =============================================================
# GNR638 Project: Deep Learning MCQ Solver
# Model: Qwen2.5-VL-7B-Instruct (offline inference)
# =============================================================
# References:
#   - Qwen2.5-VL: https://github.com/QwenLM/Qwen2.5-VL
#   - HuggingFace Transformers: https://github.com/huggingface/transformers
#   - qwen-vl-utils: https://github.com/QwenLM/qwen-vl-utils

import os
import re
import argparse
from collections import Counter

import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# =============================================================
# ARGS
# =============================================================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--test_dir",
    required=True,
    help="Absolute path to test directory containing test.csv and images/",
)
args = parser.parse_args()

# =============================================================
# PATHS
# =============================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR    = os.path.join(args.test_dir, "images")
TEST_CSV     = os.path.join(args.test_dir, "test.csv")
OUTPUT_CSV   = os.path.join(SCRIPT_DIR, "submission.csv")   # always in script dir
LOCAL_MODEL_DIR = os.path.join(SCRIPT_DIR, "saved_qwen25vl_7b")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================
# PROMPTS  (A=1, B=2, C=3, D=4 mapping explicit in each)
# =============================================================
PROMPTS = [
    """You are a deep learning expert solving an exam question.
Examine the MCQ image carefully.

IMPORTANT: Options A, B, C, D correspond to answer values 1, 2, 3, 4 respectively.

Instructions:
- Read the question and all options thoroughly
- Apply your deep learning knowledge step by step
- Identify the single correct answer

End your response with EXACTLY this line:
ANSWER: [digit]
where [digit] is 1 (for A), 2 (for B), 3 (for C), or 4 (for D).
Write ANSWER: 5 only if you are genuinely uncertain.""",

    """You are an expert deep learning engineer taking an exam.
Look at this multiple-choice question image.
Remember: A=1, B=2, C=3, D=4.

Use the elimination method:
- Go through each option and explain why it is correct or incorrect
- Select the best answer based on your reasoning

Finish your response with:
FINAL: [digit]
(1=A, 2=B, 3=C, 4=D, or 5 if unsure)""",

    """Solve this deep learning MCQ from the image.
Mapping: option A → 1, option B → 2, option C → 3, option D → 4.

Think through the problem:
1. What concept is being tested?
2. Which options can be eliminated and why?
3. What is the correct answer?

After reasoning, write:
ANS: [digit]
([digit] must be 1, 2, 3, 4, or 5 for uncertain)""",
]

# =============================================================
# ANSWER PARSER
# =============================================================
_PATTERNS = [
    r'ANSWER:\s*([1-5])',
    r'FINAL:\s*([1-5])',
    r'ANS:\s*([1-5])',
    r'answer\s+is\s+([1-5])\b',
    r'correct\s+(?:answer|option)\s+is\s+([1-5])\b',
]

def parse_answer(text: str) -> int:
    for pattern in _PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    digits = re.findall(r'\b([1-5])\b', text)
    return int(digits[-1]) if digits else 5


# =============================================================
# SINGLE INFERENCE
# =============================================================
def run_prompt(image: Image.Image, prompt: str) -> tuple[int, str]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=False,
        )

    generated_ids = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, output_ids)
    ]
    raw_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return parse_answer(raw_text), raw_text


# =============================================================
# ENSEMBLE PREDICTION
# =============================================================
def predict_image(image_path: str) -> int:
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  [ERROR] Could not load image: {e}")
        return 5

    votes = []
    for i, prompt in enumerate(PROMPTS):
        pred, raw = run_prompt(image, prompt)
        snippet = raw.strip().replace("\n", " ")[:100]
        print(f"  Prompt {i + 1}: pred={pred}  raw={snippet!r}")
        votes.append(pred)

    valid = [v for v in votes if v in (1, 2, 3, 4)]

    if not valid:
        return 5

    counts = Counter(valid)
    best_pred, best_count = counts.most_common(1)[0]

    # Require majority agreement (at least 2 of 3) to avoid -0.25 penalty
    if best_count < 2:
        return 5

    return best_pred


# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"Cannot find {TEST_CSV}")
    if not os.path.exists(IMAGE_DIR):
        raise FileNotFoundError(f"Cannot find images dir: {IMAGE_DIR}")

    print("Loading model (offline)...")
    processor = AutoProcessor.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        LOCAL_MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()
    print("Model loaded.\n")

    test_df = pd.read_csv(TEST_CSV)
    print(f"Running inference on {len(test_df)} images...\n")

    predictions = []
    for _, row in test_df.iterrows():
        image_name = str(row["image_name"])
        fname = image_name if image_name.endswith(".png") else f"{image_name}.png"
        image_path = os.path.join(IMAGE_DIR, fname)

        print(f"[{image_name}]")
        pred = predict_image(image_path)
        print(f"  => Final prediction: {pred}\n")

        predictions.append({
            "id": image_name,
            "image_name": image_name,
            "option": pred,
        })

    submission_df = pd.DataFrame(predictions)[["id", "image_name", "option"]]
    submission_df["_sort_key"] = submission_df["image_name"].str.extract(r'(\d+)$').astype(int)
    submission_df = submission_df.sort_values("_sort_key").drop(columns="_sort_key").reset_index(drop=True)
    submission_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {OUTPUT_CSV}")
    print(submission_df.to_string(index=False))
