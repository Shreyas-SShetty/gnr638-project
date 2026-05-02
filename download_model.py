# Run once during environment setup (internet required).
# Saves model locally so inference.py runs fully offline.

import os
from huggingface_hub import snapshot_download

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID   = "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"
LOCAL_DIR  = os.path.join(SCRIPT_DIR, "saved_qwen25vl_32b")

print(f"Downloading {MODEL_ID}")
print(f"Destination: {LOCAL_DIR}")
print("This is ~18 GB. Please wait...\n")

snapshot_download(repo_id=MODEL_ID, local_dir=LOCAL_DIR)

print(f"\nDone. Model saved to: {LOCAL_DIR}")
