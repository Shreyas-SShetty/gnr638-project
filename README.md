# Deep Learning MCQ Solver

Solves PNG images of deep learning multiple-choice questions using
**Qwen2.5-VL-7B-Instruct**, a vision-language model with strong OCR and
reasoning capabilities. A 3-prompt chain-of-thought ensemble with majority
voting is used.
---

## Automated Setup (grading)

```bash
bash setup.bash          # clones repo, creates env, downloads weights
conda activate gnr_project_env
python inference.py --test_dir <absolute_path_to_test_dir>
```

---

## Expected directory layout at inference time

```
<working_dir>/
├── inference.py
├── download_model.py
├── requirements.txt
├── saved_qwen25vl_7b/      ← downloaded by download_model.py
└── submission.csv          ← created by inference.py
```

```
<test_dir>/
├── images/
│   ├── image_1.png
│   └── ...
└── test.csv
```

Output `submission.csv` is written to the working directory (not test_dir).

Expected runtime on 48 GB L40s: **~10–15 minutes** for 50 questions.

---

## Citations

- Qwen2.5-VL: Wang et al., "Qwen2.5-VL Technical Report", 2025.
  https://github.com/QwenLM/Qwen2.5-VL
- HuggingFace Transformers: https://github.com/huggingface/transformers
- qwen-vl-utils: https://github.com/QwenLM/qwen-vl-utils

## Honor Code

- Codex coding agent for debugging purposes.
