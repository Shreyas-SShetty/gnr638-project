#!/bin/bash
set -e  # stop immediately if any command fails

# =============================================================
# GNR638 Project Setup
# Clones repo, creates conda env, installs deps, downloads model
# Internet is available at this stage.
# =============================================================

# 1. Clone repository into current directory
# TODO: replace with your actual public GitHub repo URL before submitting
git clone https://github.com/Shreyas788/gnr638-project.git .

# 2. Create conda environment with exact name and python version required
conda create -n gnr_project_env python=3.11 -y

# 3. Install Python dependencies into the environment
conda run -n gnr_project_env pip install -r requirements.txt

# 4. Download model weights from HuggingFace (~16 GB, internet required)
conda run -n gnr_project_env python download_model.py

echo ""
echo "Setup complete. Run:"
echo "  conda activate gnr_project_env"
echo "  python inference.py --test_dir <absolute_path_to_test_dir>"
