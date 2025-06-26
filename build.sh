#!/usr/bin/env bash

# Optional: Update apt and install curl (in case not present)
apt-get update && apt-get install -y curl

# OPTIONAL: Git LFS install (you no longer use it, but keeping for safety)
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install -y git-lfs

# OPTIONAL: Only needed if you use LFS again
git lfs install
git lfs pull || echo "Skipping git-lfs pull"

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# --- NEW: Download MRI model from Hugging Face (instead of renaming) --- #
echo "📦 Downloading MRI model from Hugging Face..."
curl -L -o mri_binary_model.keras "https://huggingface.co/datasets/DarkxCrafter/mri_model_backup/resolve/main/mri_binary_model.keras"

if [ -f "mri_binary_model.keras" ]; then
    echo "✅ MRI model downloaded successfully."
else
    echo "❌ Failed to download MRI model." >&2
    exit 1
fi
