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

# --- Custom Fix: Rename model file for TensorFlow --- #
# Render blocks uploading .keras files, so we renamed it to .keras.txt
# Now we rename it back during build so TF can load it

if [ -f "mri_binary_model.keras.txt" ]; then
    mv mri_binary_model.keras.txt mri_binary_model.keras
    echo "✅ Renamed MRI model to mri_binary_model.keras"
fi

# You can add similar renaming for other models if needed:
# Example:
# if [ -f "emotion_model_cpu.pth.txt" ]; then
#     mv emotion_model_cpu.pth.txt emotion_model_cpu.pth
#     echo "✅ Renamed eye model"
# fi
