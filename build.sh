#!/usr/bin/env bash

# Install git-lfs in Render-compatible way
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install -y git-lfs

# Enable and pull LFS files
git lfs install
git lfs pull

# Install Python dependencies
pip install -r requirements.txt
