import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import random

# Paths
data_dir = "data/Data"
img_size = (128, 128)
max_per_class = 700  # ✏️ Adjust if needed (not using full dataset deliberately)

# Label mapping (OASIS)
label_map = {
    "Non Demented": 0,
    "Very mild Dementia": 1,
    "Mild Dementia": 2,
    "Moderate Dementia": 3
}

# Storage
class_images = defaultdict(list)
seen_hashes = set()

def hash_image(img_array):
    return hash(img_array.tobytes())

# Load and filter images
for label_name, label in label_map.items():
    folder_path = os.path.join(data_dir, label_name)
    if not os.path.isdir(folder_path):
        print(f"⚠️ Warning: Folder not found -> {folder_path}")
        continue

    all_files = os.listdir(folder_path)
    random.shuffle(all_files)  # For variability

    for img_name in tqdm(all_files, desc=f"Processing {label_name}"):
        if len(class_images[label]) >= max_per_class:
            break

        try:
            img_path = os.path.join(folder_path, img_name)
            img = Image.open(img_path).convert("L")
            img = img.resize(img_size)
            img_array = np.asarray(img, dtype=np.float32) / 255.0

            h = hash_image(img_array)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            class_images[label].append(img_array)
        except Exception as e:
            print(f"❌ Error loading {img_name}: {e}")

# Assemble final dataset
X = []
y = []

for label, images in class_images.items():
    for img_array in images:
        X.append(img_array[..., np.newaxis])  # Add channel
        y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y)

# Shuffle
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Save
np.save("X_mri_balanced.npy", X)
np.save("y_mri_balanced.npy", y)

print(f"\n✅ Final dataset saved!")
print(f"   X shape: {X.shape}")
print(f"   y shape: {y.shape}")
print(f"   Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
