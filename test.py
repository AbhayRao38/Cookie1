import numpy as np

# Load the results dictionary with allow_pickle=True
results = np.load(
    r"C:\Users\praoa\OneDrive\Desktop\Projects\multimodal-emotion-mci-system\model_mri\mri_evaluation_results.npy",
    allow_pickle=True
).item()

# Extract softmax-like probabilities (1D array for class 'MCI')
probs = np.array(results['probabilities'])

print("Shape:", probs.shape)           # Should be something like (N,)
print("First 5 values:", probs[:5])    # Preview

# OPTIONAL: Convert to 2D softmax-style output: [P(class=0), P(class=1)]
probs_2d = np.stack([1 - probs, probs], axis=1)
print("Converted softmax shape:", probs_2d.shape)
print("Sample softmax rows:", probs_2d[:5])
