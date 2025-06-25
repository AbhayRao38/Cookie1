import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    average_precision_score, f1_score
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')
import logging

# Logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ✅ Load the trained model
logger.info("📥 Loading trained model...")
model = tf.keras.models.load_model('mri_binary_model.keras')
logger.info("✅ Model loaded successfully!")

# ✅ Load test data
logger.info("\n📥 Loading test data...")
X = np.load('X_mri_balanced.npy')
y_raw = np.load('y_mri_balanced.npy')
y_binary = np.array([0 if label in [0, 1] else 1 for label in y_raw])
X_test_idx = np.load('X_mri_test_idx.npy')
y_test = np.load('y_mri_test.npy')

logger.info(f"✅ Test set: {len(X_test_idx)} samples")
logger.info(f"   - Non-MCI (0): {np.sum(y_test == 0)} samples")
logger.info(f"   - MCI (1): {np.sum(y_test == 1)} samples")

# ✅ Dataset pipeline
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

def create_generator(X, y, indices):
    def gen():
        for i in indices:
            x = X[i].astype(np.float32)
            if x.ndim == 3:
                x = x[:, :, 0]
            x = np.expand_dims(x, axis=-1)
            x = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x))
            yield x, y[i]
    return gen

def get_dataset(X, y, indices):
    ds = tf.data.Dataset.from_generator(
        create_generator(X, y, indices),
        output_signature=(
            tf.TensorSpec(shape=(128, 128, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int64)
        )
    ).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

test_ds = get_dataset(X, y_binary, X_test_idx)

# ✅ Model Evaluation
logger.info("\n🔍 Evaluating model on test set...")
test_loss, test_accuracy, test_auc = model.evaluate(test_ds, verbose=0)
logger.info(f"📊 Test Loss: {test_loss:.4f}")
logger.info(f"✅ Test Accuracy: {test_accuracy:.4f}")
logger.info(f"⭐ AUC: {test_auc:.4f}")

# ✅ Predictions
logger.info("\n🔮 Generating predictions...")
y_pred_proba = model.predict(test_ds, verbose=0).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

# ✅ Class-wise Accuracy + Variance
logger.info("\n📈 Per-Class Accuracy:")
class_accuracies = []
for class_label in [0, 1]:
    class_name = "Non-MCI" if class_label == 0 else "MCI"
    correct = np.sum((y_test == class_label) & (y_pred == class_label))
    total = np.sum(y_test == class_label)
    acc = correct / total if total > 0 else 0
    class_accuracies.append(acc)
    logger.info(f"   - {class_name}: {acc * 100:.2f}%")

acc_variance = np.var(class_accuracies)
acc_std = np.std(class_accuracies)
logger.info(f"📈 Class-wise Accuracy Variance: {acc_variance:.6f}")
logger.info(f"📉 Class-wise Accuracy Std Dev: {acc_std:.6f}")

# ✅ Classification Report and Confusion Matrix
logger.info("\n🧪 Classification Report:\n" + classification_report(y_test, y_pred, target_names=['Non-MCI', 'MCI']))
cm = confusion_matrix(y_test, y_pred)
logger.info(f"📊 Confusion Matrix:\n{cm}")

# ✅ Extra Metrics
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
f1 = f1_score(y_test, y_pred)

# ✅ ROC and PR curves
fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

# ✅ Optimal Threshold
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = roc_thresholds[optimal_idx]
optimal_sensitivity = tpr[optimal_idx]
optimal_specificity = 1 - fpr[optimal_idx]
y_pred_optimal = (y_pred_proba > optimal_threshold).astype(int)
f1_optimal = f1_score(y_test, y_pred_optimal)

# ✅ Calibration
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred_proba, n_bins=10)
calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))

# ✅ Plotting Results
plt.figure(figsize=(15, 12))

# ROC
plt.subplot(2, 3, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100,
            label=f'Optimal (threshold={optimal_threshold:.3f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)

# PR
plt.subplot(2, 3, 2)
plt.plot(recall_vals, precision_vals, color='blue', lw=2,
         label=f'PR curve (AP = {avg_precision:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True, alpha=0.3)

# Confusion Matrix
plt.subplot(2, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-MCI', 'MCI'],
            yticklabels=['Non-MCI', 'MCI'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Prediction Distribution
plt.subplot(2, 3, 4)
plt.hist(y_pred_proba[y_test == 0], alpha=0.7, label='Non-MCI', bins=30, color='blue')
plt.hist(y_pred_proba[y_test == 1], alpha=0.7, label='MCI', bins=30, color='red')
plt.axvline(0.5, color='black', linestyle='--', label='Default threshold')
plt.axvline(optimal_threshold, color='green', linestyle='--', label='Optimal threshold')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Prediction Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

# Calibration Plot
plt.subplot(2, 3, 5)
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Plot')
plt.legend()
plt.grid(True, alpha=0.3)

# Threshold Analysis
plt.subplot(2, 3, 6)
min_len = min(len(roc_thresholds), len(tpr), len(fpr), len(j_scores))
plt.plot(roc_thresholds[:min_len], tpr[:min_len], label='Sensitivity', color='red')
plt.plot(roc_thresholds[:min_len], 1 - fpr[:min_len], label='Specificity', color='blue')
plt.plot(roc_thresholds[:min_len], j_scores[:min_len], label='Youden J', color='green')
plt.axvline(optimal_threshold, color='black', linestyle='--', label='Optimal')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Threshold Analysis')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mri_model_evaluation.png', dpi=300, bbox_inches='tight')
plt.show()

# ✅ Save Detailed Results
results_dict = {
    'test_loss': test_loss,
    'test_accuracy': test_accuracy,
    'test_auc': test_auc,
    'roc_auc': roc_auc,
    'average_precision': avg_precision,
    'f1_score': f1,
    'sensitivity': sensitivity,
    'specificity': specificity,
    'precision': precision,
    'optimal_threshold': optimal_threshold,
    'f1_score_optimal': f1_optimal,
    'calibration_error': calibration_error,
    'class_accuracies': class_accuracies,
    'accuracy_variance': acc_variance,
    'accuracy_std_dev': acc_std,
    'confusion_matrix': cm.tolist(),
    'predictions': y_pred.tolist(),
    'probabilities': y_pred_proba.tolist(),
    'true_labels': y_test.tolist()
}

np.save('mri_evaluation_results.npy', results_dict)
logger.info("💾 Results saved to 'mri_evaluation_results.npy'")
logger.info("✅ Evaluation Complete!")
