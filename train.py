import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ✅ Load dataset (eager mode)
X = np.load('X_mri_balanced.npy')  # no mmap_mode
y_raw = np.load('y_mri_balanced.npy')

# ✅ Convert to binary labels: 0,1 → 0 (Non-MCI), 2,3 → 1 (MCI)
y_binary = np.array([0 if label in [0, 1] else 1 for label in y_raw])
print(f"✅ Loaded {X.shape[0]} samples | MCI ratio: {np.mean(y_binary):.2f}")

# ✅ Stratified re-split after loading
indices = np.arange(len(X))
X_train_idx, X_temp_idx, y_train, y_temp = train_test_split(indices, y_binary, test_size=0.3, stratify=y_binary, random_state=42)
X_val_idx, X_test_idx, y_val, y_test = train_test_split(X_temp_idx, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# ✅ Save test indices
np.save('X_mri_test_idx.npy', X_test_idx)
np.save('y_mri_test.npy', y_test)

# ✅ Class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_binary), y=y_binary)
class_weight_dict = dict(enumerate(class_weights))
print(f"✅ Class Weights: {class_weight_dict}")

# ✅ Augmentation
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.04),
    layers.RandomZoom(0.05),
    layers.RandomContrast(0.05),
])

def create_generator(X, y, indices):
    def gen():
        for i in indices:
            x = X[i].astype(np.float32)
            if x.ndim == 3:
                x = x[:, :, 0]  # remove extra channels
            x = np.expand_dims(x, axis=-1)
            x = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x))  # shape: (128, 128, 3)
            yield x, y[i]
    return gen

def get_dataset(X, y, indices, augment_enabled=False):
    ds = tf.data.Dataset.from_generator(
        create_generator(X, y, indices),
        output_signature=(
            tf.TensorSpec(shape=(128, 128, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int64)
        )
    ).batch(BATCH_SIZE)

    if augment_enabled:
        ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)

    return ds.prefetch(AUTOTUNE)

train_ds = get_dataset(X, y_binary, X_train_idx, augment_enabled=True)
val_ds   = get_dataset(X, y_binary, X_val_idx)
test_ds  = get_dataset(X, y_binary, X_test_idx)

# ✅ Model
base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc')
    ]
)

model.summary()

# ✅ Callbacks
callbacks = [
    EarlyStopping(patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
    ModelCheckpoint('mri_binary_model.keras', save_best_only=True)
]

# ✅ Training
print("\n🚀 Starting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=2
)

# ✅ Evaluation
print("\n🔍 Evaluation on Test Set:")
loss, acc, auc = model.evaluate(test_ds, verbose=2)
print(f"📊 Test Loss: {loss:.4f}")
print(f"✅ Test Accuracy: {acc:.4f}")
print(f"⭐ AUC: {auc:.4f}")

# ✅ Custom Threshold-Based Evaluation (Threshold = 0.31)
print("\n🎯 Custom Threshold Evaluation (Threshold = 0.31):")

# Step 1: Get raw test inputs and true labels
X_test = np.array([X[i] for i in X_test_idx])
y_test_true = np.array([y_binary[i] for i in X_test_idx])

# Step 2: Preprocess test inputs (grayscale → RGB)
X_test_rgb = np.repeat(X_test, 3, axis=-1)  # shape: (N, 128, 128, 3)

# Step 3: Predict probabilities
y_probs = model.predict(X_test_rgb, batch_size=32, verbose=0).flatten()

# Step 4: Apply custom threshold
custom_threshold = 0.312
y_pred_custom = (y_probs >= custom_threshold).astype(int)

# Step 5: Show classification metrics
print(f"ROC AUC: {roc_auc_score(y_test_true, y_probs):.4f}")
print("Classification Report:")
print(classification_report(y_test_true, y_pred_custom, target_names=["Non-MCI", "MCI"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test_true, y_pred_custom))
