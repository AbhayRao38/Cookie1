from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import logging
import os
import requests

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# Model download info
MODEL_URL = "https://huggingface.co/datasets/DarkxCrafter/mri_model_backup/resolve/main/mri_binary_model.keras"
MODEL_PATH = "mri_binary_model.keras"
mri_model = None

def ensure_model_loaded():
    global mri_model
    if mri_model is not None:
        return

    if not os.path.exists(MODEL_PATH):
        logging.info("Downloading MRI model from Hugging Face...")
        try:
            response = requests.get(MODEL_URL)
            response.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                f.write(response.content)
            logging.info("Model download complete.")
        except Exception as e:
            logging.error(f"Failed to download model: {e}")
            raise RuntimeError("Model download failed") from e

    try:
        mri_model = tf.keras.models.load_model(MODEL_PATH)
        logging.info("MRI model (MobileNetV2-based) loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load MRI model: {e}")
        raise RuntimeError("Model load failed") from e

def preprocess_mri_image(image):
    """
    Preprocess MRI image to match training format (128x128x3).
    """
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0

    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)

    image = np.expand_dims(image, axis=0)
    return image

@app.route('/health', methods=['GET'])
def health_check():
    try:
        ensure_model_loaded()
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

    return jsonify({
        'status': 'healthy',
        'model_loaded': mri_model is not None,
        'tensorflow_version': tf.__version__,
        'model_type': 'MobileNetV2-based',
        'input_size': '128x128x3'
    })

@app.route('/predict/mri', methods=['POST'])
def predict_mri():
    try:
        ensure_model_loaded()

        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400

        image_file = request.files['file']
        if image_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400

        image = Image.open(image_file.stream)
        image_np = np.array(image)
        processed_image = preprocess_mri_image(image_np)

        prediction = mri_model.predict(processed_image, verbose=0)
        mci_probability = float(prediction[0][0])
        probabilities = [1 - mci_probability, mci_probability]

        custom_threshold = 0.312
        predicted_class = 1 if mci_probability >= custom_threshold else 0

        confidence = abs(mci_probability - custom_threshold) / max(custom_threshold, 1 - custom_threshold)
        confidence = min(confidence + 0.5, 1.0)

        return jsonify({
            'success': True,
            'probabilities': probabilities,
            'confidence': confidence,
            'mci_probability': mci_probability,
            'predicted_class': predicted_class,
            'threshold_used': custom_threshold,
            'raw_prediction': prediction.tolist()
        })

    except Exception as e:
        logging.error(f"Error in MRI prediction: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=False)
