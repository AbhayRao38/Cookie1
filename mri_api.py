from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# Load MRI model (MobileNetV2-based from your training)
try:
    mri_model = tf.keras.models.load_model('mri_binary_model.keras')
    logging.info("MRI model (MobileNetV2-based) loaded successfully")
except Exception as e:
    logging.error(f"Failed to load MRI model: {e}")
    mri_model = None

def preprocess_mri_image(image):
    """
    Preprocess MRI image to match your training format (128x128x3)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    
    # Resize to model input size (128x128 as per your training)
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Convert grayscale to RGB (repeat channels) - your model expects RGB
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': mri_model is not None,
        'tensorflow_version': tf.__version__,
        'model_type': 'MobileNetV2-based',
        'input_size': '128x128x3'
    })

@app.route('/predict/mri', methods=['POST'])
def predict_mri():
    if mri_model is None:
        return jsonify({
            'success': False,
            'error': 'MRI model not loaded'
        }), 500
    
    try:
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
        
        # Load and preprocess image
        image = Image.open(image_file.stream)
        image_np = np.array(image)
        
        # Preprocess for MRI model
        processed_image = preprocess_mri_image(image_np)
        
        # Get prediction
        prediction = mri_model.predict(processed_image, verbose=0)
        
        # Extract probability (sigmoid output for binary classification)
        mci_probability = float(prediction[0][0])
        probabilities = [1 - mci_probability, mci_probability]  # [Non-MCI, MCI]
        
        # Apply custom threshold from your training (0.312)
        custom_threshold = 0.312
        predicted_class = 1 if mci_probability >= custom_threshold else 0
        
        # Calculate confidence based on distance from threshold
        confidence = abs(mci_probability - custom_threshold) / max(custom_threshold, 1 - custom_threshold)
        confidence = min(confidence + 0.5, 1.0)  # Ensure confidence is reasonable
        
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