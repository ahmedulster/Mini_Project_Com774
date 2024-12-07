from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import logging
from datetime import datetime
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Model paths
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "random_forest.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

def load_models():
    """Load all required models and preprocessors."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        logger.info("Models loaded successfully")
        return model, scaler, label_encoder
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

# Load models at startup
try:
    model, scaler, label_encoder = load_models()
except Exception as e:
    logger.error(f"Failed to load models at startup: {str(e)}")
    raise

def validate_features(features):
    """Validate and preprocess input features."""
    EXPECTED_FEATURES = 562
    
    if len(features) < EXPECTED_FEATURES:
        logger.warning(f"Padding {EXPECTED_FEATURES - len(features)} missing features with zeros")
        features = features + [0] * (EXPECTED_FEATURES - len(features))
    elif len(features) > EXPECTED_FEATURES:
        logger.warning(f"Truncating {len(features) - EXPECTED_FEATURES} extra features")
        features = features[:EXPECTED_FEATURES]
    
    return features

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint to check API health."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint."""
    try:
        # Validate request format
        if not request.is_json:
            logger.error("Request content-type is not JSON")
            return jsonify({'error': 'Request must be JSON'}), 415

        data = request.get_json()
        
        # Validate request content
        if 'features' not in data:
            logger.error("Missing 'features' key in request")
            return jsonify({'error': '"features" key is required'}), 400
            
        if not isinstance(data['features'], list):
            logger.error("'features' must be a list")
            return jsonify({'error': '"features" must be a list'}), 400

        # Process features
        features = validate_features(data['features'])
        features_array = np.array(features).reshape(1, -1)
        
        # Generate prediction
        features_scaled = scaler.transform(features_array)
        prediction = model.predict(features_scaled)
        predicted_label = label_encoder.inverse_transform(prediction)
        
        # Log successful prediction
        logger.info(f"Successful prediction: {predicted_label[0]}")
        
        return jsonify({
            'status': 'success',
            'prediction': predicted_label[0],
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': f"Internal server error: {str(e)}",
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/models/metadata', methods=['GET'])
def model_metadata():
    """Endpoint to get model metadata."""
    return jsonify({
        'model_path': str(MODEL_PATH),
        'scaler_path': str(SCALER_PATH),
        'label_encoder_path': str(LABEL_ENCODER_PATH),
        'expected_features': 562,
        'possible_activities': list(label_encoder.classes_)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)