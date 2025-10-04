import os
import json
import logging
import numpy as np
import tensorflow as tf
from flask import Flask, send_from_directory, request, jsonify
from joblib import load # Used for loading the scaler

# Configure logging
# We'll use the basic configuration, which should output to the console 
# where you run the Flask server.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)

# Define the root directory of the application for robust file serving
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Global Model and Scaler Variables ---
MODEL = None
SCALER = None
GLOBAL_LOAD_ERROR = None # Flag to hold fatal loading errors

# --- File Paths ---
MODEL_PATH = os.path.join(APP_ROOT, 'deep_learning_model.h5')
SCALER_PATH = os.path.join(APP_ROOT, 'scaler.pkl')

def load_resources():
    """
    Loads the Keras model and the MinMaxScaler.
    THIS FUNCTION REQUIRES deep_learning_model.h5 AND scaler.pkl to be present.
    """
    global MODEL, SCALER, GLOBAL_LOAD_ERROR
    
    # Reset model variables
    MODEL = None
    SCALER = None
    GLOBAL_LOAD_ERROR = None
    
    try:
        # 1. Load the Keras Model
        logging.info(f"Attempting to load Keras model from: {MODEL_PATH}")
        # Use suppress_warnings=True to keep the console clean during load
        MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
        logging.info("Keras model loaded successfully.")
        
        # 2. Load the Scaler
        logging.info(f"Attempting to load scaler from: {SCALER_PATH}")
        SCALER = load(SCALER_PATH)
        logging.info("Scaler loaded successfully.")
        
    except FileNotFoundError as e:
        filename = os.path.basename(e.filename)
        error_msg = f"FATAL: Missing resource file: {filename}. Prediction service will be unavailable."
        logging.error(error_msg)
        GLOBAL_LOAD_ERROR = error_msg
    except Exception as e:
        error_msg = f"FATAL: Failed to load AI resources: {type(e).__name__}: {str(e)}"
        logging.error(error_msg, exc_info=True)
        GLOBAL_LOAD_ERROR = error_msg


# Load resources when the Flask app starts
with app.app_context():
    load_resources()


# --- API Endpoint for Prediction ---

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives Kepler object data (5 features), preprocesses it, 
    and returns the prediction probabilities from the real deep learning model.
    """
    # 1. Check for fatal resource loading errors first
    if GLOBAL_LOAD_ERROR is not None or MODEL is None or SCALER is None:
        error_msg = GLOBAL_LOAD_ERROR if GLOBAL_LOAD_ERROR else "Model or Scaler not loaded. Prediction service unavailable (Status 503)."
        logging.error(error_msg)
        return jsonify({'error': error_msg}), 503 

    try:
        # 2. Get JSON data from request
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'error': 'Missing "features" key in request body or request is empty.'}), 400

        features = data['features']
        
        # 3. Validate and convert features
        if not isinstance(features, list) or len(features) != 5:
            return jsonify({'error': f'Expected 5 features, but received {len(features)}.'}), 400

        # Convert all features to float
        try:
            input_features = np.array([float(f) for f in features]).reshape(1, -1)
        except ValueError:
            return jsonify({'error': 'All features must be valid numerical values.'}), 400
        
        logging.info(f"Input features received: {input_features.tolist()}")

        # 4. Preprocess (Scale) the input using the loaded SCALER
        scaled_features = SCALER.transform(input_features)
        
        # 5. Make REAL Prediction using the loaded MODEL
        # Predict returns an array of probabilities, e.g., [[0.05, 0.95]]
        prediction = MODEL.predict(scaled_features)

        # 6. Format and Return Result
        # prediction[0][1] is the probability of index 1 (Exoplanet Candidate)
        probability_candidate = float(prediction[0][0])
        probability_false_positive = 1.0 - probability_candidate
        
        probability = [probability_false_positive, probability_candidate]

        logging.info(f"REAL Prediction: {probability}")

        return jsonify({
            'success': True,
            'probability': probability, 
            'dispositions': ['False Positive', 'Exoplanet Candidate'] # For reference
        })

    except Exception as e:
        # Catch any unexpected errors during processing
        error_msg = f"An internal server error occurred during prediction: {type(e).__name__}: {str(e)}"
        logging.error(error_msg, exc_info=True)
        return jsonify({'error': error_msg}), 500


# --- Serving Static Content (index.html and other files) ---

# Route for the root URL ('/')
@app.route('/')
def serve_index():
    """
    Renders the index.html file located in the same directory.
    """
    try:
        app.logger.info("Serving index.html")
        return send_from_directory(APP_ROOT, 'index.html')
    except FileNotFoundError:
        app.logger.error(f"FATAL ERROR: index.html not found at path: {APP_ROOT}")
        return "Error: index.html not found.", 404

# Route for serving static assets
@app.route('/<path:filename>')
def serve_static(filename):
    """
    Serves other files (like the scaler.pkl or model.h5) 
    """
    app.logger.info(f"Serving static file requested: {filename}")
    return send_from_directory(APP_ROOT, filename)


# --- Run the application ---
if __name__ == '__main__':
    # Setting host='0.0.0.0' makes it accessible externally, which is often 
    # required in containerized environments like the Canvas.
    # Added threaded=False to potentially improve log visibility in some environments.
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=False)
