import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
import joblib # Used to save the scaler object

# --- CONFIGURATION ---
DATA_FILE = 'cumulative_2025.10.02_14.53.51.csv' 
MODEL_FILE = 'deep_learning_model.h5'
SCALER_FILE = 'scaler.pkl'

# Features we are using for the model (must match app.py)
FEATURES = [
    'koi_period', 
    'koi_depth', 
    'koi_duration', 
    'koi_srad', 
    'koi_steff'
]
TARGET = 'koi_disposition'

# --- 1. DATA PREPARATION ---

def load_and_clean_data(file_path):
    """Loads the Kepler KOI data, selects features, cleans missing values, and prepares labels."""
    print("Loading and preparing data...")
    
    # Load the data, skipping the initial header/comment lines (assuming 53 lines based on upload info)
    # The 'comment' parameter handles the initial metadata block in the NASA CSV.
    try:
        # NOTE: Using skiprows=53 as per the file metadata
        df = pd.read_csv(file_path, comment='#', skiprows=53, usecols=[TARGET] + FEATURES)
    except Exception as e:
        print(f"Error reading CSV: {e}. Check if the file is correctly placed or if headers are consistent.")
        raise FileNotFoundError(f"Could not read data file: {file_path}")

    # Remove rows where any of the required features are missing
    df.dropna(subset=FEATURES, inplace=True)

    # Filter for the three main disposition categories: CONFIRMED, CANDIDATE, and FALSE POSITIVE
    df = df[df[TARGET].isin(['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'])]
    
    # Map the disposition labels to binary numerical labels:
    # 1: Positive (Planet Candidate or Confirmed)
    # 0: Negative (False Positive)
    df['is_exoplanet'] = df[TARGET].apply(lambda x: 1 if x in ['CONFIRMED', 'CANDIDATE'] else 0)

    X = df[FEATURES]
    y = df['is_exoplanet']
    
    print(f"Data ready. Total samples: {len(df)}")
    print(f"Positive (Planet/Candidate) samples: {y.sum()}")
    print(f"Negative (False Positive) samples: {len(y) - y.sum()}")
    
    return X, y

def scale_and_split_data(X, y):
    """Scales features using MinMaxScaler and splits data into training and testing sets."""
    # Initialize MinMaxScaler
    scaler = MinMaxScaler()
    
    # Fit the scaler on the entire feature set and transform the data
    X_scaled = scaler.fit_transform(X)
    
    # Save the fitted scaler object for future use in the Flask API
    joblib.dump(scaler, SCALER_FILE)
    print(f"Scaler saved to {SCALER_FILE}.")
    
    # Split the data (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y.values, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

# --- 2. MODEL DEFINITION AND TRAINING ---

def create_deep_learning_model(input_shape):
    """Defines an enhanced Sequential Deep Learning model with more capacity and depth."""
    print("Defining deep learning model architecture...")
    
    # Check if a GPU is available and preferred (leveraging CUDA)
    device_name = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
    print(f"Training on device: {device_name}")
    
    with tf.device(device_name):
        model = tf.keras.Sequential([
            # Input layer: 5 features, increased neurons
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            # Hidden layer 1 (Increased neurons)
            tf.keras.layers.Dense(64, activation='relu'),
            # Hidden layer 2 (Increased neurons)
            tf.keras.layers.Dense(32, activation='relu'),
            # Hidden layer 3 (New layer for increased depth)
            tf.keras.layers.Dense(16, activation='relu'),
            # Output layer: Binary classification
            tf.keras.layers.Dense(1, activation='sigmoid') 
        ])

        # Compile the model
        model.compile(
            optimizer='adam', 
            loss='binary_crossentropy', 
            metrics=['accuracy']
        )
        
    model.summary()
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    """Trains the model and evaluates performance. Increased epochs to 8000."""
    print("Starting model training (8000 epochs)...")
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=8000, # Increased epochs for better convergence
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    print("\n--- Model Evaluation ---")
    
    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Generate predictions and classification report
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['False Positive (0)', 'Planet Candidate (1)']))
    
    return model

# --- 3. EXECUTION ---

if __name__ == '__main__':
    try:
        # Load and clean data
        X, y = load_and_clean_data(DATA_FILE)

        # Scale and split data
        X_train, X_test, y_train, y_test = scale_and_split_data(X, y)

        # Create model
        model = create_deep_learning_model(X_train.shape[1])

        # Train and evaluate model
        trained_model = train_model(model, X_train, y_train, X_test, y_test)

        # Save the trained model to disk
        trained_model.save(MODEL_FILE)
        print(f"\nSuccessfully trained and saved model to {MODEL_FILE}")
        
        print("\n--- Next Step ---")
        print("Run 'python app.py' to start the API and use the model!")
        
    except FileNotFoundError:
        print(f"ERROR: Data file not found. Make sure '{DATA_FILE}' is in the same directory.")
    except Exception as e:
        print(f"An error occurred during model training: {e}")
