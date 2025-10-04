
Prerequisites
You must have Python 3.11 installed.
1. Clone the Repositorygit clone [YOUR_REPOSITORY_URL]
cd [YOUR_PROJECT_FOLDER_NAME]
2. Create and Activate a Virtual Environment
3. It's highly recommended to isolate the project dependencies.
 # Create the environment
python3 -m venv venv
# Activate the environment (macOS/Linux)
source venv/bin/activate
# Activate the environment (Windows)
.\venv\Scripts\activate

3. Install Dependencies
Install all required Python libraries using the requirements.txt file:pip install -r requirements.txt
Training and DeploymentThis project requires two steps to run the web app: first, train the model to generate the necessary files; second, start the Flask API.Step 1: Train the Deep Learning Mode
l The training script processes the raw data, scales the features, trains the Deep Learning model, and saves the resulting files that the API needs.This step creates:deep_learning_model.h5 scaler.pkl Run the training script from your terminal:python train_exofinder_model.py
(Note: Ensure you have the required cumulative_2025.10.02_14.53.51.csv data file in the root directory for this script to run successfully.)Step 2: Run the Flask Web ServerOnce the model files are present, start the Flask backend:python app.py
You will see output indicating the server is running. Open your web browser and navigate to the address provided (usually: http://127.0.0.1:5000)or(http://http://192.168.1.37:5000). Project StructureFile/FolderDescriptionapp.pyThe main Flask application that loads the trained model and scaler, defines the /predict API route, and serves the index.html.train_exofinder_model.pyScript used to clean data, train the Deep Learning Model, evaluate performance, and save the model (.h5) and scaler (.pkl).index.html The single-page frontend application with the user interface (HTML/Tailwind CSS/JS) for input and result display.deep_learning_model.h5The serialized TensorFlow/Keras model used for real-time classification.scaler.pklThe serialized MinMaxScaler object, necessary to correctly preprocess new user inputs before prediction.requirements.txtLists all necessary Python dependencies (Flask, TensorFlow, NumPy, Joblib).exo_finder_proposal.mdDetailed project proposal outlining the problem, methodology, and value proposition.
