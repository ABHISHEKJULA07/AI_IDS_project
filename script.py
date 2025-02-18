import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import time

# Constants
DATASET_PATH = "data/CICIDS2017.csv"  # Path to your dataset
MODEL_SAVE_PATH = "saved_models/random_forest_model.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Load Dataset
def load_data(file_path):
    """
    Load the dataset from the specified file path.
    """
    print("Loading dataset...")
    data = pd.read_csv(file_path)
    print(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns.")
    return data

# Preprocess Data
def preprocess_data(data):
    """
    Preprocess the dataset: handle missing values, encode labels, and scale features.
    """
    print("Preprocessing data...")

    # Drop missing values
    data = data.dropna()

    # Encode labels (e.g., 'BENIGN' -> 0, 'DDoS' -> 1, etc.)
    label_encoder = LabelEncoder()
    data['Label'] = label_encoder.fit_transform(data['Label'])

    # Separate features and labels
    X = data.drop(columns=['Label'])
    y = data['Label']

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Save the scaler for later use
    joblib.dump(scaler, "saved_models/scaler.pkl")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    print("Data preprocessing completed.")
    return X_train, X_test, y_train, y_test

# Train Model
def train_model(X_train, y_train):
    """
    Train a Random Forest Classifier on the preprocessed data.
    """
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    print("Model training completed.")
    return model

# Evaluate Model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test dataset.
    """
    print("Evaluating model...")
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

# Save Model
def save_model(model, file_path):
    """
    Save the trained model to a file.
    """
    print("Saving model...")
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}.")

# Real-Time Detection (Simulated)
def real_time_detection(model, scaler):
    """
    Simulate real-time network traffic detection using the trained model.
    """
    print("Starting real-time detection...")
    try:
        while True:
            # Simulate incoming network traffic (replace with actual data)
            simulated_traffic = np.random.rand(1, 78)  # Adjust based on your dataset features
            scaled_traffic = scaler.transform(simulated_traffic)

            # Predict using the trained model
            prediction = model.predict(scaled_traffic)
            print(f"Predicted Label: {prediction[0]}")

            # Sleep for a short interval (simulate real-time delay)
            time.sleep(2)
    except KeyboardInterrupt:
        print("Real-time detection stopped.")

# Main Function
def main():
    # Step 1: Load Data
    data = load_data(DATASET_PATH)

    # Step 2: Preprocess Data
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Step 3: Train Model
    model = train_model(X_train, y_train)

    # Step 4: Evaluate Model
    evaluate_model(model, X_test, y_test)

    # Step 5: Save Model
    save_model(model, MODEL_SAVE_PATH)

    # Step 6: Real-Time Detection (Simulated)
    scaler = joblib.load("saved_models/scaler.pkl")
    real_time_detection(model, scaler)

if __name__ == "__main__":
    main()