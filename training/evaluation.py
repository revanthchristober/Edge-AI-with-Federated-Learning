import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import logging
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to load the model
def load_model(model_path):
    """
    Load a trained TensorFlow model from the specified path.
    
    Args:
        model_path (str): Path to the saved model.
    
    Returns:
        model (tf.keras.Model): Loaded TensorFlow model.
    """
    logger.info(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    logger.info(f"Model loaded successfully from {model_path}")
    return model

# Function to preprocess test data
def preprocess_test_data(test_data):
    """
    Preprocess test data to match the input format expected by the model.
    
    Args:
        test_data (np.array): Raw test data (e.g., images or sequences).
    
    Returns:
        processed_data (np.array): Preprocessed test data.
    """
    logger.info("Preprocessing test data...")
    
    # Assuming the data is images, normalize pixel values
    processed_data = test_data / 255.0
    
    logger.info("Test data preprocessing complete.")
    return processed_data

# Function to evaluate the model
def evaluate_model(model, x_test, y_test):
    """
    Evaluate the TensorFlow model on test data and log performance metrics.
    
    Args:
        model (tf.keras.Model): The trained model.
        x_test (np.array): Test data features.
        y_test (np.array): Test data labels.
    
    Returns:
        results (dict): Dictionary containing accuracy, precision, recall, and F1-score.
    """
    logger.info("Evaluating model on test data...")
    
    # Make predictions on test data
    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predicted_classes)
    precision = precision_score(y_test, predicted_classes, average='weighted')
    recall = recall_score(y_test, predicted_classes, average='weighted')
    f1 = f1_score(y_test, predicted_classes, average='weighted')

    logger.info("Model evaluation complete.")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")

    # Generate a detailed classification report
    class_report = classification_report(y_test, predicted_classes)
    logger.info("\nClassification Report:\n" + class_report)

    # Return a dictionary of evaluation metrics
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": class_report
    }
    
    return results

# Function to load test data
def load_test_data(test_data_path):
    """
    Load the test dataset from the specified path.
    
    Args:
        test_data_path (str): Path to the test data (assumes NumPy .npz format).
    
    Returns:
        x_test, y_test (tuple): Features and labels of the test data.
    """
    logger.info(f"Loading test data from {test_data_path}...")
    
    # Load test data from .npz file
    with np.load(test_data_path) as data:
        x_test = data['x_test']
        y_test = data['y_test']
    
    logger.info("Test data loaded successfully.")
    return x_test, y_test

# Function to save evaluation results to a file
def save_evaluation_results(results, save_path):
    """
    Save evaluation results to a text file.
    
    Args:
        results (dict): Dictionary containing the evaluation metrics.
        save_path (str): Path to save the evaluation results.
    
    Returns:
        None
    """
    logger.info(f"Saving evaluation results to {save_path}...")
    
    with open(save_path, 'w') as f:
        f.write("Model Evaluation Results:\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall: {results['recall']:.4f}\n")
        f.write(f"F1-Score: {results['f1_score']:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(results['classification_report'])
    
    logger.info(f"Evaluation results saved to {save_path}")

# Main function to evaluate the model
def main(model_path, test_data_path, save_path):
    """
    Main function to load the model, evaluate it on test data, and save the results.
    
    Args:
        model_path (str): Path to the saved model.
        test_data_path (str): Path to the test data.
        save_path (str): Path to save the evaluation results.
    
    Returns:
        None
    """
    # Step 1: Load the model
    model = load_model(model_path)

    # Step 2: Load and preprocess the test data
    x_test, y_test = load_test_data(test_data_path)
    x_test = preprocess_test_data(x_test)

    # Step 3: Evaluate the model on the test data
    results = evaluate_model(model, x_test, y_test)

    # Step 4: Save the evaluation results
    save_evaluation_results(results, save_path)

if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Evaluate a trained model on test data")
    parser.add_argument('--model-path', type=str, required=True, help="Path to the saved model")
    parser.add_argument('--test-data-path', type=str, required=True, help="Path to the test data (.npz format)")
    parser.add_argument('--save-path', type=str, required=True, help="Path to save the evaluation results")

    args = parser.parse_args()
    
    # Run the evaluation
    main(args.model_path, args.test_data_path, args.save_path)
