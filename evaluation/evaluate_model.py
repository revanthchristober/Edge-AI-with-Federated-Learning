import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import logging
import argparse
import syft as sy

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Evaluate the model on the test data
def evaluate_model(model, test_data, batch_size):
    """Evaluate the model on the test data."""
    logger.info("Evaluating model performance...")

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Evaluate the model on test data
    results = model.evaluate(test_data, batch_size=batch_size)
    logger.info(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

    return results


# Function to parse arguments for evaluation
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the trained model on test data.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model file.")
    parser.add_argument('--test_data_path', type=str, required=True, help="Path to the test data.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for evaluation. Default is 32.")
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    # Load the model
    logger.info(f"Loading model from {args.model_path}...")
    model = keras.models.load_model(args.model_path)

    # Load the test data
    logger.info(f"Loading test data from {args.test_data_path}...")
    test_data = np.load(args.test_data_path)

    # Run the evaluation
    evaluate_model(model, test_data, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
