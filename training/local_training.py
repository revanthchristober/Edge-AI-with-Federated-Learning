import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import os
import tensorflow as tf
import syft as sy
import numpy as np
from tensorflow.keras.optimizers import Adam
from src.models.cnn_model import create_cnn_model  # Assuming you have CNN architecture
from src.utils.data_utils import load_local_data  # Assuming you have a function to load local data
from src.utils.model_utils import save_model_weights, print_model_summary
import logging
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up directories
DATA_DIR = os.path.join(os.getcwd(), 'data', 'processed')
MODEL_DIR = os.path.join(os.getcwd(), 'models', 'trained_models')

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Set up a virtual Syft worker
hook = sy.TorchHook(tf)

# Define helper functions
def local_train_worker(worker, model, train_data, val_data, epochs=10, learning_rate=0.001):
    """
    Train the model locally on the specified worker.
    
    Args:
        worker: Syft Virtual Worker
        model: TensorFlow model
        train_data: Training data for the worker
        val_data: Validation data for the worker
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
    
    Returns:
        Trained model and its weights
    """
    logger.info(f"Starting training on {worker.id}...")

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    # Send the model to the worker
    model.send(worker)
    
    # Train the model on the worker's data
    history = model.fit(train_data, validation_data=val_data, epochs=epochs, verbose=1)
    
    # Retrieve the model back from the worker
    model.get()
    
    logger.info(f"Training completed on {worker.id}.")
    
    return model, history

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Local training script for federated learning")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs for training")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for training")
    args = parser.parse_args()

    # Load local training data
    logger.info("Loading local training data...")
    train_data, val_data = load_local_data(DAT
