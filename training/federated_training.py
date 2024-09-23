import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import syft as sy
import tensorflow as tf
import numpy as np
import logging
import argparse
from tensorflow.keras.optimizers import Adam
from src.models.cnn_model import create_cnn_model
from src.utils.data_loader import load_federated_data
from src.utils.model_utils import average_weights, save_model_weights, print_model_summary

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up Syft Hook for TensorFlow
hook = sy.TorchHook(tf)

# Define Federated Aggregation Function (FedAvg)
def federated_averaging(models):
    """Perform federated averaging (FedAvg) on the received models' weights."""
    logger.info("Performing federated averaging...")
    averaged_weights = average_weights([model.get_weights() for model in models])
    return averaged_weights

# Define helper function to train on each worker
def train_on_worker(worker, model, train_data, epochs, learning_rate):
    """
    Train the model on the specified Syft worker.
    
    Args:
        worker: Syft Virtual Worker
        model: TensorFlow model to train
        train_data: Training data for the worker
        epochs: Number of epochs to train
        learning_rate: Learning rate for training
    
    Returns:
        Trained model with updated weights
    """
    logger.info(f"Training model on {worker.id}...")

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Send the model to the worker
    model.send(worker)

    # Train the model on worker's local data
    model.fit(train_data, epochs=epochs, verbose=1)

    # Retrieve the model from the worker
    model.get()

    return model

# Define the Federated Training Setup
def federated_training(workers, model, federated_train_data, rounds, epochs_per_round, learning_rate):
    """
    Perform federated learning over multiple rounds.
    
    Args:
        workers: List of Syft workers
        model: TensorFlow model to be trained federated
        federated_train_data: Federated data distributed across workers
        rounds: Number of federated learning rounds
        epochs_per_round: Epochs to train per round
        learning_rate: Learning rate for training
    
    Returns:
        Final global model after federated learning
    """
    logger.info("Starting Federated Learning...")
    
    for round_num in range(1, rounds + 1):
        logger.info(f"---- Federated Learning Round {round_num}/{rounds} ----")

        worker_models = []

        # Train each worker's model
        for worker, train_data in zip(workers, federated_train_data):
            worker_model = tf.keras.models.clone_model(model)
            worker_model.set_weights(model.get_weights())
            
            # Train model on worker
            trained_worker_model = train_on_worker(worker, worker_model, train_data, epochs=epochs_per_round, learning_rate=learning_rate)
            
            # Collect the trained model from the worker
            worker_models.append(trained_worker_model)

        # Perform Federated Averaging
        averaged_weights = federated_averaging(worker_models)
        model.set_weights(averaged_weights)

        logger.info(f"Global model updated after round {round_num}")

    logger.info("Federated Learning completed.")
    return model

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Federated Learning Script")
    parser.add_argument('--rounds', type=int, default=5, help="Number of federated learning rounds")
    parser.add_argument('--epochs_per_round', type=int, default=1, help="Number of epochs per federated round")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for federated training")
    args = parser.parse_args()

    # Load federated data
    logger.info("Loading federated training data...")
    federated_train_data, workers = load_federated_data()  # You should have this function to load the data

    # Initialize the model
    logger.info("Initializing CNN model...")
    global_model = create_cnn_model()  # Load CNN architecture
    print_model_summary(global_model)

    # Perform federated learning
    logger.info("Starting federated learning...")
    final_model = federated_training(workers, global_model, federated_train_data, 
                                     rounds=args.rounds, epochs_per_round=args.epochs_per_round,
                                     learning_rate=args.learning_rate)

    # Save the global model after federated learning
    logger.info("Saving the global model after federated learning...")
    save_model_weights(final_model, './models/trained_models', 'cnn_federated_model')

if __name__ == "__main__":
    main()
