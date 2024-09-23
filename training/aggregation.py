import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import logging
import numpy as np
import tensorflow as tf

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Aggregation Function: Federated Averaging (FedAvg)
def federated_averaging(worker_weights):
    """
    Perform Federated Averaging (FedAvg) on the collected model weights from multiple workers.
    
    Args:
        worker_weights (list): List of weights from each worker (list of NumPy arrays)
    
    Returns:
        averaged_weights: Federated averaged weights
    """
    logger.info("Starting Federated Averaging (FedAvg) across workers...")
    
    # Initialize averaging using the first worker's weights as a baseline
    averaged_weights = [np.zeros_like(weights) for weights in worker_weights[0]]
    
    # Iterate through each layer and sum the weights from all workers
    for layer_index in range(len(averaged_weights)):
        for worker_index in range(len(worker_weights)):
            averaged_weights[layer_index] += worker_weights[worker_index][layer_index]
        
        # Average the summed weights by dividing by the number of workers
        averaged_weights[layer_index] = averaged_weights[layer_index] / len(worker_weights)
    
    logger.info("FedAvg complete. Averaged weights computed.")
    return averaged_weights

# Utility to convert TensorFlow model weights to NumPy arrays
def get_model_weights_as_numpy(model):
    """
    Retrieve TensorFlow model weights as NumPy arrays for aggregation.
    
    Args:
        model (tf.keras.Model): TensorFlow model
    
    Returns:
        List of NumPy arrays representing model weights
    """
    return [layer_weights.numpy() for layer_weights in model.get_weights()]

# Utility to set model weights from NumPy arrays
def set_model_weights_from_numpy(model, weights):
    """
    Set the TensorFlow model's weights from a list of NumPy arrays.
    
    Args:
        model (tf.keras.Model): TensorFlow model
        weights (list): List of NumPy arrays representing model weights
    
    Returns:
        None
    """
    model.set_weights(weights)

# Example usage: Aggregating weights from multiple models
def aggregate_worker_models(models):
    """
    Aggregates models using FedAvg by collecting weights and applying federated averaging.
    
    Args:
        models (list): List of TensorFlow models (one for each worker)
    
    Returns:
        global_model: TensorFlow model after applying FedAvg
    """
    logger.info("Collecting worker model weights for aggregation...")

    # Get the weights from each worker's model in NumPy array format
    worker_weights = [get_model_weights_as_numpy(model) for model in models]

    # Perform federated averaging on collected weights
    averaged_weights = federated_averaging(worker_weights)
    
    # Use the first model as the global model template
    global_model = tf.keras.models.clone_model(models[0])
    
    # Set the averaged weights to the global model
    set_model_weights_from_numpy(global_model, averaged_weights)
    
    logger.info("Global model aggregated using FedAvg.")
    return global_model

# Example function to load and save model weights
def save_aggregated_model(model, save_path):
    """
    Save the aggregated model after FedAvg to a specified path.
    
    Args:
        model (tf.keras.Model): Aggregated global model
        save_path (str): Path to save the model
    
    Returns:
        None
    """
    logger.info(f"Saving aggregated model to {save_path}...")
    model.save(save_path)
    logger.info("Model saved successfully.")

def load_worker_model(model_path):
    """
    Load a TensorFlow model for a worker.
    
    Args:
        model_path (str): Path to the worker's saved model.
    
    Returns:
        model: Loaded TensorFlow model
    """
    logger.info(f"Loading model from {model_path}...")
    return tf.keras.models.load_model(model_path)

# Example aggregation scenario
def example_fedavg_scenario():
    """
    Example scenario: simulate FedAvg by aggregating weights from multiple workers.
    """
    logger.info("Simulating FedAvg scenario...")

    # Example paths where the worker models are stored
    worker_model_paths = [
        './models/worker_model_1.h5',
        './models/worker_model_2.h5',
        './models/worker_model_3.h5'
    ]

    # Load models from worker model paths
    worker_models = [load_worker_model(path) for path in worker_model_paths]

    # Aggregate the worker models using FedAvg
    global_model = aggregate_worker_models(worker_models)

    # Save the aggregated global model
    save_aggregated_model(global_model, './models/global_fedavg_model.h5')

if __name__ == "__main__":
    # Run an example federated averaging scenario
    example_fedavg_scenario()
