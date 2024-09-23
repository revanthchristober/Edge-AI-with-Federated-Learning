import os
import tensorflow as tf
import syft as sy
from tensorflow.keras.models import save_model, load_model
import logging

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Save the model locally
def save_model_weights(model, model_dir, model_name):
    """Save model weights to a specified directory."""
    model_path = os.path.join(model_dir, f"{model_name}.h5")
    logger.info(f"Saving model weights to {model_path}")
    model.save_weights(model_path)
    logger.info(f"Model weights saved successfully.")

# Load model weights from a file
def load_model_weights(model, model_dir, model_name):
    """Load model weights from a specified directory."""
    model_path = os.path.join(model_dir, f"{model_name}.h5")
    if os.path.exists(model_path):
        logger.info(f"Loading model weights from {model_path}")
        model.load_weights(model_path)
        logger.info(f"Model weights loaded successfully.")
    else:
        logger.error(f"Model weights not found at {model_path}. Unable to load weights.")

# Federated model update
def aggregate_model_weights(models):
    """Aggregate model weights from multiple clients for federated learning."""
    logger.info("Aggregating model weights from multiple clients...")
    new_weights = []
    
    # Initialize with zeros
    for layer_weights in models[0].get_weights():
        new_weights.append(tf.zeros_like(layer_weights))
    
    # Sum weights across all models (federated averaging)
    for model in models:
        weights = model.get_weights()
        for i in range(len(weights)):
            new_weights[i] += weights[i]
    
    # Average the weights
    num_clients = len(models)
    new_weights = [weight / num_clients for weight in new_weights]
    
    logger.info("Aggregated model weights successfully.")
    return new_weights

# Update model weights from aggregated weights
def set_aggregated_weights(model, aggregated_weights):
    """Set the model weights with the aggregated weights."""
    logger.info("Setting aggregated weights to the model...")
    model.set_weights(aggregated_weights)
    logger.info("Model weights updated successfully.")

# Federated learning setup using Syft
def setup_federated_learning(workers, hook):
    """Set up federated learning with Syft."""
    logger.info("Setting up federated learning environment with Syft...")
    # Register remote workers
    worker_list = [sy.VirtualWorker(hook, id=f"worker_{i}") for i in range(workers)]
    logger.info(f"Created {workers} virtual workers for federated learning.")
    return worker_list

# Federated training function using Syft
def train_federated(model, federated_train_data, worker_list, epochs, batch_size):
    """Train the model in a federated learning setup."""
    logger.info("Starting federated training...")

    # Send the model to all workers
    model_ptrs = [model.send(worker) for worker in worker_list]

    # Train the model on each worker
    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch + 1}/{epochs}")
        for worker, model_ptr in zip(worker_list, model_ptrs):
            logger.info(f"Training on {worker.id}")
            federated_train_batch = federated_train_data[worker]
            for batch in federated_train_batch.batch(batch_size):
                model_ptr.fit(batch['x'], batch['y'], batch_size=batch_size, epochs=1, verbose=1)
    
    # Get the updated model from all workers
    updated_model_ptrs = [model_ptr.get() for model_ptr in model_ptrs]
    
    logger.info("Federated training completed.")
    return updated_model_ptrs

# Learning rate scheduler
def lr_scheduler(epoch, lr):
    """Learning rate scheduler for training."""
    if epoch < 5:
        return lr
    elif 5 <= epoch < 10:
        return lr * 0.1
    else:
        return lr * 0.01

def get_lr_scheduler_callback():
    """Returns a learning rate scheduler callback."""
    return tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

# Differential Privacy (Optional)
def apply_differential_privacy(model, noise_multiplier=1.1, l2_norm_clip=1.0):
    """Apply differential privacy to the model using TensorFlow Privacy."""
    try:
        import tensorflow_privacy as tf_privacy
        optimizer = tf_privacy.DPKerasAdamOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=1,  # Set number of microbatches for privacy
            learning_rate=0.001
        )
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        logger.info(f"Differential Privacy applied: noise_multiplier={noise_multiplier}, l2_norm_clip={l2_norm_clip}")
        return model
    except ImportError as e:
        logger.error("Failed to import TensorFlow Privacy. Please install it using 'pip install tensorflow-privacy'.")
        raise e

# Model architecture utilities
def count_trainable_parameters(model):
    """Count the number of trainable parameters in the model."""
    return int(tf.reduce_sum([tf.reduce_prod(var.shape) for var in model.trainable_variables]))

def print_model_summary(model):
    """Prints the summary of the model."""
    logger.info("Model Summary:")
    model.summary()

