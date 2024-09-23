import logging
import os
import json
from datetime import datetime

# Define log formats
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Directory to store logs
LOG_DIR = os.path.join(os.getcwd(), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Function to set up a logger
def setup_logger(name, log_file, level=logging.INFO):
    """
    Sets up a structured logger with the specified name and log file.
    
    Args:
    - name (str): Name of the logger.
    - log_file (str): File path to store the log.
    - level: Logging level (default: logging.INFO).
    
    Returns:
    - logger: Configured logger object.
    """
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    # Stream logs to console as well
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# Global logger instance
main_logger = setup_logger('main', os.path.join(LOG_DIR, 'federated_training.log'))

# Function to log model metrics (structured as JSON)
def log_metrics(logger, client_id, epoch, metrics):
    """
    Logs model metrics in a structured format (JSON).
    
    Args:
    - logger: Logger instance to use for logging.
    - client_id (int): ID of the federated learning client.
    - epoch (int): Current epoch number.
    - metrics (dict): Dictionary containing metric names and their values (e.g., accuracy, loss).
    
    Example:
    metrics = {
        'accuracy': 0.95,
        'loss': 0.05
    }
    """
    log_data = {
        "client_id": client_id,
        "epoch": epoch,
        "metrics": metrics,
        "timestamp": datetime.utcnow().strftime(DATE_FORMAT)
    }
    
    # Convert to JSON string for structured logging
    log_str = json.dumps(log_data)
    logger.info(f"Metrics: {log_str}")

# Log model training start
def log_training_start(logger, client_id, model_name, total_epochs):
    """
    Logs the start of the training process for a client.
    
    Args:
    - logger: Logger instance to use for logging.
    - client_id (int): ID of the federated learning client.
    - model_name (str): Name of the model being trained.
    - total_epochs (int): Number of epochs for training.
    """
    logger.info(f"Training started for Client {client_id} using model {model_name}. Total epochs: {total_epochs}")

# Log model training end
def log_training_end(logger, client_id, model_name, total_epochs, final_metrics):
    """
    Logs the end of the training process for a client.
    
    Args:
    - logger: Logger instance to use for logging.
    - client_id (int): ID of the federated learning client.
    - model_name (str): Name of the model that was trained.
    - total_epochs (int): Total number of epochs.
    - final_metrics (dict): Final evaluation metrics of the model (e.g., accuracy, loss).
    """
    logger.info(f"Training completed for Client {client_id} using model {model_name}.")
    logger.info(f"Final metrics after {total_epochs} epochs: {final_metrics}")

# Function to log errors
def log_error(logger, client_id, error_msg):
    """
    Logs an error message with the associated client ID.
    
    Args:
    - logger: Logger instance to use for logging.
    - client_id (int): ID of the federated learning client where the error occurred.
    - error_msg (str): Description of the error.
    """
    logger.error(f"Error in Client {client_id}: {error_msg}")

# Function to log model saving
def log_model_saving(logger, client_id, model_path):
    """
    Logs the event of saving the model.
    
    Args:
    - logger: Logger instance to use for logging.
    - client_id (int): ID of the federated learning client.
    - model_path (str): Path where the model is saved.
    """
    logger.info(f"Model for Client {client_id} saved at {model_path}")

# Function to log aggregation process (for central server)
def log_aggregation_start(logger, round_num, num_clients):
    """
    Logs the start of the aggregation process on the central server.
    
    Args:
    - logger: Logger instance to use for logging.
    - round_num (int): Current round of aggregation.
    - num_clients (int): Number of clients participating in the round.
    """
    logger.info(f"Aggregation round {round_num} started with {num_clients} clients.")

# Function to log aggregated model result
def log_aggregation_result(logger, round_num, aggregated_metrics):
    """
    Logs the aggregated model metrics after the aggregation process.
    
    Args:
    - logger: Logger instance to use for logging.
    - round_num (int): Current round of aggregation.
    - aggregated_metrics (dict): Dictionary containing aggregated metrics (e.g., accuracy, loss).
    """
    log_data = {
        "round_num": round_num,
        "aggregated_metrics": aggregated_metrics,
        "timestamp": datetime.utcnow().strftime(DATE_FORMAT)
    }
    
    log_str = json.dumps(log_data)
    logger.info(f"Aggregated metrics for round {round_num}: {log_str}")

# Example of usage
if __name__ == "__main__":
    # Log a training start event for client 1
    log_training_start(main_logger, client_id=1, model_name='CNN', total_epochs=10)

    # Log metrics for a client after epoch 1
    metrics = {
        'accuracy': 0.88,
        'loss': 0.12
    }
    log_metrics(main_logger, client_id=1, epoch=1, metrics=metrics)

    # Log an error event for client 1
    log_error(main_logger, client_id=1, error_msg="Failed to load dataset.")

    # Log model saving for client 1
    log_model_saving(main_logger, client_id=1, model_path='/models/client1_cnn.h5')

    # Log aggregation start on server
    log_aggregation_start(main_logger, round_num=1, num_clients=5)

    # Log aggregated metrics after aggregation
    aggregated_metrics = {
        'accuracy': 0.91,
        'loss': 0.09
    }
    log_aggregation_result(main_logger, round_num=1, aggregated_metrics=aggregated_metrics)
