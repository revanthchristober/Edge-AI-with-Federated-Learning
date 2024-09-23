import syft as sy
import tensorflow as tf
import numpy as np
import logging
from syft.core.node.common.client import Client
from syft.core.node.common.node_service.get_all_requests_message import GetAllRequestsMessage
from syft.core.node.common.node_service.accept_request_message import AcceptRequestMessage

# Setup logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Define the federated communication utilities for Syft and TensorFlow
class FederatedUtils:
    def __init__(self, server_url, port, auth_token=None):
        """
        Initializes the federated utilities to manage communication between nodes in the federated system.

        Args:
        - server_url (str): The URL of the federated learning server.
        - port (int): Port number for server communication.
        - auth_token (str): Optional authentication token for secure communication.
        """
        self.server_url = server_url
        self.port = port
        self.auth_token = auth_token
        self.client = self.connect_to_server()

    def connect_to_server(self):
        """
        Establishes a connection to the federated server using PySyft.
        
        Returns:
        - client (sy.Client): PySyft client instance connected to the federated server.
        """
        logger.info(f"Connecting to federated server at {self.server_url}:{self.port}")
        try:
            client = sy.login(url=self.server_url, port=self.port, token=self.auth_token)
            logger.info(f"Connected to federated server: {client}")
            return client
        except Exception as e:
            logger.error(f"Failed to connect to the server: {e}")
            raise e

    def get_available_datasets(self):
        """
        Retrieves the list of available datasets on the server.
        
        Returns:
        - datasets (list): A list of available datasets on the server.
        """
        try:
            datasets = self.client.datasets
            logger.info(f"Available datasets on server: {datasets}")
            return datasets
        except Exception as e:
            logger.error(f"Error retrieving datasets: {e}")
            raise e

    def load_remote_dataset(self, dataset_name):
        """
        Loads a specific dataset from the server.

        Args:
        - dataset_name (str): The name of the dataset to load.

        Returns:
        - dataset: The loaded dataset from the server.
        """
        logger.info(f"Loading dataset: {dataset_name}")
        try:
            dataset = self.client.datasets[dataset_name]
            logger.info(f"Successfully loaded dataset: {dataset_name}")
            return dataset
        except KeyError:
            logger.error(f"Dataset '{dataset_name}' not found on server.")
            raise KeyError(f"Dataset '{dataset_name}' not available.")
        except Exception as e:
            logger.error(f"Failed to load dataset '{dataset_name}': {e}")
            raise e

    def create_remote_model(self, model_fn, input_shape, output_shape):
        """
        Creates and sends a TensorFlow model to the server for federated training.

        Args:
        - model_fn (function): A function to build the TensorFlow model.
        - input_shape (tuple): The input shape for the model.
        - output_shape (tuple): The output shape (number of classes) for the model.

        Returns:
        - model: The compiled TensorFlow model ready for federated learning.
        """
        logger.info("Creating and sending model to the federated server.")
        try:
            # Define the model using the provided function
            model = model_fn(input_shape, output_shape)
            # Convert model to TensorFlow/Keras format
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            logger.info("Model compiled successfully.")
            return model
        except Exception as e:
            logger.error(f"Error creating the model: {e}")
            raise e

    def send_model_to_server(self, model, model_name):
        """
        Sends the compiled model to the federated server for training.

        Args:
        - model: The TensorFlow model to be sent.
        - model_name (str): The name to assign to the model on the server.
        """
        logger.info(f"Sending model '{model_name}' to federated server.")
        try:
            self.client.models.upload(model, model_name=model_name)
            logger.info(f"Model '{model_name}' successfully sent to server.")
        except Exception as e:
            logger.error(f"Failed to send model '{model_name}' to server: {e}")
            raise e

    def request_model_update(self, model_name, client_name):
        """
        Requests the federated server to update the model parameters for the current client.

        Args:
        - model_name (str): The name of the model to update.
        - client_name (str): Name of the client requesting the update.
        """
        logger.info(f"Requesting model update for client '{client_name}' from server...")
        try:
            # Retrieve all requests for model updates
            requests = self.client.send(GetAllRequestsMessage)
            for req in requests:
                if req.model_name == model_name and req.client_name == client_name:
                    # Accept the request for this client and model
                    self.client.send(AcceptRequestMessage(req.request_id))
                    logger.info(f"Accepted model update request for client '{client_name}'.")
                    return req
        except Exception as e:
            logger.error(f"Failed to request model update: {e}")
            raise e

    def receive_model_from_server(self, model_name):
        """
        Receives the updated model from the federated server after aggregation.

        Args:
        - model_name (str): The name of the model to retrieve.

        Returns:
        - model: The updated model with new parameters.
        """
        logger.info(f"Receiving updated model '{model_name}' from server.")
        try:
            model = self.client.models[model_name]
            logger.info(f"Successfully received updated model: {model_name}")
            return model
        except KeyError:
            logger.error(f"Model '{model_name}' not found on server.")
            raise KeyError(f"Model '{model_name}' not available.")
        except Exception as e:
            logger.error(f"Failed to receive model '{model_name}': {e}")
            raise e

    def train_model_on_client(self, model, train_data, epochs, batch_size):
        """
        Trains the model on the client node using local data.

        Args:
        - model: The TensorFlow model to be trained.
        - train_data: The local training dataset.
        - epochs (int): Number of training epochs.
        - batch_size (int): Batch size for training.

        Returns:
        - model: The trained model with updated weights.
        """
        logger.info(f"Training model locally on client for {epochs} epochs.")
        try:
            model.fit(train_data, epochs=epochs, batch_size=batch_size)
            logger.info(f"Training completed successfully.")
            return model
        except Exception as e:
            logger.error(f"Error during local training: {e}")
            raise e

    def aggregate_models(self, model_list):
        """
        Aggregates multiple models using federated averaging (FedAvg).

        Args:
        - model_list (list): A list of models (as TensorFlow models) from different clients.

        Returns:
        - aggregated_model: The aggregated model after averaging the parameters.
        """
        logger.info("Performing federated averaging on client models.")
        try:
            # Initialize an empty model for aggregation
            aggregated_model = model_list[0]
            aggregated_weights = np.array(aggregated_model.get_weights())
            
            # Average weights from all models
            for model in model_list[1:]:
                weights = np.array(model.get_weights())
                aggregated_weights = np.add(aggregated_weights, weights)

            # Average out the weights by dividing by the number of clients
            aggregated_weights /= len(model_list)
            aggregated_model.set_weights(aggregated_weights)

            logger.info("Model aggregation successful.")
            return aggregated_model
        except Exception as e:
            logger.error(f"Error during model aggregation: {e}")
            raise e

# Example model function to create a TensorFlow model
def example_model_fn(input_shape, output_shape):
    """
    Example model function to create a simple TensorFlow CNN model.
    Args:
    - input_shape (tuple): Shape of the input data.
    - output_shape (int): Number of output classes.

    Returns:
    - model: Compiled TensorFlow CNN model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(output_shape, activation='softmax')
    ])
    
    return model

# Example usage:
if __name__ == "__main__":
    # Initialize federated utility instance
    federated_utils = FederatedUtils(server_url="localhost", port=5000, auth_token="federated_secret")
    
    # Connect to server, send model, request updates, etc.
    model = federated_utils.create_remote_model(example_model_fn, input_shape=(28, 28, 1), output_shape=10)
    federated_utils.send_model_to_server(model, model_name="CNN_FEMNIST_Model")
