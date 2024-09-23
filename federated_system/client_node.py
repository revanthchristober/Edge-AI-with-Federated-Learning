import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import tensorflow as tf
import syft as sy
import numpy as np
import logging
from syft.core.node.common.client import Client
from syft.core.plan.plan import Plan
from syft.tensorflow import KerasHook

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Apply Syft hook to TensorFlow for federated learning capabilities
hook = KerasHook(tf.keras)

class ClientNode:
    def __init__(self, client_id, model, dataset, server_address, batch_size=32):
        self.client_id = client_id
        self.model = model
        self.dataset = dataset
        self.server_address = server_address
        self.batch_size = batch_size

        # Create a PySyft client to communicate with the server
        self.syft_client = Client(network_url=server_address, name=f"client_{client_id}")
        logger.info(f"Initialized client {self.client_id} and connected to server at {self.server_address}")

    def preprocess_data(self, data):
        """
        Preprocess the local dataset (e.g., normalization).
        """
        logger.info(f"Client {self.client_id}: Preprocessing data...")
        data = data / 255.0  # Normalize image data
        return data

    def train_model(self, epochs=5):
        """
        Train the local model on the dataset.
        """
        logger.info(f"Client {self.client_id}: Starting local training for {epochs} epochs.")

        # Unpack the dataset
        x_train, y_train = self.dataset

        # Preprocess the dataset
        x_train = self.preprocess_data(x_train)

        # Train the model on local data
        history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=self.batch_size, verbose=1)
        logger.info(f"Client {self.client_id}: Finished training.")

        return history.history

    def send_model_to_server(self):
        """
        Send the updated model's weights to the central server.
        """
        logger.info(f"Client {self.client_id}: Sending model updates to server...")

        # Extract model weights
        weights = self.model.get_weights()
        weight_list = [np.array(w) for w in weights]

        # Send weights to the federated server via PySyft
        self.syft_client.send_model_update(weights=weight_list)

        logger.info(f"Client {self.client_id}: Model update sent to server.")

    def receive_model_from_server(self):
        """
        Receive the global model weights from the server and update the local model.
        """
        logger.info(f"Client {self.client_id}: Receiving model from server...")
        weights = self.syft_client.get_model()

        if weights:
            self.model.set_weights(weights)
            logger.info(f"Client {self.client_id}: Model weights updated from server.")
        else:
            logger.warning(f"Client {self.client_id}: No model received from server.")

    def run(self, epochs=5):
        """
        Full lifecycle of the client node: training and communication with the server.
        """
        # Step 1: Receive global model from server
        self.receive_model_from_server()

        # Step 2: Train model locally
        self.train_model(epochs)

        # Step 3: Send updated model to server
        self.send_model_to_server()

if __name__ == "__main__":
    import argparse

    # Parse client-specific arguments
    parser = argparse.ArgumentParser(description="Federated Learning Client Node")
    parser.add_argument('--client-id', type=int, required=True, help="Client node identifier")
    parser.add_argument('--server-address', type=str, required=True, help="Address of the federated server node")
    
    args = parser.parse_args()

    # Load some example data (e.g., MNIST dataset)
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1)  # Reshape for CNN input
    y_train = tf.keras.utils.to_categorical(y_train, 10)  # Convert labels to one-hot encoding

    # Define a simple CNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model with loss and optimizer
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Create the client node instance
    client_node = ClientNode(
        client_id=args.client_id, 
        model=model, 
        dataset=(x_train, y_train), 
        server_address=args.server_address
    )

    # Start the client node
    client_node.run(epochs=5)
