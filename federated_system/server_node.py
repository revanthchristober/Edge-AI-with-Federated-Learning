import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import tensorflow as tf
import syft as sy
import numpy as np
import logging
from syft.core.node.common.client import Client
from syft.core.node.vm.vm import VirtualMachine
from syft.core.plan.plan import Plan
from syft.core.node.device.device import Device
from syft.lib.tensorflow import KerasHook

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Apply Syft hook to TensorFlow for federated learning capabilities
hook = KerasHook(tf.keras)

# Server node responsible for aggregating model updates
class ServerNode:
    def __init__(self, model, aggregation_strategy="fed_avg", threshold_clients=3):
        """
        Initialize the central server node for federated learning.

        Args:
        - model: The global model that will be updated using federated learning.
        - aggregation_strategy: Aggregation method for combining client updates. Default is "fed_avg".
        - threshold_clients: Number of client updates required to perform aggregation.
        """
        self.model = model
        self.aggregation_strategy = aggregation_strategy
        self.threshold_clients = threshold_clients
        self.client_updates = []

        # Initialize the virtual machine as the server to receive updates from clients
        self.vm = VirtualMachine(name="server_vm")
        self.syft_server = self.vm.get_root_client()

        logger.info(f"Server initialized with aggregation strategy: {self.aggregation_strategy}")

    def receive_model_update(self, weights):
        """
        Receive model weights update from a client.

        Args:
        - weights: The updated model weights from the client.
        """
        self.client_updates.append(weights)
        logger.info(f"Received model update from client. Total updates received: {len(self.client_updates)}")

        # If enough clients have sent updates, perform aggregation
        if len(self.client_updates) >= self.threshold_clients:
            self.aggregate_model_updates()

    def aggregate_model_updates(self):
        """
        Aggregate model updates from multiple clients using the specified aggregation strategy.
        """
        logger.info("Aggregating model updates from clients...")

        if self.aggregation_strategy == "fed_avg":
            # Perform Federated Averaging (FedAvg) aggregation
            new_weights = self.federated_averaging(self.client_updates)
            self.model.set_weights(new_weights)

            # Reset client updates after aggregation
            self.client_updates = []

            logger.info("Model aggregation complete.")
        else:
            raise NotImplementedError(f"Aggregation strategy {self.aggregation_strategy} is not implemented.")

    def federated_averaging(self, client_updates):
        """
        Federated Averaging (FedAvg) algorithm to average model weights.

        Args:
        - client_updates: List of model weights from different clients.

        Returns:
        - Averaged weights to update the global model.
        """
        logger.info("Performing Federated Averaging (FedAvg)...")

        num_clients = len(client_updates)
        new_weights = []

        # Iterate over layers of weights
        for layer_weights in zip(*client_updates):
            # Average the weights across clients
            avg_weights = np.mean(layer_weights, axis=0)
            new_weights.append(avg_weights)

        logger.info(f"FedAvg aggregation completed. Averaged across {num_clients} clients.")
        return new_weights

    def send_global_model(self, client_id):
        """
        Send the global model weights to a client.

        Args:
        - client_id: The ID of the client requesting the model.
        """
        weights = self.model.get_weights()

        # Send weights to client via Syft
        logger.info(f"Sending global model to client {client_id}")
        self.syft_server.send_model_update(weights=weights)

    def run(self):
        """
        Start the server to continuously listen for client updates and perform aggregation.
        """
        logger.info("Server is running and listening for client updates...")
        while True:
            pass  # In an actual system, this would have more robust event-driven behavior

if __name__ == "__main__":
    # Load a simple CNN model for the server to distribute and aggregate
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model (though the server does not train)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Create the server node
    server_node = ServerNode(model=model, aggregation_strategy="fed_avg", threshold_clients=3)

    # Simulate server receiving model updates and aggregating
    # In practice, this would be part of the server’s event loop
    logger.info("Server is ready to receive updates.")

    # Run the server
    server_node.run()
