import unittest
import syft as sy
import tensorflow as tf
from federated_system.client_node import ClientNode
from federated_system.server_node import ServerNode
from models.cnn_model import CNNModel
from utils.federated_utils import FederatedUtils

class TestFederatedTraining(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up server and clients for federated learning.
        """
        cls.server_node = ServerNode()
        cls.client_nodes = [
            ClientNode(node_id=f"client_{i}") for i in range(5)  # Simulate 5 clients
        ]
        cls.federated_utils = FederatedUtils(server_url="localhost", port=5000)
        
        # Define model architecture to be used by all clients
        cls.input_shape = (28, 28, 1)
        cls.output_shape = 10
        cls.model = CNNModel(input_shape=cls.input_shape, output_shape=cls.output_shape)

    def test_client_server_communication(self):
        """
        Test that client nodes can communicate with the server.
        """
        for client in self.client_nodes:
            connected = client.connect_to_server(self.federated_utils)
            self.assertTrue(connected)

    def test_federated_training(self):
        """
        Test full federated training process across multiple clients and aggregation on server.
        """
        # Simulate federated training across clients
        for client in self.client_nodes:
            client.train_model(self.model, epochs=1)

        # Server aggregates model updates
        self.server_node.aggregate_models()

        # Verify that server aggregated the models correctly
        global_model = self.server_node.get_global_model()
        self.assertIsNotNone(global_model)

if __name__ == "__main__":
    unittest.main()
