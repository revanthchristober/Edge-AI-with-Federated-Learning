import unittest
import time
import tensorflow as tf
from federated_system.server_node import ServerNode
from federated_system.client_node import ClientNode
from utils.profiler_utils import PerformanceProfiler

class TestSystemPerformance(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up server, clients, and performance profiler.
        """
        cls.server_node = ServerNode()
        cls.client_nodes = [
            ClientNode(node_id=f"client_{i}") for i in range(50)  # Stress test with 50 clients
        ]
        cls.profiler = PerformanceProfiler()

        # Model architecture for testing
        cls.input_shape = (28, 28, 1)
        cls.output_shape = 10
        cls.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=cls.input_shape),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(cls.output_shape, activation='softmax')
        ])

    def test_large_scale_federated_training(self):
        """
        Test large-scale federated training with multiple clients and track performance metrics.
        """
        start_time = time.time()
        
        # Initialize profiling
        self.profiler.start()

        for client in self.client_nodes:
            client.train_model(self.model, epochs=1)

        # Aggregate updates on server
        self.server_node.aggregate_models()

        # End profiling
        self.profiler.stop()

        elapsed_time = time.time() - start_time
        memory_usage = self.profiler.get_memory_usage()

        # Ensure that system can handle large-scale training within reasonable time and memory limits
        self.assertLess(elapsed_time, 300)  # Test should complete in under 5 minutes
        self.assertLess(memory_usage, 5000)  # Memory usage should be under 5GB

if __name__ == "__main__":
    unittest.main()
