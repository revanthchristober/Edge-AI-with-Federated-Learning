import unittest
import tensorflow as tf
from training.local_training import LocalTrainer
from training.federated_training import FederatedTrainer
from utils.data_utils import load_data
from models.cnn_model import CNNModel
import syft as sy

class TestTrainingProcess(unittest.TestCase):
    
    def setUp(self):
        """
        Setup function to load data and initialize model for testing.
        """
        self.input_shape = (28, 28, 1)
        self.output_shape = 10
        self.model = CNNModel(self.input_shape, self.output_shape)
        
        # Mock dataset for testing
        self.train_data, self.test_data = load_data("test_data_path", batch_size=32)

    def test_local_training(self):
        """
        Test the local training process and ensure it runs without errors.
        """
        trainer = LocalTrainer(model=self.model, train_data=self.train_data, test_data=self.test_data)
        initial_loss = trainer.evaluate(self.test_data)

        # Perform training for one epoch
        trainer.train(epochs=1)
        final_loss = trainer.evaluate(self.test_data)

        # Ensure training improves the model (lower loss)
        self.assertLess(final_loss, initial_loss)

    def test_federated_training(self):
        """
        Test the federated training process using Syft and ensure it runs without errors.
        """
        # Simulate federated client environment
        client_node = sy.VirtualMachineClient()

        federated_trainer = FederatedTrainer(model=self.model, client_node=client_node, train_data=self.train_data)
        initial_loss = federated_trainer.evaluate(self.test_data)

        # Simulate one round of federated training
        federated_trainer.train(epochs=1)
        final_loss = federated_trainer.evaluate(self.test_data)

        # Ensure the federated training reduces the loss
        self.assertLess(final_loss, initial_loss)

if __name__ == "__main__":
    unittest.main()
