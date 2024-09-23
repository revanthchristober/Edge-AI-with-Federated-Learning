import unittest
from training.local_training import LocalTrainer
from models.cnn_model import CNNModel
from utils.data_utils import load_data

class TestLocalTraining(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Load data and initialize model for local training.
        """
        cls.input_shape = (28, 28, 1)
        cls.output_shape = 10
        cls.model = CNNModel(input_shape=cls.input_shape, output_shape=cls.output_shape)
        
        # Load data for testing
        cls.train_data, cls.test_data = load_data("data_path", batch_size=32)

    def test_local_training_process(self):
        """
        Test the local training process to ensure the model is updated.
        """
        trainer = LocalTrainer(model=self.model, train_data=self.train_data, test_data=self.test_data)
        initial_loss = trainer.evaluate(self.test_data)
        
        # Train for one epoch
        trainer.train(epochs=1)
        
        final_loss = trainer.evaluate(self.test_data)
        
        # Ensure that training reduced the loss
        self.assertLess(final_loss, initial_loss)

if __name__ == "__main__":
    unittest.main()
