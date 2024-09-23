import unittest
import tensorflow as tf
from models.cnn_model import CNNModel
from models.lstm_model import LSTMModel

class TestModelArchitecture(unittest.TestCase):
    
    def test_cnn_model(self):
        """
        Test if CNN model can be built correctly with the specified input shape and output classes.
        """
        input_shape = (28, 28, 1)  # Example input shape for FEMNIST
        output_shape = 10  # Example number of output classes
        model = CNNModel(input_shape, output_shape)
        
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(model.input_shape[1:], input_shape)
        self.assertEqual(model.output_shape[-1], output_shape)

    def test_lstm_model(self):
        """
        Test if LSTM model can be built correctly for sequence data.
        """
        input_shape = (100, 28)  # Example sequence length and feature dimension
        output_shape = 10  # Number of output classes
        model = LSTMModel(input_shape, output_shape)
        
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(model.input_shape[1:], input_shape)
        self.assertEqual(model.output_shape[-1], output_shape)

if __name__ == "__main__":
    unittest.main()