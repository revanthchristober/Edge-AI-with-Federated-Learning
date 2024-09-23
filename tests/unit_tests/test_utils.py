import unittest
from utils.data_utils import split_data, preprocess_data
from utils.federated_utils import FederatedUtils
from utils.logging_utils import setup_logger
import os

class TestDataUtils(unittest.TestCase):

    def test_split_data(self):
        """
        Test the split_data function to ensure correct train-test split.
        """
        data = [(i, i) for i in range(100)]  # Mock data
        train_data, test_data = split_data(data, train_fraction=0.8)

        self.assertEqual(len(train_data), 80)
        self.assertEqual(len(test_data), 20)

    def test_preprocess_data(self):
        """
        Test the preprocess_data function for handling raw image data.
        """
        raw_data = [([0] * 784, 1)]  # Mock a 28x28 flattened image with a label
        processed_data = preprocess_data(raw_data)

        # Check that processed data is reshaped correctly
        self.assertEqual(processed_data[0][0].shape, (28, 28))
        self.assertEqual(processed_data[0][1], 1)

class TestFederatedUtils(unittest.TestCase):

    def setUp(self):
        self.server_url = "localhost"
        self.port = 5000
        self.federated_utils = FederatedUtils(self.server_url, self.port)

    def test_connect_to_server(self):
        """
        Test if the client can successfully connect to the server.
        """
        client = self.federated_utils.connect_to_server()
        self.assertIsNotNone(client)

    def test_dataset_loading(self):
        """
        Test loading datasets from the server.
        """
        datasets = self.federated_utils.get_available_datasets()
        self.assertIn("FEMNIST", datasets)

class TestLoggingUtils(unittest.TestCase):

    def test_setup_logger(self):
        """
        Test logger setup and ensure it creates the appropriate log file.
        """
        logger = setup_logger("test_logger", log_file="test.log")
        logger.info("Test log entry")

        # Check that the log file was created
        self.assertTrue(os.path.exists("test.log"))

        # Clean up log file after the test
        os.remove("test.log")

if __name__ == "__main__":
    unittest.main()
