import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import syft as sy

# Image and Dataset Configuration Constants
IMAGE_SIZE = (28, 28)  # For FEMNIST (can be adjusted as per dataset)
NUM_CLASSES = 62  # FEMNIST has 62 classes (digits and letters)

# Set up PySyft hook for TensorFlow
hook = sy.TFFederatedTensorFlowHook(tf)

# Preprocess individual image and label
def preprocess_image(image, label):
    """
    Preprocesses image data by resizing and normalizing.
    
    Args:
    - image: Input image to preprocess.
    - label: Label associated with the image.
    
    Returns:
    - Preprocessed image and label.
    """
    image = tf.image.resize(image, IMAGE_SIZE)
    image = image / 255.0  # Normalize to [0, 1]
    return image, tf.one_hot(label, NUM_CLASSES)  # Convert label to one-hot encoding

# Load and preprocess the dataset (e.g., FEMNIST)
def load_femnist_data(data_dir, batch_size=32, shuffle=True):
    """
    Loads FEMNIST data and preprocesses it into TensorFlow datasets.
    
    Args:
    - data_dir (str): Directory where data is stored.
    - batch_size (int): Batch size for training.
    - shuffle (bool): Whether to shuffle the dataset.
    
    Returns:
    - train_dataset: Preprocessed TensorFlow dataset for training.
    - test_dataset: Preprocessed TensorFlow dataset for testing.
    """
    # Load dataset using image_dataset_from_directory
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, 'train'),
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode='int'
    )
    
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, 'test'),
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode='int'
    )
    
    # Preprocess datasets
    train_dataset = train_dataset.map(preprocess_image)
    test_dataset = test_dataset.map(preprocess_image)

    if shuffle:
        train_dataset = train_dataset.shuffle(buffer_size=1000)
    
    return train_dataset, test_dataset

# Split dataset for federated learning
def split_data_across_clients(dataset, num_clients):
    """
    Splits dataset across multiple clients for federated learning.
    
    Args:
    - dataset: TensorFlow dataset to split.
    - num_clients: Number of clients to split the data between.
    
    Returns:
    - A list of datasets, one for each client.
    """
    dataset_list = list(dataset)
    client_data_size = len(dataset_list) // num_clients
    
    client_datasets = [
        tf.data.Dataset.from_tensor_slices(dataset_list[i * client_data_size:(i + 1) * client_data_size])
        for i in range(num_clients)
    ]
    
    return client_datasets

# Convert TensorFlow dataset into PySyft DataLoader
def create_syft_dataloader(dataset, batch_size):
    """
    Converts TensorFlow dataset into PySyft-compatible DataLoader.
    
    Args:
    - dataset: TensorFlow dataset.
    - batch_size: Size of each batch.
    
    Returns:
    - syft_dataset: A PySyft DataLoader.
    """
    dataloader = dataset.batch(batch_size)
    
    # PySyft-specific: Wrap the TensorFlow DataLoader into a PySyft Dataset
    syft_dataset = sy.Dataset(dataloader)
    return syft_dataset

# Function to distribute the data to clients in a federated setup
def prepare_federated_data(data_dir, num_clients, batch_size=32):
    """
    Prepare the FEMNIST data for federated learning by splitting across clients.
    
    Args:
    - data_dir (str): Directory containing the FEMNIST dataset.
    - num_clients (int): Number of clients.
    - batch_size (int): Batch size for training.
    
    Returns:
    - client_datasets: List of datasets, one per client.
    - test_dataset: A test dataset used for evaluation.
    """
    # Load and preprocess FEMNIST dataset
    train_dataset, test_dataset = load_femnist_data(data_dir, batch_size=batch_size)
    
    # Split the training data across clients
    client_datasets = split_data_across_clients(train_dataset, num_clients)
    
    # Convert each client dataset to Syft DataLoader
    syft_client_datasets = [create_syft_dataloader(client_data, batch_size) for client_data in client_datasets]
    
    return syft_client_datasets, test_dataset

# Helper function to preprocess a single client's dataset
def preprocess_client_data(client_data, batch_size):
    """
    Preprocess a single client's data for federated training.
    
    Args:
    - client_data: TensorFlow dataset for a single client.
    - batch_size: Batch size for training.
    
    Returns:
    - Batched dataset ready for training.
    """
    return client_data.batch(batch_size)

# Function to prepare test data
def prepare_test_data(data_dir, batch_size=32):
    """
    Prepare the test data for evaluation in federated learning.
    
    Args:
    - data_dir (str): Directory where the test dataset is stored.
    - batch_size (int): Batch size for evaluation.
    
    Returns:
    - test_dataset: Preprocessed TensorFlow dataset for testing.
    """
    _, test_dataset = load_femnist_data(data_dir, batch_size=batch_size)
    return test_dataset

# Function to save dataset statistics
def save_dataset_statistics(data, file_path):
    """
    Save statistics of the dataset (e.g., size, distribution) to a file.
    
    Args:
    - data: TensorFlow dataset to gather statistics on.
    - file_path: Path to save statistics.
    """
    num_samples = len(list(data))
    with open(file_path, 'w') as f:
        f.write(f"Dataset Statistics:\n")
        f.write(f"Number of samples: {num_samples}\n")
        f.write(f"Number of classes: {NUM_CLASSES}\n")
    
    print(f"Dataset statistics saved to {file_path}")

# Example usage
if __name__ == "__main__":
    data_dir = './data/raw/femnist'
    num_clients = 5
    batch_size = 32

    # Prepare federated datasets
    client_datasets, test_dataset = prepare_federated_data(data_dir, num_clients, batch_size)

    # Example: Save dataset statistics for a client
    save_dataset_statistics(client_datasets[0], './data/client_0_stats.txt')

    # Preprocess test dataset
    test_data = prepare_test_data(data_dir, batch_size)
