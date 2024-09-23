import matplotlib.pyplot as plt
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_loss_curve(loss_history, title="Loss Curve"):
    """Plot the loss curve based on the history of loss values."""
    logger.info("Plotting loss curve...")
    plt.plot(loss_history)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

def plot_accuracy_curve(acc_history, title="Accuracy Curve"):
    """Plot the accuracy curve based on the history of accuracy values."""
    logger.info("Plotting accuracy curve...")
    plt.plot(acc_history)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

# Function to load history from a numpy file and plot both accuracy and loss
def plot_training_history(history_file):
    """Load training history and plot both loss and accuracy curves."""
    logger.info(f"Loading training history from {history_file}...")

    # Load history (assuming the history file is in npz format with keys 'loss' and 'accuracy')
    data = np.load(history_file)
    loss_history = data['loss']
    acc_history = data['accuracy']

    # Plot loss and accuracy curves
    plot_loss_curve(loss_history, title="Training Loss Curve")
    plot_accuracy_curve(acc_history, title="Training Accuracy Curve")

# Example usage
if __name__ == "__main__":
    # Simulated data for testing
    epochs = 10
    loss_history = np.random.rand(epochs)  # Simulated loss values
    acc_history = np.random.rand(epochs)   # Simulated accuracy values

    # Plotting the loss and accuracy
    plot_loss_curve(loss_history)
    plot_accuracy_curve(acc_history)
