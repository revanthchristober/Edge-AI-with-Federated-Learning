from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_metrics(y_true, y_pred):
    """Compute precision, recall, and F1-score for the model's predictions."""
    logger.info("Computing custom evaluation metrics...")

    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    
    return precision, recall, f1

# Example of how to use the metrics function after predictions
if __name__ == "__main__":
    # Example data (y_true: actual labels, y_pred: predicted labels)
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 0, 1])

    # Calculate metrics
    compute_metrics(y_true, y_pred)
