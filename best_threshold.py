from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from scipy.optimize import minimize_scalar
import numpy as np

import numpy as np
import torch
from sklearn.metrics import f1_score


def find_best_threshold_with_scipy(y_scores, y_true, metric=f1_score):
    """
    Find the best threshold for binary classification using raw scores and a specified metric.

    Parameters:
        y_true (np.ndarray): True labels (0 or 1).
        y_scores (np.ndarray): Raw scores from the classifier.
        metric (callable): Function to evaluate the threshold.

    Returns:
        float: The best threshold.
        float: The best metric score.
    """
    # Ensure inputs are on GPU using PyTorch
    y_true_tensor = torch.tensor(y_true, dtype=torch.int, device='cuda')
    y_scores_tensor = torch.tensor(y_scores, dtype=torch.float64, device='cuda')

    # Generate candidate thresholds from the range of scores
    thresholds = torch.linspace(y_scores_tensor.min(), y_scores_tensor.max(), steps=1000, device='cuda')

    best_threshold = 0
    best_metric_score = -1
    best_precision = 0
    best_recall = 0
    # Iterate over thresholds to find the best one
    for threshold in thresholds:
        # Threshold predictions
        preds = (y_scores_tensor <= threshold).int()

        # Compute TP, FP, FN for F1-score (or similar metrics)
        tp = torch.sum((preds == 1) & (y_true_tensor == 1))
        fp = torch.sum((preds == 1) & (y_true_tensor == 0))
        fn = torch.sum((preds == 0) & (y_true_tensor == 1))

        # Compute F1-score or other metrics
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        score = 2 * precision * recall / (precision + recall + 1e-8)

        if score > best_metric_score:
            best_metric_score = score
            best_threshold = threshold.item()
            best_recall = recall.item()
            best_precision = precision.item()

    return best_threshold, best_metric_score, best_precision, best_recall


if __name__ == '__main__':
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    labels = np.array([0, 0, 1, 1])

    best_threshold, best_f1, precision, recall = find_best_threshold_with_scipy(scores, labels)
    print(f"Best Threshold: {best_threshold}")
    print(f"Best F1 Score: {best_f1}, Precision: {precision}, Recall: {recall}")
