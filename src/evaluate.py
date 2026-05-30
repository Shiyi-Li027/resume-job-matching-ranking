import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def classification_metrics(y_true, y_pred) -> dict:
    """Calculate classification metrics for binary hiring prediction."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def precision_at_k(y_true, y_score, k: int = 10) -> float:
    """Calculate Precision@k based on predicted matching scores."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if len(y_true) == 0:
        return 0.0

    k = min(k, len(y_true))
    order = np.argsort(y_score)[::-1][:k]
    return float(np.mean(y_true[order]))


def dcg_at_k(labels, k: int = 10) -> float:
    """Calculate discounted cumulative gain at k."""
    labels = np.asarray(labels)[:k]
    gains = (2 ** labels - 1) / np.log2(np.arange(2, labels.size + 2))
    return float(np.sum(gains))


def ndcg_at_k(y_true, y_score, k: int = 10) -> float:
    """Calculate NDCG@k based on predicted matching scores."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if len(y_true) == 0:
        return 0.0

    order = np.argsort(y_score)[::-1]
    ranked_labels = y_true[order]
    ideal_labels = np.sort(y_true)[::-1]
    ideal = dcg_at_k(ideal_labels, k)

    if ideal == 0:
        return 0.0

    return dcg_at_k(ranked_labels, k) / ideal


def ranking_metrics(y_true, y_score, k: int = 10) -> dict:
    """Calculate ranking metrics for candidate ranking."""
    return {
        f"precision@{k}": precision_at_k(y_true, y_score, k=k),
        f"ndcg@{k}": ndcg_at_k(y_true, y_score, k=k),
    }


def evaluate_predictions(y_true, y_pred, y_score, k: int = 10) -> dict:
    """Combine classification and ranking metrics into one dictionary."""
    metrics = classification_metrics(y_true, y_pred)
    metrics.update(ranking_metrics(y_true, y_score, k=k))
    return metrics


def make_confusion_matrix_df(y_true, y_pred) -> pd.DataFrame:
    """Create a readable confusion matrix DataFrame."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        cm,
        index=["true_reject", "true_select"],
        columns=["pred_reject", "pred_select"],
    )


def print_report(y_true, y_pred):
    """Print sklearn classification report."""
    print(classification_report(y_true, y_pred, zero_division=0))
