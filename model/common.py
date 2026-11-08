"""Shared paths, feature definitions and evaluation functions."""

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"
DATA_FILE = PROJECT_ROOT / "data" / "students_dropout_academic_success.csv"
TEST_FILE = PROJECT_ROOT / "test_data.csv"
METADATA_FILE = MODEL_DIR / "model_metadata.json"
RESULTS_FILE = MODEL_DIR / "test_metrics.csv"

TARGET_COL = "Target"
CLASS_LABELS = ["Dropout", "Enrolled", "Graduate"]

# Treat these integer-coded attributes as categorical features.
CODED_CATEGORY_COLS = [
    "Marital_status",
    "Application_mode",
    "Course",
    "Previous_qualification",
    "Nacionality",  # spelling kept as in the UCI file
    "Mothers_qualification",
    "Fathers_qualification",
    "Mothers_occupation",
    "Fathers_occupation",
]

# Display name -> file name of the saved pipeline inside model/
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes (Gaussian)": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def predict_in_label_order(model, X):
    """Predicted labels plus class probabilities with columns in CLASS_LABELS order."""
    proba = model.predict_proba(X)
    column_order = [list(model.classes_).index(label) for label in CLASS_LABELS]
    return model.predict(X), proba[:, column_order]


def macro_ovr_auc(y_true, proba, classes):
    """One-vs-rest AUC averaged over classes (macro).

    Classes that are missing from y_true (or that make up all of it) are skipped,
    so a small uploaded file that only contains some classes still gets a score.
    Returns NaN when no class can be scored.
    """
    y_true = np.asarray(y_true)
    per_class = []
    for idx, label in enumerate(classes):
        is_label = (y_true == label).astype(int)
        if 0 < is_label.sum() < len(is_label):
            per_class.append(roc_auc_score(is_label, proba[:, idx]))
    return float(np.mean(per_class)) if per_class else float("nan")


def score_predictions(y_true, y_pred, proba, classes):
    """Calculate the six metrics, giving each class equal weight in macro averages."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": macro_ovr_auc(y_true, proba, classes),
        "Precision": precision_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        ),
        "F1": f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
