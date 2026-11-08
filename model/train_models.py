"""Train and evaluate five classifiers for ML Assignment 2.

Hyperparameters are selected using five-fold cross-validation on the training
split. Fitted pipelines, test scores and training metadata are saved in model/.
Run from the project directory with: python model/train_models.py
"""

import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.common import (  # noqa: E402
    CLASS_LABELS,
    CODED_CATEGORY_COLS,
    DATA_FILE,
    METADATA_FILE,
    METRIC_NAMES,
    MODEL_DIR,
    MODEL_FILES,
    RESULTS_FILE,
    TARGET_COL,
    TEST_FILE,
    predict_in_label_order,
    score_predictions,
)

RANDOM_STATE = 42
TEST_FRACTION = 0.20


def make_preprocessor(numeric_cols):
    """Encode categorical features and standardise the remaining columns.

    Codes seen fewer than 10 times in training are pooled into one
    'infrequent' column to limit the number of encoded features.
    """
    return ColumnTransformer(
        [
            (
                "categories",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=10,
                    sparse_output=False,  # GaussianNB needs a dense matrix
                ),
                CODED_CATEGORY_COLS,
            ),
            ("numbers", StandardScaler(), numeric_cols),
        ]
    )


def candidate_models():
    """Each entry: (estimator, hyperparameter grid searched with 5-fold CV)."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=3000),
            {"clf__C": [0.05, 0.1, 0.5, 1, 5], "clf__class_weight": [None, "balanced"]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "clf__criterion": ["gini", "entropy"],
                "clf__max_depth": [4, 6, 8, 12, None],
                "clf__min_samples_leaf": [1, 10, 25],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "kNN": (
            KNeighborsClassifier(),
            {"clf__n_neighbors": [5, 11, 17, 25, 35], "clf__weights": ["uniform", "distance"]},
        ),
        "Naive Bayes (Gaussian)": (
            GaussianNB(),
            {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-2, 1e-1, 0.3, 1.0]},
        ),
        # Sigmoid calibration provides class probabilities for AUC evaluation.
        "Support Vector Machine": (
            CalibratedClassifierCV(SVC(kernel="rbf"), ensemble=False, cv=5),
            {
                "clf__estimator__C": [1, 5, 10],
                "clf__estimator__gamma": ["scale", 0.002, 0.005],
                "clf__estimator__class_weight": [None, "balanced"],
            },
        ),
    }


def main():
    data = pd.read_csv(DATA_FILE)
    features = data.drop(columns=[TARGET_COL])
    target = data[TARGET_COL]
    numeric_cols = [c for c in features.columns if c not in CODED_CATEGORY_COLS]

    print(f"Dataset: {data.shape[0]} rows, {features.shape[1]} features")
    print("Class counts:", target.value_counts().to_dict())

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=TEST_FRACTION, stratify=target, random_state=RANDOM_STATE
    )
    pd.concat([X_test, y_test], axis=1).to_csv(TEST_FILE, index=False)
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)} -> saved {TEST_FILE.name}\n")

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, metadata = [], {
        "sklearn_version": sklearn.__version__,
        "random_state": RANDOM_STATE,
        "test_fraction": TEST_FRACTION,
        "feature_columns": list(features.columns),
        "class_labels": CLASS_LABELS,
        "models": {},
    }

    for name, (estimator, grid) in candidate_models().items():
        pipeline = Pipeline([("prep", make_preprocessor(numeric_cols)), ("clf", estimator)])
        search = GridSearchCV(pipeline, grid, scoring="f1_macro", cv=folds, n_jobs=-1)

        started = time.perf_counter()
        search.fit(X_train, y_train)
        seconds = time.perf_counter() - started

        best = search.best_estimator_
        y_pred, proba = predict_in_label_order(best, X_test)
        scores = score_predictions(y_test, y_pred, proba, CLASS_LABELS)

        joblib.dump(best, MODEL_DIR / MODEL_FILES[name])
        rows.append({"Model": name, **scores})
        metadata["models"][name] = {
            "file": MODEL_FILES[name],
            "best_params": {
                k.replace("clf__", "").replace("estimator__", ""): v
                for k, v in search.best_params_.items()
            },
            "cv_f1_macro": round(search.best_score_, 4),
            "tuning_seconds": round(seconds, 1),
        }
        print(f"{name:<24} best params {metadata['models'][name]['best_params']}")
        print(f"{'':<24} CV macro-F1 {search.best_score_:.4f} | tuned in {seconds:.1f}s")

    results = pd.DataFrame(rows).set_index("Model")[METRIC_NAMES].round(4)
    results.to_csv(RESULTS_FILE)
    METADATA_FILE.write_text(json.dumps(metadata, indent=2, default=str))

    print("\nTest-set results (precision/recall/F1 macro-averaged, AUC one-vs-rest macro):")
    print(results.to_string())
    print(f"\nBest by F1: {results['F1'].idxmax()} | best by MCC: {results['MCC'].idxmax()}")
    print(
        f"Models saved with scikit-learn {sklearn.__version__}. "
        "requirements.txt must pin this same version for the app to load them."
    )


if __name__ == "__main__":
    main()
