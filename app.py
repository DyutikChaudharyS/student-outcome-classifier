"""Streamlit interface for evaluating the student outcome classifiers.

Run from the project directory with: streamlit run app.py
"""

import io
import json
import re

import altair as alt
import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from model.common import (
    CLASS_LABELS,
    METADATA_FILE,
    METRIC_NAMES,
    MODEL_DIR,
    MODEL_FILES,
    TARGET_COL,
    TEST_FILE,
    predict_in_label_order,
    score_predictions,
)

st.set_page_config(page_title="Student Outcome Predictor", layout="wide")


BLUE_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BAR_BLUE = "#2a78d6"

METRIC_HELP = {
    "Accuracy": "Share of students whose outcome was predicted correctly.",
    "AUC": "Macro one-vs-rest ROC AUC. Classes without positive or negative examples are excluded.",
    "Precision": "Macro average: of the students predicted as a class, how many really were (each class weighted equally).",
    "Recall": "Macro average: of the students truly in a class, how many the model found (each class weighted equally).",
    "F1": "Macro-averaged harmonic mean of precision and recall.",
    "MCC": "Matthews correlation coefficient, ranging from -1 to 1. Higher values indicate better agreement.",
}


@st.cache_resource(show_spinner="Loading the five trained models...")
def load_models():
    return {name: joblib.load(MODEL_DIR / file) for name, file in MODEL_FILES.items()}


@st.cache_data
def load_metadata():
    return json.loads(METADATA_FILE.read_text())


def simplify(column_name):
    """'Mother's qualification' and 'Mothers_qualification' -> 'mothers_qualification'."""
    return re.sub(r"[^0-9a-z]+", "_", column_name.lower().replace("'", "")).strip("_")


@st.cache_data
def read_student_csv(raw_bytes, expected_columns):
    """Read a CSV and match its headers to the training feature names.

    This also accepts the original UCI file, whose headers use spaces and
    apostrophes instead of underscores.
    """
    frame = pd.read_csv(io.BytesIO(raw_bytes), sep=None, engine="python", encoding="utf-8-sig")
    lookup = {simplify(col): col for col in expected_columns + [TARGET_COL]}
    return frame.rename(columns=lambda c: lookup.get(simplify(str(c)), c))


@st.cache_data
def evaluate_every_model(_models, X, y):
    """Return evaluation scores and predictions for each classifier."""
    table, outputs = [], {}
    for name, model in _models.items():
        y_pred, proba = predict_in_label_order(model, X)
        outputs[name] = (y_pred, proba)
        if y is not None:
            table.append({"Model": name, **score_predictions(y, y_pred, proba, CLASS_LABELS)})
    scores = pd.DataFrame(table).set_index("Model") if table else None
    return scores, outputs


def confusion_matrix_chart(y_true, y_pred):
    counts = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    cells = []
    for i, actual in enumerate(CLASS_LABELS):
        row_total = counts[i].sum()
        for j, predicted in enumerate(CLASS_LABELS):
            cells.append(
                {
                    "Actual": actual,
                    "Predicted": predicted,
                    "Students": int(counts[i, j]),
                    "Share of actual class": counts[i, j] / row_total if row_total else 0.0,
                }
            )
    cells = pd.DataFrame(cells)

    base = alt.Chart(cells).encode(
        x=alt.X(
            "Predicted:N",
            sort=CLASS_LABELS,
            scale=alt.Scale(paddingInner=0.04),
            axis=alt.Axis(labelAngle=0, orient="top", title="Predicted"),
        ),
        y=alt.Y("Actual:N", sort=CLASS_LABELS, scale=alt.Scale(paddingInner=0.04), title="Actual"),
    )
    squares = base.mark_rect(cornerRadius=4).encode(
        color=alt.Color(
            "Share of actual class:Q",
            scale=alt.Scale(domain=[0, 1], range=BLUE_STEPS),
            legend=alt.Legend(format="%", title="Share of row"),
        ),
        tooltip=[
            "Actual:N",
            "Predicted:N",
            "Students:Q",
            alt.Tooltip("Share of actual class:Q", format=".1%"),
        ],
    )
    numbers = base.mark_text(fontSize=16, fontWeight="bold").encode(
        text="Students:Q",
        color=alt.condition(alt.datum["Share of actual class"] > 0.55, alt.value("white"), alt.value("#0b0b0b")),
    )
    return (squares + numbers).properties(height=320)


def metric_bar_chart(scores, metric):
    data = scores.reset_index()[["Model", metric]]
    low = min(0.0, float(data[metric].min()))
    return (
        alt.Chart(data)
        .mark_bar(color=BAR_BLUE, cornerRadiusEnd=4, size=26)
        .encode(
            x=alt.X(f"{metric}:Q", scale=alt.Scale(domain=[low, 1]), title=metric),
            y=alt.Y("Model:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
            tooltip=["Model:N", alt.Tooltip(f"{metric}:Q", format=".4f")],
        )
        .properties(height=260)
    )


def main():
    """Load the selected dataset and display model results."""
    models = load_models()
    metadata = load_metadata()
    feature_cols = metadata["feature_columns"]

    with st.sidebar:
        st.header("Test data")
        uploaded = st.file_uploader(
            "Upload student data (CSV)",
            type=["csv"],
            help="Include all 36 features. Target is optional for predictions and required for evaluation.",
        )
        st.download_button(
            "Download sample test_data.csv",
            data=TEST_FILE.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
            width="stretch",
        )

        st.header("Classification model")
        chosen_model = st.selectbox("Classifier", list(MODEL_FILES), index=len(MODEL_FILES) - 1)

        with st.expander("Training details"):
            st.markdown(
                f"- Stratified 80/20 split (seed {metadata['random_state']})\n"
                "- One-hot encoding for coded categories, standard scaling for numbers\n"
                "- 5-fold grid search on the training split, scored by macro F1"
            )
            tuned = pd.DataFrame(
                {
                    name: {
                        "Selected parameters": ", ".join(f"{k}={v}" for k, v in info["best_params"].items()),
                        "CV macro-F1": info["cv_f1_macro"],
                    }
                    for name, info in metadata["models"].items()
                }
            ).T
            st.dataframe(tuned, width="stretch")


    st.title("Student Outcome Classification")
    st.caption(
        "Predicts whether a student will **drop out**, still be **enrolled**, or **graduate** at the end "
        "of the normal course duration, using five classifiers trained on the UCI *Predict Students' "
        "Dropout and Academic Success* dataset."
    )

    if metadata["sklearn_version"] != sklearn.__version__:
        st.warning(
            f"Models were saved with scikit-learn {metadata['sklearn_version']} but "
            f"{sklearn.__version__} is installed. Pin the same version in requirements.txt."
        )

    if uploaded is not None:
        source_name, raw = uploaded.name, uploaded.getvalue()
    else:
        source_name, raw = TEST_FILE.name, TEST_FILE.read_bytes()
        st.info("Using the bundled test set: **test_data.csv** (20% of the dataset).")

    try:
        students = read_student_csv(raw, feature_cols)
    except Exception as err:
        st.error(f"Could not read {source_name} as CSV: {err}")
        st.stop()

    missing = [c for c in feature_cols if c not in students.columns]
    if missing:
        st.error(
            f"{source_name} is missing {len(missing)} required column(s): "
            + ", ".join(missing[:8])
            + (" ..." if len(missing) > 8 else "")
        )
        st.stop()

    X = students[feature_cols]
    has_target = TARGET_COL in students.columns
    y = None
    if has_target:
        y = students[TARGET_COL].astype(str).str.strip().str.capitalize()
        unknown = sorted(set(y) - set(CLASS_LABELS))
        if unknown:
            st.error(f"Target has unexpected values {unknown}. Expected {CLASS_LABELS}.")
            st.stop()

    all_scores, all_outputs = evaluate_every_model(models, X, y)
    y_pred, proba = all_outputs[chosen_model]

    tab_model, tab_compare, tab_data = st.tabs(["Selected model", "Compare all 5 models", "Uploaded data"])

    with tab_model:
        st.subheader(f"{chosen_model} on {source_name} ({len(X)} students)")
        if not has_target:
            st.info("This file has no Target column, so only predictions are shown (metrics need true labels).")
        else:
            scores = all_scores.loc[chosen_model]
            tile_rows = [METRIC_NAMES[:3], METRIC_NAMES[3:]]
            for names in tile_rows:
                for column, metric in zip(st.columns(3), names):
                    value = scores[metric]
                    column.metric(metric, "n/a" if np.isnan(value) else f"{value:.4f}", help=METRIC_HELP[metric])
            f1_rank = int(all_scores["F1"].rank(ascending=False, method="min")[chosen_model])
            st.caption(f"Macro F1 rank on this dataset: {f1_rank} of 5.")

            left, right = st.columns([1.1, 1])
            with left:
                st.markdown("**Confusion matrix**")
                st.altair_chart(confusion_matrix_chart(y, y_pred), width="stretch")
                st.caption("Numbers are student counts; shading is the share of each actual class (row).")
            with right:
                st.markdown("**Classification report**")
                report = classification_report(
                    y, y_pred, labels=CLASS_LABELS, output_dict=True, zero_division=0
                )
                report.pop("accuracy", None)
                report = pd.DataFrame(report).T
                report["support"] = report["support"].astype(int)
                st.dataframe(report.style.format({c: "{:.3f}" for c in ["precision", "recall", "f1-score"]}),
                             width="stretch")

        predictions = students.copy()
        predictions["Predicted"] = y_pred
        for idx, label in enumerate(CLASS_LABELS):
            predictions[f"P({label})"] = proba[:, idx].round(4)
        st.download_button(
            "Download predictions as CSV",
            predictions.to_csv(index=False).encode(),
            file_name=f"predictions_{MODEL_FILES[chosen_model].split('.')[0]}.csv",
            mime="text/csv",
        )

    with tab_compare:
        if not has_target:
            st.info("Upload a file with a Target column to compare the models.")
        else:
            st.subheader("All five models on the same file")
            st.dataframe(
                all_scores.style.format("{:.4f}").highlight_max(axis=0, props="font-weight: bold; background-color: rgba(42,120,214,0.18)"),
                width="stretch",
            )
            st.caption("Best value in each column is highlighted.")
            metric_to_plot = st.radio("Metric to chart", METRIC_NAMES, horizontal=True, index=4)
            st.altair_chart(metric_bar_chart(all_scores, metric_to_plot), width="stretch")

    with tab_data:
        st.subheader(source_name)
        st.write(f"{len(students)} rows × {students.shape[1]} columns")
        if has_target:
            counts = y.value_counts().reindex(CLASS_LABELS, fill_value=0)
            st.dataframe(
                pd.DataFrame({"Students": counts, "Share": (counts / counts.sum()).map("{:.1%}".format)}),
                width="content",
            )
        st.dataframe(students.head(50), width="stretch")


if __name__ == "__main__":
    main()
