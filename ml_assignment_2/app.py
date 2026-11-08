from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}

st.set_page_config(page_title="ML Model Evaluation", layout="wide")
st.title("Classification Model Evaluation")
st.write(
    "Upload a labeled test CSV containing all model features and a `target` "
    "column. The app evaluates a saved model; it does not retrain."
)

model_name = st.selectbox("Select a model", list(MODEL_FILES))
uploaded_file = st.file_uploader("Upload test data CSV", type=["csv"])

if uploaded_file is None:
    st.info("Upload a test CSV to see evaluation results.")
    st.stop()

model_path = Path(MODEL_FILES[model_name])
if not model_path.is_file():
    st.error(f"Saved model not found: {model_path}. Run train_models.py first.")
    st.stop()

try:
    test_data = pd.read_csv(BytesIO(uploaded_file.getvalue()))
except (pd.errors.ParserError, UnicodeDecodeError) as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

if "target" not in test_data.columns:
    st.error("The uploaded CSV must include a `target` column.")
    st.stop()

model = joblib.load(model_path)
expected_features = list(model.feature_names_in_)
uploaded_features = [column for column in test_data.columns if column != "target"]

missing = [column for column in expected_features if column not in uploaded_features]
unexpected = [column for column in uploaded_features if column not in expected_features]

if missing or unexpected:
    st.error(
        "The CSV feature columns do not match the saved model. "
        f"Missing: {missing or 'none'}. "
        f"Unexpected: {unexpected or 'none'}."
    )
    st.stop()

X_test = test_data.reindex(columns=expected_features).apply(
    pd.to_numeric, errors="coerce"
)
if X_test.isna().any().any():
    st.error("Feature columns must contain numeric values with no missing entries.")
    st.stop()

if not np.isfinite(X_test.to_numpy()).all():
    st.error("Feature columns must not contain infinite values.")
    st.stop()

y_true = pd.to_numeric(test_data["target"], errors="coerce")
if y_true.isna().any():
    st.error("The target column must contain numeric class labels.")
    st.stop()

classes = model.classes_
if set(y_true.unique()) != set(classes):
    st.error(
        "The uploaded test CSV must include at least one row from every "
        f"model class: {list(classes)}."
    )
    st.stop()

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

if len(classes) == 2:
    positive_class_index = list(classes).index(1)
    auc = roc_auc_score(y_true, probabilities[:, positive_class_index])
else:
    auc = roc_auc_score(
        y_true,
        probabilities,
        labels=classes,
        multi_class="ovr",
        average="macro",
    )

results = {
    "Accuracy": accuracy_score(y_true, predictions),
    "AUC": auc,
    "Precision": precision_score(
        y_true, predictions, average="macro", zero_division=0
    ),
    "Recall": recall_score(
        y_true, predictions, average="macro", zero_division=0
    ),
    "F1": f1_score(
        y_true, predictions, average="macro", zero_division=0
    ),
    "MCC": matthews_corrcoef(y_true, predictions),
}

st.subheader(f"Results: {model_name}")
metric_columns = st.columns(3)
for index, (metric_name, value) in enumerate(results.items()):
    metric_columns[index % 3].metric(metric_name, f"{value:.4f}")

st.subheader("Confusion matrix")
matrix = confusion_matrix(y_true, predictions, labels=classes)
st.dataframe(
    pd.DataFrame(
        matrix,
        index=[f"Actual {label}" for label in classes],
        columns=[f"Predicted {label}" for label in classes],
    ),
    use_container_width=True,
)

st.subheader("Classification report")
report = classification_report(
    y_true,
    predictions,
    labels=classes,
    output_dict=True,
    zero_division=0,
)
st.dataframe(pd.DataFrame(report).T, use_container_width=True)