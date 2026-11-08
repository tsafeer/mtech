from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

# The target is 0 = malignant, 1 = benign.
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target

print(f"Dataset rows: {len(X)}")
print(f"Feature count: {X.shape[1]}")
print(f"Classes: {dict(enumerate(data.target_names))}")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42),
    ),
    "Decision Tree": make_pipeline(
        DecisionTreeClassifier(random_state=42),
    ),
    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5),
    ),
    "Naive Bayes": make_pipeline(
        StandardScaler(),
        GaussianNB(),
    ),
    "Support Vector Machine": make_pipeline(
        StandardScaler(),
        SVC(probability=True, random_state=42),
    ),
}

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

rows = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        # Class 1 is benign in this dataset.
        "AUC": roc_auc_score(y_test, probabilities[:, 1]),
        "Precision": precision_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "MCC": matthews_corrcoef(y_test, predictions),
    }
    rows.append(metrics)

    joblib.dump(model, MODEL_DIR / model_files[name], compress=3)
    print(f"\n{name}")
    for metric, value in metrics.items():
        if metric != "Model":
            print(f"  {metric}: {value:.4f}")

# The uploaded test CSV must contain the features and the target.
test_data = X_test.copy()
test_data["target"] = y_test
test_data.to_csv("test_data.csv", index=False)

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv("metrics.csv", index=False)

print("\nSaved five models in model/")
print("Saved test_data.csv and metrics.csv")
print(metrics_df.to_string(index=False))