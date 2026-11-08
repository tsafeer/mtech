# Machine Learning Assignment 2

This project compares five classification models on one dataset and provides a Streamlit application for evaluating saved models on an uploaded, labeled test CSV.
The app should load the trained models; it should not train or refit them when a user uploads test data.

## Problem statement

The goal is to compare five supervised classification algorithms on the same dataset, using a consistent train/test split and a common set of evaluation metrics.
The saved models are then made available through an interactive Streamlit app.

**Chosen classification problem:** 

This is a supervised binary classification task using the Breast cance Wisconsin dataset. The modelpredicts whether the recorded diagnosis is maligant or benign. 

## Dataset

- **Dataset:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- **Dataset access in this project:** `sklearn.datasets.load_breast_cancer`
- **Rows:** 569
- **Input features:** 30 numeric features
- **Target:** diagnosis class; in scikit-learn's version, `0` is malignant and
  `1` is benign.



## Models and evaluation

The five models required by the assignment are:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Support Vector Machine (SVM)

Evaluate every model on the same held-out test data. Report the following
metrics, using the same averaging convention for each model:

- Accuracy
- AUC (ROC AUC)
- Precision
- Recall
- F1 score
- Matthews Correlation Coefficient (MCC)

### Results

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9812 | 0.9812 | 0.9812 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9019 | 0.9157 | 0.9075 | 0.8174 |
| KNN | 0.9561 | 0.9788 | 0.9551 | 0.9504 | 0.9526 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9246 | 0.9246 | 0.9246 | 0.8492 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9812 | 0.9812 | 0.9812 | 0.9623 |


### Performance observations

WLogistic Regression and SVM performed best overall, tying on accuracy (98.25%), precision, recall, F1, and MCC. Logistic Regression had a slightly higher AUC (0.9954 vs. 0.9950). KNN also performed well, while Naive Bayes had a high AUC but lower scores on the other metrics. The Decision Tree had the lowest scores overall. Based on this test split, Logistic Regression is the strongest overall model, although results may vary with a different data split.

- **Logistic Regression:** Achieved 98.25% accuracy and the highest AUC (0.9954). Its macro precision, recall, and F1 were all 0.9812, with an MCC of 0.9623, indicating strong and balanced performance on this test split.
- **Decision Tree:** Had the lowest accuracy (91.23%), AUC (0.9157), macro F1 (0.9075), and MCC (0.8174) among the five models. Its macro recall (0.9157) was higher than its macro precision (0.9019).
- **KNN:** Reached 95.61% accuracy, 0.9788 AUC, and 0.9526 macro F1. Its MCC was 0.9054, placing its overall results below Logistic Regression and SVM but above the Decision Tree and Naive Bayes.
- **Naive Bayes:** Recorded 92.98% accuracy and 0.9246 macro F1. Its AUC (0.9868) was higher than KNN's, despite lower accuracy, macro F1, and MCC (0.8492), showing that the metrics rank these models differently.
- **Support Vector Machine:** Tied Logistic Regression on accuracy (98.25%), macro precision, recall, and F1 (0.9812), and MCC (0.9623). Its AUC (0.9950)  was slightly below Logistic Regression's.
- **Overall winner and reason:** Logistic Regression was strongest on this test split: it tied SVM on the main threshold-based metrics and had the highest AUC. This comparison is based on one split; results may change with another split or different model settings.

## Project files

Update this list if your repository uses different file or folder names.

```text
.
├── app.py
├── train-models.py
├── requirements.txt
├── test_data.csv
├── metrics.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── support_vector_machine.joblib
```

`train-models.py` trains and saves the classifiers, creates the labeled
`test_data.csv`, and writes the comparison metrics to `metrics.csv`.
`app.py` loads a selected saved model and evaluates the uploaded test data.

## Setup and run locally

Use Python 3.10 or later. install the listed dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Train the models and generate the test CSV:

```bash
python train_models.py
```

Start the Streamlit app:

```bash
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. Upload `test_data.csv`, select a
model, and check its metrics, confusion matrix, and classification report.

The uploaded CSV must contain every model input feature and the true target
column. Keep the target column name and label encoding consistent with the
training code. The app should evaluate the uploaded rows using the saved model
and must not retrain on them.

