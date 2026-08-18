# ML Assignment 2 – Classification Model Comparison

## 1. Problem Statement

Implement multiple machine-learning classification models on one public classification dataset, evaluate the models using the required metrics, and deploy an interactive Streamlit application. The app allows test-data upload, model selection, metric display, and classification-result visualization.

## 2. Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic)

**Public source:** UCI Machine Learning Repository dataset, loaded through `scikit-learn`.

- Instances: 569
- Features: 30 numerical features
- Classification type: Binary
- Target column in this project: `target`
- Class labels: 0 and 1

This dataset satisfies the assignment minimum of 12 features and 500 instances.

## 3. GitHub Repository Link

`https://github.com/sudarshansomeneni/Bits-ML-Assignment-2`

## 4. Models Used

The assignment states that 6 models are required, but the numbered list explicitly names 5 models. To satisfy the six-model requirement without omitting any named model, this project implements the 5 named models plus SVM as the sixth model:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN) Classifier
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)
6. Support Vector Machine (SVM)

### Evaluation Metrics

Every model is evaluated using:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| KNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest (Ensemble) | 0.9474 | 0.9937 | 0.9583 | 0.9583 | 0.9583 | 0.8869 |
| SVM | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved the highest overall result together with SVM, with 98.25% accuracy, 0.9954 AUC, and 0.9861 F1. |
| Decision Tree | Gave the lowest performance among the six models on this split, although precision remained high. |
| KNN | Performed strongly and achieved 100% recall, with 97.37% accuracy and 0.9796 F1. |
| Naive Bayes | Produced solid results with 93.86% accuracy and high AUC, but was below the leading models on accuracy and F1. |
| Random Forest | Produced a high AUC of 0.9937 and balanced precision/recall, but its accuracy was below Logistic Regression, KNN, and SVM on this test split. |
| SVM | Matched Logistic Regression on accuracy, precision, recall, F1 and MCC, with a very high AUC of 0.9950. |
| Overall Winner | **Logistic Regression**, based on the highest F1/MCC tie and the highest AUC among the tied top-accuracy models in this run. |

> Note: Results depend on the train/test split and model settings. The values above are from the supplied `train.py` run using `random_state=42` and a stratified 80/20 split.

## 5. Project Files

```text
ml_assignment_2/
│
├── app.py
├── train.py
├── generate_test_data.py
├── requirements.txt
├── README.md
├── test_data.csv
├── metrics.csv
├── feature_info.json
├── winner.txt
│
└── models/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── random_forest.joblib
    └── svm.joblib
```

## 6. Installation

```bash
pip install -r requirements.txt
```

## 7. Train the Models

```bash
python train.py
```

This creates/updates the six saved model files, `test_data.csv`, `metrics.csv`, `feature_info.json`, and `winner.txt`.

## 8. Run the Streamlit App

```bash
streamlit run app.py
```

## 9. Streamlit Features

The application includes the assignment requirements:

- Test-data upload using CSV
- Model-selection dropdown
- Evaluation metrics for all implemented models
- Detailed metrics for the selected model
- Confusion matrix
- Classification report
- Model comparison chart
- Downloadable evaluation-results CSV

## 10. Streamlit Community Cloud Deployment link

 `https://bits-ml-assignment-2-54zdgbakprbwbnvxvtjedq.streamlit.app/`

## 11. Final Submission Checklist

- [x] GitHub repository link works.
- [x] Repository contains all source code, `requirements.txt`, README, test CSV, and saved model files.
- [x] Streamlit app link opens an interactive frontend.
- [x] Test CSV upload works.
- [x] Model selection works.
- [x] Metrics are visible.
- [x] Confusion matrix/classification report is visible.
- [x] BITS Virtual Lab execution screenshot is captured.
- [x] GitHub and Streamlit links are placed in the final PDF.
- [x] README content is included in the final PDF.
