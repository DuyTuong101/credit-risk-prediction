# Credit Risk Prediction

## 1. Project Overview

This project focuses on **Credit Risk Classification** using Machine Learning techniques.

The main objective is to predict whether a loan applicant is likely to default based on their personal, financial, and credit-related information.

The project uses the `credit_risk_dataset.csv` dataset, with `loan_status` as the target variable.

- `loan_status = 0`: Non-default
- `loan_status = 1`: Default

## 2. Project Objectives

- Explore and understand the credit risk dataset.
- Perform data cleaning and preprocessing.
- Handle missing values and invalid data.
- Encode categorical features and scale numerical features.
- Train and compare different Machine Learning classification models.
- Evaluate model performance using appropriate classification metrics.

## 3. Repository Structure

```text
credit-risk-prediction/
│
├── data/
│   ├── raw/
│   │   └── credit_risk_dataset.csv
│   └── processed/
│
├── features/
│   ├── X_train_baseline.npy
│   ├── X_test_baseline.npy
│   ├── y_train_baseline.npy
│   ├── y_test_baseline.npy
│   └── feature_names_baseline.json
│
├── models/
│   └── best_model.joblib
├── modules/
│   ├── data_processing.py
│   └── model_training.py
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_Model_Evaluation.ipynb
│
├── reports/
│   ├── figures/
│   ├── model_comparison.csv
│   ├── test_metrics.json
│   ├── classification_report.json
│   └── confusion_matrix.csv
│
├── .gitignore
├── README.md
└── requirements.txt
```

## 4. Dataset Description

The dataset contains information about loan applicants, including:

| Feature | Description |
|---|---|
| `person_age` | Age of the applicant |
| `person_income` | Annual income |
| `person_home_ownership` | Home ownership status |
| `person_emp_length` | Employment length |
| `loan_intent` | Purpose of the loan |
| `loan_grade` | Loan grade |
| `loan_amnt` | Loan amount |
| `loan_int_rate` | Loan interest rate |
| `loan_percent_income` | Loan amount as a percentage of income |
| `cb_person_default_on_file` | Historical default indicator |
| `cb_person_cred_hist_length` | Credit history length |
| `loan_status` | Target variable |

## 5. Data Processing Pipeline

The project follows these main preprocessing steps:

1. Load the original dataset.
2. Remove duplicate records.
3. Identify and handle invalid values.
4. Split the dataset into training and testing sets.
5. Handle missing values using imputation techniques.
6. Encode categorical variables.
7. Scale numerical features.
8. Save the processed features for model training.

**Data Leakage Prevention:** Statistical preprocessing operations are fitted only on the training set through Scikit-learn pipelines. The test set is kept separate for final evaluation.

## 6. Machine Learning Models

The project will investigate and compare classification algorithms, including:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

The final model selection will be based on experimental results.

## 7. Model Evaluation

The models will be evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

Special attention will be paid to Recall and F1-score because the dataset contains an imbalanced target distribution.

## 8. Installation and Setup

### Requirements

- Python 3.10 or later
- pip
- Jupyter Notebook

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run Jupyter Notebook

```bash
jupyter notebook
```

Open the notebooks in the `notebooks/` directory to explore and preprocess the data. Train and compare Logistic Regression, SVM, and Random Forest from the repository root with:

```bash
python -m modules.model_training
```

The script loads `data/processed/credit_risk_cleaned.csv`, creates a stratified train/test split, and fits imputation, scaling, encoding, and optional PCA within each model pipeline. Each candidate is compared using stratified cross-validation on the training split; the highest mean-F1 model is refitted on all training data and evaluated once on the held-out test split. The fitted preprocessing-and-model pipeline is saved as `models/best_model.joblib`; cross-validation scores and held-out metrics/reports are saved under `reports/`.

Pipeline choices and estimator hyperparameters can be set on the command line:

```bash
# Standard scaling and PCA retaining 95% of variance
python -m modules.model_training --scaler standard --pca-variance 0.95

# MinMax scaling into [-1, 1], PCA retaining 90%, and selected models
python -m modules.model_training --scaler minmax --feature-range -1 1 --pca-variance 0.90 --models logistic_regression svm random_forest

# Tune model parameters
python -m modules.model_training --logistic-c 0.5 --svm-c 2 --svm-kernel linear --rf-n-estimators 300 --rf-max-depth 15
```

Omit `--pca-variance` to disable dimensionality reduction. Available options, including imputation, split size, fold count, and output paths, are listed by `python -m modules.model_training --help`. The same settings are available as keyword arguments to `train_and_evaluate` in `modules/model_training.py`.

## 9. Team Project

This is a group project developed as part of the Machine Learning course at Ho Chi Minh City University of Technology (HCMUT).

The team collaborates on data exploration, preprocessing, model development, evaluation, and result analysis.

## 10. Project Status

**Current stage:** Data preprocessing and traditional model training.

Further work includes reviewing model results and selecting a final model based on the project's credit-risk objectives.
