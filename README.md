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
│
├── modules/
│   └── data_processing.py
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_Model_Evaluation.ipynb
│
├── reports/
│   └── figures/
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

Open the notebooks in the `notebooks/` directory to explore the dataset, preprocess the data, train models, and evaluate their performance.

## 9. Team Project

This is a group project developed as part of the Machine Learning course at Ho Chi Minh City University of Technology (HCMUT).

The team collaborates on data exploration, preprocessing, model development, evaluation, and result analysis.

## 10. Project Status

**Current stage:** Data preprocessing and project organization.

Further work includes exploratory data analysis, model training, performance evaluation, and comparison of classification algorithms.
