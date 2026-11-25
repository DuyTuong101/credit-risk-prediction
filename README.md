# Credit Risk Prediction Using Machine Learning

A machine learning project that predicts whether a loan applicant is likely to default based on applicant, loan, and credit-history information. The project was developed for the Machine Learning course at Ho Chi Minh City University of Technology (HCMUT), Vietnam National University Ho Chi Minh City (VNU-HCM).

## Course Information

| Item | Details |
|---|---|
| Course | Machine Learning |
| Course code | CO3117 |
| Semester | Semester I |
| Academic year | 2026–2027 |
| University | Ho Chi Minh City University of Technology (HCMUT), VNU-HCM |
| Advisor / Instructor | ThD Truong Vinh Lan |

## Assignment Objectives

The project aims to apply a traditional machine learning pipeline to a real-world tabular dataset. The main objectives are to:

- Perform exploratory data analysis (EDA) to understand the dataset, feature distributions, missing values, outliers, and target-class distribution.
- Build a reproducible data preprocessing and feature-generation workflow.
- Train and compare multiple traditional classification algorithms.
- Experiment with preprocessing configurations and model hyperparameters.
- Evaluate model performance using appropriate classification metrics and analyze the results.

## 1. Project Overview

The goal is to build and evaluate classification models for predicting loan default. The target variable is `loan_status`:

- `0` — Non-default
- `1` — Default

The workflow covers exploratory data analysis (EDA), data cleaning, preprocessing, feature generation, model training, experiments, and final evaluation.

## 2. Dataset

The project uses `credit_risk_dataset.csv`. The dataset contains applicant and loan-related attributes, including:

| Feature | Description |
|---|---|
| `person_age` | Applicant's age |
| `person_income` | Applicant's annual income |
| `person_home_ownership` | Home ownership status |
| `person_emp_length` | Employment length |
| `loan_intent` | Purpose of the loan |
| `loan_grade` | Loan grade |
| `loan_amnt` | Loan amount |
| `loan_int_rate` | Loan interest rate |
| `loan_percent_income` | Loan amount as a proportion of income |
| `cb_person_default_on_file` | Historical default indicator |
| `cb_person_cred_hist_length` | Length of credit history |
| `loan_status` | Target variable: default status |

The dataset is publicly available on Kaggle: [Credit Risk Dataset – laotse](https://www.kaggle.com/datasets/laotse/credit-risk-dataset). The raw dataset used by the project is stored at `data/raw/credit_risk_dataset.csv`. The cleaned dataset is stored at `data/processed/credit_risk_cleaned.csv`.

For the Google Colab workflow, use the linked [master notebook](https://colab.research.google.com/github/DuyTuong101/credit-risk-prediction/blob/main/notebooks/master_notebook.ipynb). Follow its data preparation cells to make the dataset available in the Colab runtime.

## 3. Repository Structure

```text
credit-risk-prediction/
├── data/
│   ├── raw/
│   │   └── credit_risk_dataset.csv
│   └── processed/
│       └── credit_risk_cleaned.csv
├── features/
│   ├── config_*.json
│   ├── feature_manifest.json
│   ├── feature_names_*.json
│   ├── preprocessor_*.joblib
│   ├── train_indices.npy
│   ├── test_indices.npy
│   ├── X_train_*.npy
│   ├── X_test_*.npy
│   ├── y_train.npy
│   └── y_test.npy
├── models/
│   ├── best_model.joblib
│   ├── experiments/
│   └── hyperparameters/
├── modules/
│   ├── data_processing.py
│   └── model_training.py
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   └── master_notebook.ipynb
├── reports/
│   ├── figures/
│   ├── experiments/
│   ├── hyperparameters/
│   ├── Credit_Risk_Prediction_HCMUT_Report.pdf  # Add the final PDF report here
│   ├── classification_report.json
│   ├── confusion_matrix.csv
│   ├── model_comparison.csv
│   ├── preprocessing_experiments.csv
│   ├── hyperparameter_experiments.csv
│   └── test_metrics.json
├── .gitignore
├── README.md
└── requirements.txt
```

## 4. Methodology

### 4.1 Exploratory Data Analysis

EDA is used to understand the dataset before modeling. It includes:

- Dataset structure, descriptive statistics, missing values, and duplicate records.
- Distribution of the target variable (`loan_status`).
- Detection and inspection of invalid values and outliers.
- Analysis of categorical features by target class.
- Correlation analysis of numerical features.

EDA figures are saved in `reports/figures/`.

### 4.2 Data Preprocessing and Feature Generation

The preprocessing workflow includes:

1. Loading and cleaning the raw dataset.
2. Checking duplicate records and invalid values.
3. Splitting the data into training and test sets.
4. Imputing missing numerical and categorical values.
5. Encoding categorical variables.
6. Scaling numerical features where appropriate.
7. Generating and saving feature sets and preprocessing configurations.

**Data leakage prevention:** the train/test split is performed before fitting preprocessing transformations. Imputers, encoders, scalers, and other learned transformations are fitted using training data only, then applied to the test data.

The `features/` directory stores generated feature arrays, feature names, split indices, preprocessing configurations, and the feature manifest.

### 4.3 Models

The project investigates the following classification algorithms:

- Logistic Regression
- Linear Support Vector Machine (Linear SVM)
- Random Forest

### 4.4 Experiments

Experiments examine preprocessing choices and model hyperparameters. The repository stores experiment-level model artifacts and evaluation outputs under:

- `models/experiments/` and `reports/experiments/`
- `models/hyperparameters/` and `reports/hyperparameters/`

The corresponding summary tables are `reports/preprocessing_experiments.csv` and `reports/hyperparameter_experiments.csv`.

## 5. Final Model Results

The saved final evaluation identifies **Random Forest** as the selected model. Its performance on the held-out test set is:

| Metric | Score |
|---|---:|
| Accuracy | 93.44% |
| Precision | 95.26% |
| Recall | 73.70% |
| F1-score | 83.10% |
| ROC-AUC | 93.55% |

The test set contains 6,482 samples. These values are taken from `reports/test_metrics.json`. Additional details are available in `reports/classification_report.json` and `reports/confusion_matrix.csv`.

## 6. Getting Started

### Requirements

- Python 3.10 or later
- pip
- Jupyter Notebook or JupyterLab
- Python packages listed in `requirements.txt`

### Run on Google Colab

The submitted notebook is available here:

[Open Master Notebook in Google Colab](https://colab.research.google.com/github/DuyTuong101/credit-risk-prediction/blob/main/notebooks/master_notebook.ipynb)

Open the notebook and select **Runtime → Run all** to execute the integrated workflow. The notebook is stored in the GitHub repository, so it can be opened directly without manually uploading the notebook file.

### Clone the repository

```bash
git clone https://github.com/DuyTuong101/credit-risk-prediction.git
cd credit-risk-prediction
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the complete workflow

The recommended entry point is the master notebook, which brings together EDA, preprocessing, model training, experiments, evaluation, and output verification:

```bash
jupyter notebook notebooks/master_notebook.ipynb
```

Open `master_notebook.ipynb` and run the cells in order (`Kernel` / `Run All`, or the equivalent command in your notebook interface). The notebook is also organized for execution in Google Colab.

### Run individual analysis notebooks

The EDA and preprocessing stages can also be explored separately:

```bash
jupyter notebook notebooks/01_EDA.ipynb
jupyter notebook notebooks/02_Data_Preprocessing.ipynb
```

Run the master notebook for the integrated end-to-end workflow rather than treating these separate notebooks as independent final pipelines.

## 7. Team

This is a group project for the Machine Learning course at Ho Chi Minh City University of Technology (HCMUT).

| Member | Student ID | Email | Main Responsibilities |
|---|---:|---|---|
| Do Viet Duy Tuong | 2353302 | tuong.do10012005@hcmut.edu.vn | Project integration, pipeline and feature extraction, experiments and analysis, report preparation, and repository organization |
| Ngo Gia Bao | 2352094 | bao.ngogiabao1608@hcmut.edu.vn | Exploratory data analysis and data processing |
| Tran Doan Binh An | 2352017 | **To be added** | Model training and evaluation |

**Advisor:** Truong Vinh Lan

## 8. Project Report

The project report is titled **Credit Risk Prediction Using Machine Learning**. It documents the project motivation, dataset analysis, methodology, experimental setup, results, and conclusion.

After adding the PDF to the `reports/` directory, it can be accessed directly from the repository:

[Download the Project Report (PDF)](reports/Credit_Risk_Prediction_HCMUT_Report.pdf)

## 9. Links

- **GitHub Repository:** [DuyTuong101/credit-risk-prediction](https://github.com/DuyTuong101/credit-risk-prediction)
- **Google Colab Notebook:** [Open master notebook](https://colab.research.google.com/github/DuyTuong101/credit-risk-prediction/blob/main/notebooks/master_notebook.ipynb)
- **Dataset:** [Credit Risk Dataset on Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)

## 10. Notes

- `models/best_model.joblib` contains the saved final model pipeline.
- The `reports/` directory contains final metrics, classification results, confusion matrix, and experiment summaries.
- The `features/` directory contains generated intermediate artifacts used by the modeling workflow.
- Model and preprocessing results should be interpreted together with the evaluation reports, especially because the target classes are imbalanced.
