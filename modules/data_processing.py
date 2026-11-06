"""
Data processing utilities for the Credit Risk project.

Contents
--------
- audit_constraints : count rows that violate business / logical constraints
- clean_raw_data    : remove duplicates + invalid rows (logs every step)
- split_data        : stratified train/test split
- build_preprocessing_pipeline : configurable sklearn Pipeline
- get_feature_names : feature names after the (optional) PCA step
- save_features     : save X/y/.npy, feature names, config, fitted pipeline
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

LOAN_GRADE_ORDER = ["A", "B", "C", "D", "E", "F", "G"]


# ---------------------------------------------------------------------------
# 1. Audit & cleaning
# ---------------------------------------------------------------------------
def audit_constraints(
    df: pd.DataFrame,
    max_age: int = 100,
    max_emp_length: float = 60,
    min_working_age: int = 14,
    min_credit_age: int = 18,
    income_threshold: float = 1_000_000,
    loan_percent_tol: float = 0.01,
) -> pd.Series:
    """Count rows violating each constraint (rows can violate several at once).

    Use this BEFORE deciding which rules to apply in `clean_raw_data`.
    """
    loan_pct_calc = df["loan_amnt"] / df["person_income"]
    checks = {
        f"person_age > {max_age}": df["person_age"] > max_age,
        f"person_emp_length > {max_emp_length}": df["person_emp_length"] > max_emp_length,
        f"person_emp_length > age - {min_working_age}": (
            df["person_emp_length"] > df["person_age"] - min_working_age
        ),
        f"cb_person_cred_hist_length > age - {min_credit_age}": (
            df["cb_person_cred_hist_length"] > df["person_age"] - min_credit_age
        ),
        f"|loan_amnt/income - loan_percent_income| > {loan_percent_tol}": (
            (loan_pct_calc - df["loan_percent_income"]).abs() > loan_percent_tol
        ),
        f"person_income > {income_threshold:,.0f}": df["person_income"] > income_threshold,
        "duplicated rows": df.duplicated(),
    }
    return pd.Series({name: int(mask.sum()) for name, mask in checks.items()})


def clean_raw_data(
    df: pd.DataFrame,
    verbose: bool = True,
    max_age: int = 100,
    max_emp_length: float = 60,
    min_working_age: int = 14,
    check_cred_hist: bool = False,
    min_credit_age: int = 18,
    check_loan_percent: bool = False,
    loan_percent_tol: float = 0.01,
    max_income: float = None,
) -> pd.DataFrame:
    """Clean raw data. Only fixed (rule-based) filters are used -> no leakage.

    Default behaviour (all optional rules OFF) reproduces the original result:
      1. drop exact duplicates
      2. drop person_age > max_age
      3. drop person_emp_length > max_emp_length or > age - min_working_age
         (NaN emp_length is kept: it is imputed later, inside the pipeline)

    Optional rules (turn on after looking at `audit_constraints`):
      - check_cred_hist    : drop cred_hist_length > age - min_credit_age
      - check_loan_percent : drop rows where loan_percent_income disagrees with
                             loan_amnt / person_income by more than the tolerance
      - max_income         : drop person_income > max_income (use with care,
                             high incomes can be real; prefer log/Robust scaling)
    """
    d = df.copy()
    n_start = len(d)

    def step(name: str, mask=None, drop_dupes: bool = False):
        nonlocal d
        before = len(d)
        d = d.drop_duplicates() if drop_dupes else d[mask]
        if verbose:
            print(f"-> {name:<45} removed {before - len(d):>5} | remaining {len(d):>6}")

    step("duplicate rows", drop_dupes=True)
    step(f"person_age > {max_age}", d["person_age"] <= max_age)
    step(
        f"emp_length > {max_emp_length} or > age - {min_working_age}",
        d["person_emp_length"].isna()
        | (
            (d["person_emp_length"] <= max_emp_length)
            & (d["person_emp_length"] <= d["person_age"] - min_working_age)
        ),
    )
    if check_cred_hist:
        step(
            f"cred_hist_length > age - {min_credit_age}",
            d["cb_person_cred_hist_length"] <= d["person_age"] - min_credit_age,
        )
    if check_loan_percent:
        diff = (d["loan_amnt"] / d["person_income"] - d["loan_percent_income"]).abs()
        step(f"loan_percent_income mismatch (tol={loan_percent_tol})", diff <= loan_percent_tol)
    if max_income is not None:
        step(f"person_income > {max_income:,.0f}", d["person_income"] <= max_income)

    if verbose:
        print(f"==> Total removed: {n_start - len(d)} | final samples: {len(d)}")
    return d.reset_index(drop=True)


def split_data(df: pd.DataFrame, target_col: str = "loan_status",
               test_size: float = 0.2, random_state: int = 42):
    """Stratified split. Must be called BEFORE fitting any imputer/scaler."""
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# ---------------------------------------------------------------------------
# 2. Preprocessing pipeline
# ---------------------------------------------------------------------------
def _make_scaler(scaler_type: str, feature_range: tuple):
    if scaler_type == "standard":
        return StandardScaler()
    if scaler_type == "minmax":
        return MinMaxScaler(feature_range=feature_range)
    if scaler_type == "robust":
        return RobustScaler()
    raise ValueError("scaler_type must be 'standard', 'minmax' or 'robust'")


def _make_imputer(strategy: str, add_indicator: bool, knn_neighbors: int):
    if strategy in ("median", "mean", "most_frequent"):
        return SimpleImputer(strategy=strategy, add_indicator=add_indicator)
    if strategy == "knn":
        return KNNImputer(n_neighbors=knn_neighbors, add_indicator=add_indicator)
    raise ValueError("impute_strategy must be 'median', 'mean', 'most_frequent' or 'knn'")


def build_preprocessing_pipeline(
    num_cols: list,
    ordinal_cols: list,
    nominal_cols: list,
    impute_strategy: str = "median",
    scaler_type: str = "standard",
    feature_range: tuple = (0, 1),
    pca_variance: float = None,
    log_cols: list = None,
    add_missing_indicator: bool = False,
    knn_neighbors: int = 5,
    ordinal_categories: list = None,
) -> Pipeline:
    """Build a configurable preprocessing Pipeline (fit on TRAIN only).

    Parameters
    ----------
    impute_strategy : 'median' | 'mean' | 'most_frequent' | 'knn'  (numeric columns)
    scaler_type     : 'standard' | 'minmax' | 'robust'
    feature_range   : only used by MinMaxScaler
    pca_variance    : None or float in (0, 1], e.g. 0.90 / 0.95
    log_cols        : subset of num_cols to transform with log1p BEFORE scaling
                      (e.g. ['person_income']) to reduce right skew
    add_missing_indicator : add a 0/1 column for each numeric column that has
                      missing values in the training set
    ordinal_categories    : order of the ordinal column(s); default A..G (loan_grade)
    """
    log_cols = list(log_cols or [])
    unknown = set(log_cols) - set(num_cols)
    if unknown:
        raise ValueError(f"log_cols must be a subset of num_cols, got extra: {unknown}")
    plain_cols = [c for c in num_cols if c not in log_cols]

    def numeric_pipe(use_log: bool) -> Pipeline:
        steps = [("imputer", _make_imputer(impute_strategy, add_missing_indicator, knn_neighbors))]
        if use_log:
            steps.append(("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
        steps.append(("scaler", _make_scaler(scaler_type, feature_range)))
        return Pipeline(steps)

    ord_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[ordinal_categories or LOAN_GRADE_ORDER] * len(ordinal_cols),
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    nom_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
    ])

    transformers = []
    if plain_cols:
        transformers.append(("num", numeric_pipe(False), plain_cols))
    if log_cols:
        transformers.append(("numlog", numeric_pipe(True), log_cols))
    transformers.append(("ord", ord_pipeline, ordinal_cols))
    transformers.append(("nom", nom_pipeline, nominal_cols))

    steps = [("preprocessor", ColumnTransformer(transformers=transformers))]
    if pca_variance is not None:
        steps.append(("pca", PCA(n_components=pca_variance, random_state=42)))
    return Pipeline(steps=steps)


def get_feature_names(pipeline: Pipeline) -> list:
    """Feature names of the transformed output (PC1..PCk if PCA is used)."""
    if "pca" in pipeline.named_steps:
        k = pipeline.named_steps["pca"].n_components_
        return [f"PC{i + 1}" for i in range(k)]
    return pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()


# ---------------------------------------------------------------------------
# 3. Saving
# ---------------------------------------------------------------------------
def save_features(name: str, pipeline: Pipeline, X_train_t, X_test_t,
                  y_train, y_test, config: dict, out_dir: str = "../features",
                  save_pipeline: bool = False) -> None:
    """Save arrays and metadata for one configuration.

    Files: X_train_<name>.npy, X_test_<name>.npy, feature_names_<name>.json,
    config_<name>.json (+ preprocessor_<name>.joblib if save_pipeline=True).
    y is shared by all configs (same split): y_train.npy / y_test.npy.
    Re-run the notebook to rebuild the .joblib file: it depends on the
    scikit-learn version.
    """
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/X_train_{name}.npy", X_train_t)
    np.save(f"{out_dir}/X_test_{name}.npy", X_test_t)
    np.save(f"{out_dir}/y_train.npy", y_train)
    np.save(f"{out_dir}/y_test.npy", y_test)
    with open(f"{out_dir}/feature_names_{name}.json", "w", encoding="utf-8") as f:
        json.dump(get_feature_names(pipeline), f, indent=4)
    with open(f"{out_dir}/config_{name}.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, default=str)
    if save_pipeline:
        joblib.dump(pipeline, f"{out_dir}/preprocessor_{name}.joblib")