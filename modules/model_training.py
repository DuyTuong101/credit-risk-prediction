"""Train, compare, and evaluate traditional credit-risk classifiers."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "credit_risk_cleaned.csv"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"


def _build_preprocessor(
    X: pd.DataFrame,
    scaler: str,
    feature_range: tuple[float, float],
    numeric_imputation: str,
    categorical_imputation: str,
) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()
    transformers = []

    if numeric_features:
        scaler_step = (
            MinMaxScaler(feature_range=feature_range)
            if scaler == "minmax"
            else StandardScaler()
        )
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy=numeric_imputation)),
                ("scaler", scaler_step),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy=categorical_imputation)),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    if not transformers:
        raise ValueError("The dataset must contain at least one predictor column.")

    return ColumnTransformer(transformers=transformers)


def _build_models(
    random_state: int,
    model_names: tuple[str, ...],
    model_params: dict | None = None,
) -> dict:
    available_models = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        ),
        "svm": SVC(class_weight="balanced", kernel="rbf", random_state=random_state),
        "random_forest": RandomForestClassifier(
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_estimators=150,
            n_jobs=-1,
            random_state=random_state,
        ),
    }
    if not model_names:
        raise ValueError("Select at least one model to train.")
    if len(set(model_names)) != len(model_names):
        raise ValueError("Each model can only be selected once.")
    unknown_models = set(model_names) - set(available_models)
    if unknown_models:
        raise ValueError(
            f"Unknown model(s): {sorted(unknown_models)}. "
            f"Choose from {sorted(available_models)}."
        )

    model_params = model_params or {}
    unknown_parameter_models = set(model_params) - set(available_models)
    if unknown_parameter_models:
        raise ValueError(
            f"Parameters supplied for unknown model(s): "
            f"{sorted(unknown_parameter_models)}."
        )

    models = {}
    for model_name in model_names:
        model = available_models[model_name]
        if model_name in model_params:
            model.set_params(**model_params[model_name])
        models[model_name] = model
    return models


def train_and_evaluate(
    df: pd.DataFrame,
    target_col: str = "loan_status",
    test_size: float = 0.2,
    cv_folds: int = 3,
    random_state: int = 42,
    scaler: str = "standard",
    feature_range: tuple[float, float] = (0, 1),
    pca_variance: float | None = None,
    numeric_imputation: str = "median",
    categorical_imputation: str = "most_frequent",
    model_names: tuple[str, ...] = (
        "logistic_regression",
        "svm",
        "random_forest",
    ),
    model_params: dict | None = None,
    models_dir: Path = DEFAULT_MODELS_DIR,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
) -> dict:
    """Compare configured classifiers with training-only CV and held-out testing."""
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col!r} is missing from the dataset.")
    if df[target_col].isna().any():
        raise ValueError("The target column must not contain missing values.")
    if set(df[target_col].unique()) != {0, 1}:
        raise ValueError("The target column must contain both binary labels 0 and 1.")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")
    if (df[target_col].value_counts() < cv_folds).any():
        raise ValueError("Each target class must have at least cv_folds rows.")
    if scaler not in {"standard", "minmax"}:
        raise ValueError("scaler must be 'standard' or 'minmax'.")
    if (
        len(feature_range) != 2
        or not np.isfinite(feature_range).all()
        or feature_range[0] >= feature_range[1]
    ):
        raise ValueError("feature_range must contain two increasing finite values.")
    if pca_variance is not None and not 0 < pca_variance <= 1:
        raise ValueError("pca_variance must be None or a value in (0, 1].")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if numeric_imputation not in {"mean", "median", "most_frequent"}:
        raise ValueError(
            "numeric_imputation must be 'mean', 'median', or 'most_frequent'."
        )
    if categorical_imputation not in {"most_frequent", "constant"}:
        raise ValueError(
            "categorical_imputation must be 'most_frequent' or 'constant'."
        )

    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = _build_preprocessor(
        X_train,
        scaler=scaler,
        feature_range=feature_range,
        numeric_imputation=numeric_imputation,
        categorical_imputation=categorical_imputation,
    )
    cross_validation = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )
    scoring = {"f1": "f1", "roc_auc": "roc_auc"}
    comparison_rows = []
    model_pipelines = {}

    models = _build_models(random_state, model_names, model_params)
    for model_name, classifier in models.items():
        steps = [("preprocessor", preprocessor)]
        if pca_variance is not None:
            steps.append(
                ("pca", PCA(n_components=pca_variance, random_state=random_state))
            )
        steps.append(("classifier", classifier))
        pipeline = Pipeline(steps)
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cross_validation,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
        comparison_rows.append(
            {
                "model": model_name,
                "cv_f1_mean": float(np.mean(scores["test_f1"])),
                "cv_f1_std": float(np.std(scores["test_f1"])),
                "cv_roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
            }
        )
        model_pipelines[model_name] = pipeline

    comparison = pd.DataFrame(comparison_rows).sort_values(
        "cv_f1_mean", ascending=False
    )
    best_model_name = comparison.iloc[0]["model"]
    best_model = model_pipelines[best_model_name]
    best_model.fit(X_train, y_train)

    predictions = best_model.predict(X_test)
    if hasattr(best_model, "predict_proba"):
        positive_class_index = list(best_model.classes_).index(1)
        positive_scores = best_model.predict_proba(X_test)[:, positive_class_index]
    else:
        positive_class_index = list(best_model.classes_).index(1)
        positive_scores = best_model.decision_function(X_test)
        if positive_class_index == 0:
            positive_scores = -positive_scores
    test_metrics = {
        "selected_model": best_model_name,
        "selection_metric": "mean cross-validation F1",
        "scaler": scaler,
        "feature_range": list(feature_range) if scaler == "minmax" else None,
        "pca_variance": pca_variance,
        "numeric_imputation": numeric_imputation,
        "categorical_imputation": categorical_imputation,
        "models_compared": list(models),
        "model_parameters": {
            name: models[name].get_params(deep=False) for name in models
        },
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, positive_scores)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "cv_folds": cv_folds,
    }

    models_dir = Path(models_dir)
    reports_dir = Path(reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(reports_dir / "model_comparison.csv", index=False)
    with (reports_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(test_metrics, file, indent=2, default=str)
    with (reports_dir / "classification_report.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                output_dict=True,
                zero_division=0,
            ),
            file,
            indent=2,
        )
    pd.DataFrame(
        confusion_matrix(y_test, predictions, labels=[0, 1]),
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    ).to_csv(reports_dir / "confusion_matrix.csv")
    joblib.dump(best_model, models_dir / "best_model.joblib")

    return {
        "comparison": comparison,
        "test_metrics": test_metrics,
        "model_path": models_dir / "best_model.joblib",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare traditional classifiers for credit-risk prediction."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--scaler", choices=("standard", "minmax"), default="standard"
    )
    parser.add_argument(
        "--feature-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0, 1),
        help="MinMaxScaler output range (used only with --scaler minmax).",
    )
    parser.add_argument(
        "--pca-variance",
        type=float,
        default=None,
        metavar="VARIANCE",
        help="Retained PCA variance, e.g. 0.90 or 0.95; omit to disable PCA.",
    )
    parser.add_argument(
        "--numeric-imputation",
        choices=("mean", "median", "most_frequent"),
        default="median",
    )
    parser.add_argument(
        "--categorical-imputation",
        choices=("most_frequent", "constant"),
        default="most_frequent",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("logistic_regression", "svm", "random_forest"),
        default=("logistic_regression", "svm", "random_forest"),
        help="Models to compare.",
    )
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument(
        "--svm-kernel",
        choices=("linear", "poly", "rbf", "sigmoid"),
        default="rbf",
    )
    parser.add_argument("--rf-n-estimators", type=int, default=150)
    parser.add_argument("--rf-max-depth", type=int, default=None)
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    model_params = {
        "logistic_regression": {"C": args.logistic_c},
        "svm": {"C": args.svm_c, "kernel": args.svm_kernel},
        "random_forest": {
            "n_estimators": args.rf_n_estimators,
            "max_depth": args.rf_max_depth,
        },
    }
    results = train_and_evaluate(
        data,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        scaler=args.scaler,
        feature_range=tuple(args.feature_range),
        pca_variance=args.pca_variance,
        numeric_imputation=args.numeric_imputation,
        categorical_imputation=args.categorical_imputation,
        model_names=tuple(args.models),
        model_params=model_params,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
    )
    print("Cross-validation comparison (sorted by mean F1):")
    print(results["comparison"].to_string(index=False))
    print("\nHeld-out test metrics:")
    print(json.dumps(results["test_metrics"], indent=2))
    print(f"\nSaved model: {results['model_path']}")


if __name__ == "__main__":
    main()
