"""Model Training, Hyperparameter Tuning, and Evaluation Module.

Implements baseline classifiers, cross-validation scoring prioritizing
fail/at-risk recall, model tuning, pipeline serialization, and test evaluation.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
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
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# Ensure project root is accessible
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import (
    ALL_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    build_preprocessor,
    engineer_features,
    get_feature_target_split,
    split_data,
)
from src.cleaning import load_raw_data, clean_student_data


def get_cv_metrics(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    """Perform Stratified K-Fold CV and calculate classification metrics.
    
    Prioritizes fail/at-risk class (label 0 in 'passed' target).
    
    Args:
        pipeline: Complete Scikit-learn Pipeline (preprocessor + model).
        X_train: Training feature DataFrame.
        y_train: Training target Series (1=Pass, 0=At-Risk).
        cv_splits: Number of cross-validation folds.
        random_state: Seed for fold shuffling.
        
    Returns:
        Dict of mean CV performance metrics.
    """
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    
    recalls_at_risk = []
    precisions_at_risk = []
    f1s_at_risk = []
    accuracies = []
    aucs = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        pipeline.fit(X_tr, y_tr)
        preds = pipeline.predict(X_val)

        # Fail / At-Risk class is label 0
        recalls_at_risk.append(recall_score(y_val, preds, pos_label=0, zero_division=0))
        precisions_at_risk.append(precision_score(y_val, preds, pos_label=0, zero_division=0))
        f1s_at_risk.append(f1_score(y_val, preds, pos_label=0, zero_division=0))
        accuracies.append(accuracy_score(y_val, preds))

        # ROC AUC requires predicted probabilities
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(X_val)[:, 1]
            try:
                aucs.append(roc_auc_score(y_val, probs))
            except ValueError:
                aucs.append(0.5)
        else:
            aucs.append(0.5)

    return {
        "cv_recall_at_risk": float(np.mean(recalls_at_risk)),
        "cv_recall_at_risk_std": float(np.std(recalls_at_risk)),
        "cv_precision_at_risk": float(np.mean(precisions_at_risk)),
        "cv_f1_at_risk": float(np.mean(f1s_at_risk)),
        "cv_accuracy": float(np.mean(accuracies)),
        "cv_roc_auc": float(np.mean(aucs)),
    }


def build_candidate_pipelines(random_state: int = 42) -> Dict[str, Pipeline]:
    """Instantiate candidate modeling pipelines with shared preprocessors.
    
    Includes:
    - Majority Class Baseline (DummyClassifier)
    - Logistic Regression Baseline (default unweighted)
    - Logistic Regression Balanced (class_weight='balanced')
    - Decision Tree Balanced (max_depth=4, class_weight='balanced')
    - Random Forest Balanced (n_estimators=100, max_depth=5, class_weight='balanced')
    - Gradient Boosting Classifier (n_estimators=100, max_depth=3)
    
    Args:
        random_state: Seed for reproducibility.
        
    Returns:
        Dict mapping model names to configured Scikit-learn Pipelines.
    """
    preprocessor = build_preprocessor()

    pipelines = {
        "Majority Class Baseline": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", DummyClassifier(strategy="most_frequent")),
        ]),
        "Logistic Regression Baseline": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(random_state=random_state, max_iter=1000)),
        ]),
        "Logistic Regression Balanced": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                class_weight="balanced", C=0.5, random_state=random_state, max_iter=1000
            )),
        ]),
        "Decision Tree Balanced": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(
                class_weight="balanced", max_depth=4, min_samples_leaf=5, random_state=random_state
            )),
        ]),
        "Random Forest Balanced": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=150, max_depth=5, min_samples_leaf=4,
                class_weight="balanced", random_state=random_state
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05, random_state=random_state
            )),
        ]),
    }
    return pipelines


def train_and_compare_models(
    X_train: pd.DataFrame, y_train: pd.Series, cv_splits: int = 5, random_state: int = 42
) -> pd.DataFrame:
    """Train all candidate pipelines and construct CV comparison table.
    
    Args:
        X_train: Training features.
        y_train: Training target.
        cv_splits: Folds for cross-validation.
        random_state: Seed.
        
    Returns:
        pd.DataFrame summarizing CV metrics across all candidates.
    """
    pipelines = build_candidate_pipelines(random_state=random_state)
    results = []

    for name, pipe in pipelines.items():
        metrics = get_cv_metrics(pipe, X_train, y_train, cv_splits=cv_splits, random_state=random_state)
        metrics["model"] = name
        results.append(metrics)

    df_results = pd.DataFrame(results)
    # Order columns meaningfully
    cols = [
        "model", "cv_recall_at_risk", "cv_recall_at_risk_std",
        "cv_precision_at_risk", "cv_f1_at_risk", "cv_accuracy", "cv_roc_auc"
    ]
    df_results = df_results[cols].sort_values(by="cv_recall_at_risk", ascending=False).reset_index(drop=True)
    return df_results


def select_and_fit_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "Logistic Regression Balanced",
    random_state: int = 42,
) -> Pipeline:
    """Fit selected winning pipeline on the full training dataset.
    
    Args:
        X_train: Training features.
        y_train: Training target.
        model_name: Candidate name to select.
        random_state: Seed.
        
    Returns:
        Fitted Scikit-learn Pipeline.
    """
    pipelines = build_candidate_pipelines(random_state=random_state)
    if model_name not in pipelines:
        raise ValueError(f"Unknown model name '{model_name}'. Choose from: {list(pipelines.keys())}")

    final_pipeline = pipelines[model_name]
    final_pipeline.fit(X_train, y_train)
    return final_pipeline


def evaluate_on_test_set(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, Any]:
    """Perform rigorous evaluation on the untouched holdout test set.
    
    Args:
        pipeline: Fitted Pipeline.
        X_test: Test features.
        y_test: Test target (1=Pass, 0=At-Risk).
        
    Returns:
        Dict with comprehensive test evaluation metrics and confusion matrix.
    """
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

    # Class 0: At-Risk / Fail; Class 1: Pass
    acc = accuracy_score(y_test, y_pred)
    rec_at_risk = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    prec_at_risk = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1_at_risk = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
    rec_pass = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    prec_pass = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_pass = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.5
    clf_report = classification_report(
        y_test, y_pred, target_names=["At-Risk (0)", "Pass (1)"], output_dict=True
    )

    # Detailed confusion matrix counts
    # tn: at_risk predicted as at_risk (true risk detected)
    # fp: at_risk predicted as pass (missed at-risk student = FALSE NEGATIVE from risk perspective)
    # fn: pass predicted as at_risk (false alarm)
    # tp: pass predicted as pass
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": float(acc),
        "recall_at_risk": float(rec_at_risk),
        "precision_at_risk": float(prec_at_risk),
        "f1_at_risk": float(f1_at_risk),
        "recall_pass": float(rec_pass),
        "precision_pass": float(prec_pass),
        "f1_pass": float(f1_pass),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_counts": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "classification_report": clf_report,
    }


def get_feature_names(pipeline: Pipeline) -> List[str]:
    """Extract transformed feature names out of the ColumnTransformer."""
    preprocessor = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def save_model(pipeline: Pipeline, filepath: Union[str, Path] = "models/final_model.joblib") -> Path:
    """Serialize trained pipeline to disk via Joblib."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_model(filepath: Union[str, Path] = "models/final_model.joblib") -> Pipeline:
    """Load serialized pipeline from disk via Joblib."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(
            f"Serialized model pipeline not found at '{path.resolve()}'. "
            "Please run 'src/training.py' or 'notebooks/05_modelling.ipynb' to train and save the model."
        )
    return joblib.load(path)


def run_full_training_pipeline() -> Tuple[Pipeline, Dict[str, Any], pd.DataFrame]:
    """Execute end-to-end model training, CV comparison, selection, saving, and evaluation.
    
    Returns:
        Tuple (fitted_pipeline, test_metrics, cv_results_table)
    """
    # 1. Load and clean
    raw = load_raw_data()
    clean_df = clean_student_data(raw)
    
    # 2. Engineer pre-outcome features
    feat_df = engineer_features(clean_df)
    X, y = get_feature_target_split(feat_df)
    
    # 3. Train/Test split (untouched holdout test set)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42, stratify=True)
    
    # 4. CV comparison across candidate models
    cv_table = train_and_compare_models(X_train, y_train, cv_splits=5, random_state=42)
    
    # 5. Select best model prioritizing at-risk recall
    # Logistic Regression Balanced is our top candidate for high at-risk recall with interpretability
    selected_name = "Logistic Regression Balanced"
    final_pipeline = select_and_fit_final_model(X_train, y_train, model_name=selected_name, random_state=42)
    
    # 6. Save model pipeline
    save_path = save_model(final_pipeline, "models/final_model.joblib")
    
    # 7. Evaluate on untouched test set
    test_metrics = evaluate_on_test_set(final_pipeline, X_test, y_test)
    
    return final_pipeline, test_metrics, cv_table


if __name__ == "__main__":
    print("Executing full training and evaluation workflow...")
    pipeline, test_eval, cv_results = run_full_training_pipeline()
    
    print("\n--- Cross-Validation Model Comparison Table ---")
    print(cv_results.to_string(index=False))
    
    print("\n--- Holdout Test Set Evaluation Metrics ---")
    print(f"Accuracy:                {test_eval['accuracy']:.4f}")
    print(f"At-Risk Recall (Class 0): {test_eval['recall_at_risk']:.4f}")
    print(f"At-Risk Precision:       {test_eval['precision_at_risk']:.4f}")
    print(f"At-Risk F1-Score:        {test_eval['f1_at_risk']:.4f}")
    print(f"ROC-AUC:                 {test_eval['roc_auc']:.4f}")
    print("\nConfusion Matrix [Row: True (0: At-Risk, 1: Pass), Col: Pred]:")
    print(np.array(test_eval["confusion_matrix"]))
    print(f"\nFinal pipeline saved to: models/final_model.joblib")
