"""Feature Engineering and Preprocessing Module for Student Performance Analyzer.

Provides feature construction strictly restricted to pre-outcome attributes
to eliminate target leakage, along with Scikit-learn ColumnTransformers and
stratified train/test splitting routines.
"""

from pathlib import Path
from typing import List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Core pre-outcome numeric features available at start of term
RAW_NUMERIC_FEATURES: List[str] = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"
]

# Engineered pre-outcome numeric features
ENGINEERED_NUMERIC_FEATURES: List[str] = [
    "parent_edu_avg", "total_alc", "support_index", "absences_capped", "attendance_rate_est"
]

# Combined numeric feature list for ColumnTransformer
ALL_NUMERIC_FEATURES: List[str] = RAW_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES

# Categorical features available at enrollment / term commencement
CATEGORICAL_FEATURES: List[str] = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
    "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic"
]

# TARGET LEAKAGE AUDIT:
# G1, G2, and G3 MUST NOT be included in the feature set.
# Grade trend features (e.g. G2 - G1) are likewise EXCLUDED because they
# require midterm and second-term outcomes, which violates the early-warning mandate.
LEAKAGE_COLUMNS: List[str] = ["G1", "G2", "G3", "passed", "at_risk"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer domain-specific features using solely pre-outcome information.
    
    New features constructed:
    1. parent_edu_avg: Arithmetic mean of mother's and father's education level.
    2. total_alc: Weighted weekly alcohol consumption (5 workdays + 2 weekend days).
    3. support_index: Combined count of extra institutional, family, and paid academic support (0-3).
    4. absences_capped: Absences clipped at 30 days to limit the leverage of extreme outliers.
    5. attendance_rate_est: Estimated attendance fraction assuming ~90 instructional sessions.
    
    Args:
        df: Input DataFrame (contains cleaned raw columns).
        
    Returns:
        pd.DataFrame with new engineered feature columns appended.
    """
    data = df.copy()

    # 1. Consolidated parental education scale (0 to 4)
    data["parent_edu_avg"] = (data["Medu"] + data["Fedu"]) / 2.0

    # 2. Weighted weekly alcohol consumption (1 to 5)
    data["total_alc"] = (data["Dalc"] * 5.0 + data["Walc"] * 2.0) / 7.0

    # 3. Academic support mechanism count (0 to 3)
    support_flags = [
        (data["schoolsup"].astype(str).str.lower() == "yes").astype(int),
        (data["famsup"].astype(str).str.lower() == "yes").astype(int),
        (data["paid"].astype(str).str.lower() == "yes").astype(int),
    ]
    data["support_index"] = sum(support_flags)

    # 4. Outlier-mitigated attendance features
    data["absences_capped"] = data["absences"].clip(upper=30)
    data["attendance_rate_est"] = (1.0 - (data["absences_capped"] / 90.0)).clip(lower=0.0, upper=1.0)

    return data


def get_feature_target_split(
    df: pd.DataFrame, target_col: str = "passed"
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate pre-outcome predictor features X from ground-truth target y.
    
    Explicitly enforces zero target leakage by dropping G1, G2, G3 and any
    target columns from the feature matrix X.
    
    Args:
        df: Feature-engineered DataFrame.
        target_col: Target column name ('passed' where 1=Pass, 0=Fail/At-Risk).
        
    Returns:
        Tuple (X, y)
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not present in DataFrame.")

    # Select only designated pre-outcome features
    feature_cols = ALL_NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing_feats = [col for col in feature_cols if col not in df.columns]
    if missing_feats:
        raise KeyError(f"Missing expected features: {missing_feats}. Did you call engineer_features()?")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Integrity verification against target leakage
    for leak in LEAKAGE_COLUMNS:
        assert leak not in X.columns, f"TARGET LEAKAGE DETECTED: Column '{leak}' found in X!"

    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Construct Scikit-learn ColumnTransformer for standardizing and encoding.
    
    - Numeric features: StandardScaler
    - Categorical features: OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    Returns:
        ColumnTransformer instance ready for pipeline integration.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, ALL_NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform reproducible train/test split with stratification.
    
    Args:
        X: Feature matrix.
        y: Target series.
        test_size: Proportion of dataset allocated to the holdout test set.
        random_state: Reproducible seed.
        stratify: Whether to preserve class proportions.
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    strat = y if stratify else None
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.cleaning import load_raw_data, clean_student_data

    print("Testing feature engineering module...")
    raw = load_raw_data()
    clean_df = clean_student_data(raw)
    feat_df = engineer_features(clean_df)
    X, y = get_feature_target_split(feat_df)
    print(f"Features X shape: {X.shape}, Target y shape: {y.shape}")
    print(f"Numeric features count: {len(ALL_NUMERIC_FEATURES)}")
    print(f"Categorical features count: {len(CATEGORICAL_FEATURES)}")
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train split: {X_train.shape[0]} rows, Test split: {X_test.shape[0]} rows")
    print(f"Train pass rate: {y_train.mean():.4f}, Test pass rate: {y_test.mean():.4f}")
    
    prep = build_preprocessor()
    X_train_trans = prep.fit_transform(X_train)
    print(f"Transformed X_train shape: {X_train_trans.shape}")
    print("Feature engineering verification completed successfully!")
