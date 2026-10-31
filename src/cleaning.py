"""Data Cleaning Module for Student Performance Analyzer.

Provides functions to load raw student data, validate integrity, apply cleaning
rules, derive target variables, and persist processed datasets.
"""

from pathlib import Path
from typing import Optional, Union
import pandas as pd


EXPECTED_COLUMNS = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime", "failures",
    "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet",
    "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health",
    "absences", "G1", "G2", "G3"
]

NUMERIC_COLUMNS = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
    "G1", "G2", "G3"
]

CATEGORICAL_COLUMNS = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
    "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic"
]


def load_raw_data(filepath: Union[str, Path] = "data/raw/student-mat.csv") -> pd.DataFrame:
    """Load raw student mathematics dataset with delimiter configuration.
    
    Args:
        filepath: Path to the raw CSV file.
        
    Returns:
        pd.DataFrame containing the raw data.
        
    Raises:
        FileNotFoundError: If the specified raw data file does not exist.
        ValueError: If file cannot be read or delimiter is invalid.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(
            f"Raw dataset file not found at '{path.resolve()}'. "
            "Please ensure 'student-mat.csv' is placed inside 'data/raw/'. "
            "Refer to 'docs/data_source.md' for download instructions."
        )

    try:
        df = pd.read_csv(path, sep=";")
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV file at '{path}': {exc}") from exc

    return df


def validate_raw_schema(df: pd.DataFrame) -> bool:
    """Validate that dataframe contains all required columns.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        True if valid.
        
    Raises:
        ValueError: If any expected column is missing.
    """
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Raw data schema validation failed. Missing columns: {sorted(missing_cols)}")
    return True


def clean_student_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform deterministic data cleaning and derive classification target.
    
    Steps performed:
    1. Validate schema integrity.
    2. Check and remove exact duplicate rows.
    3. Strip whitespace from string/categorical columns.
    4. Validate and enforce appropriate data types.
    5. Validate numerical ranges (e.g., age, absences, Likert scales).
    6. Derive binary target: 'passed' (1 if G3 >= 10 else 0) and 'at_risk' (1 - passed).
    
    Args:
        df: Input DataFrame (raw or loaded).
        
    Returns:
        pd.DataFrame: Cleaned and validated dataset.
    """
    validate_raw_schema(df)
    cleaned = df.copy()

    # 1. Deduplication
    cleaned = cleaned.drop_duplicates()

    # 2. String standardization (strip extra whitespace and quotes if any)
    for col in CATEGORICAL_COLUMNS:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype(str).str.strip()

    # 3. Numeric conversion and boundary clipping/validation
    for col in NUMERIC_COLUMNS:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # Range validation
    cleaned = cleaned[cleaned["age"].between(10, 30)]
    cleaned = cleaned[cleaned["absences"] >= 0]
    cleaned = cleaned[cleaned["G3"].between(0, 20)]

    # 4. Target derivation: Portuguese 0-20 scale: Pass is G3 >= 10
    # passed: 1 for Pass, 0 for Fail / At-Risk
    # at_risk: 1 for At-Risk (G3 < 10), 0 for Pass
    cleaned["passed"] = (cleaned["G3"] >= 10).astype(int)
    cleaned["at_risk"] = (cleaned["G3"] < 10).astype(int)

    # Reset index for consistent indexing
    cleaned = cleaned.reset_index(drop=True)
    return cleaned


def save_processed_data(
    df: pd.DataFrame, output_path: Union[str, Path] = "data/processed/student_clean.csv"
) -> Path:
    """Save cleaned dataset to destination path.
    
    Args:
        df: Cleaned DataFrame.
        output_path: Target CSV file path.
        
    Returns:
        Path to the saved processed dataset.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    print("Executing standalone cleaning pipeline...")
    raw = load_raw_data()
    print(f"Loaded raw data with shape: {raw.shape}")
    clean_df = clean_student_data(raw)
    out = save_processed_data(clean_df)
    print(f"Cleaned dataset saved successfully to '{out}' with shape: {clean_df.shape}")
    print(f"Pass count: {(clean_df['passed'] == 1).sum()}, At-Risk count: {(clean_df['at_risk'] == 1).sum()}")
