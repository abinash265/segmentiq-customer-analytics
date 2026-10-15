"""Load, validate, clean and standardise the customer data."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_loader import DATA_PATH, FEATURE_COLUMNS, REQUIRED_COLUMNS

# Plausible limits; values outside them are treated as invalid and imputed.
VALID_RANGES = {
    "Age": (10, 100),
    "AnnualIncome": (0, 10000),
    "SpendingScore": (1, 100),
    "PurchaseFrequency": (0, 1000),
    "AverageOrderValue": (0, 1_000_000),
    "MembershipYears": (0, 80),
}


def validate_columns(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("The dataset is missing required columns: " + ", ".join(missing))


def load_csv(path=DATA_PATH):
    """Read the CSV (read-only) and check that the required columns exist."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise ValueError(f"Could not read {path.name}: {error}") from error
    validate_columns(df)
    return df


def prepare_data(df):
    """Clean the data and standardise the six clustering features.

    Returns (cleaned_df, scaled_X, scaler, quality_summary).
    The input DataFrame and the CSV file are never modified.
    """
    validate_columns(df)
    if df.empty:
        raise ValueError("The dataset has no rows.")

    original_rows = len(df)
    missing_by_column = {c: int(df[c].isna().sum()) for c in FEATURE_COLUMNS}
    duplicate_rows = int(df.duplicated().sum())

    # 1) Remove exact duplicate records.
    cleaned = df.drop_duplicates().reset_index(drop=True).copy()

    # 2) Make features numeric; mark invalid entries (text, impossible values) as missing.
    invalid_values = 0
    for column in FEATURE_COLUMNS:
        numeric = pd.to_numeric(cleaned[column], errors="coerce")
        not_a_number = numeric.isna() & cleaned[column].notna()   # e.g. the text "abc"
        low, high = VALID_RANGES[column]
        out_of_range = numeric.notna() & ((numeric < low) | (numeric > high))
        invalid = not_a_number | out_of_range
        invalid_values += int(invalid.sum())
        cleaned[column] = numeric.mask(invalid)

    # 3) Fill missing values with the column median.
    imputed_values = 0
    for column in FEATURE_COLUMNS:
        if cleaned[column].isna().all():
            raise ValueError(f"Column '{column}' has no valid values to compute a median.")
        imputed_values += int(cleaned[column].isna().sum())
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    # 4) Standardise (mean 0, std 1). CustomerID is NOT included.
    scaler = StandardScaler()
    scaled_x = scaler.fit_transform(cleaned[FEATURE_COLUMNS])
    if not np.isfinite(scaled_x).all():
        raise ValueError("Standardisation produced non-finite values.")

    summary = {
        "original_rows": original_rows,
        "missing_values": sum(missing_by_column.values()),
        "missing_by_column": missing_by_column,
        "duplicate_rows": duplicate_rows,
        "invalid_values": invalid_values,
        "imputed_values": imputed_values,
        "cleaned_rows": len(cleaned),
    }
    return cleaned, scaled_x, scaler, summary
