"""Synthetic customer data: generation, saving and loading.

The dataset is created ONCE and saved to data/customers.csv.
The app loads that saved file; it never silently regenerates it.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Project root = folder that contains main.py (works from any working directory).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "customers.csv"
SEED = 42

# CustomerID is only an identifier. These six columns are the clustering features.
FEATURE_COLUMNS = [
    "Age",
    "AnnualIncome",
    "SpendingScore",
    "PurchaseFrequency",
    "AverageOrderValue",
    "MembershipYears",
]
REQUIRED_COLUMNS = ["CustomerID"] + FEATURE_COLUMNS

# Each group: (count, Age, Income, Spending, Frequency, OrderValue, Membership)
# where each pair is (mean, standard deviation). Different patterns per group.
CUSTOMER_GROUPS = [
    (90, (24, 4), (35, 7), (70, 10), (40, 8), (45, 10), (2, 1)),     # young budget shoppers
    (80, (33, 5), (95, 12), (82, 8), (55, 8), (160, 25), (5, 2)),    # affluent big spenders
    (80, (45, 7), (100, 12), (20, 8), (12, 4), (120, 25), (6, 2)),   # high income, low spending
    (85, (50, 8), (30, 7), (25, 8), (10, 4), (30, 8), (7, 3)),       # low income, careful
    (100, (38, 8), (60, 8), (50, 8), (25, 6), (75, 15), (4, 2)),     # average customers
    (65, (60, 5), (55, 8), (45, 10), (20, 5), (90, 15), (12, 3)),    # older long-term members
]


def generate_customers(seed=SEED, add_issues=True):
    """Return a synthetic customer DataFrame (500 base records).

    add_issues=True adds a few missing values and duplicate rows so that the
    preprocessing step has something real to clean.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for count, age, income, spend, freq, order, member in CUSTOMER_GROUPS:
        parts.append(pd.DataFrame({
            "Age": rng.normal(age[0], age[1], count),
            "AnnualIncome": rng.normal(income[0], income[1], count),
            "SpendingScore": rng.normal(spend[0], spend[1], count),
            "PurchaseFrequency": rng.normal(freq[0], freq[1], count),
            "AverageOrderValue": rng.normal(order[0], order[1], count),
            "MembershipYears": rng.normal(member[0], member[1], count),
        }))
    df = pd.concat(parts, ignore_index=True)
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)  # mix the groups

    # Keep values in realistic ranges and round them.
    df["Age"] = df["Age"].clip(18, 75).round(0)
    df["AnnualIncome"] = df["AnnualIncome"].clip(10, 150).round(1)   # thousands
    df["SpendingScore"] = df["SpendingScore"].clip(1, 100).round(0)
    df["PurchaseFrequency"] = df["PurchaseFrequency"].clip(1, 120).round(0)
    df["AverageOrderValue"] = df["AverageOrderValue"].clip(10, 300).round(2)
    df["MembershipYears"] = df["MembershipYears"].clip(0, 20).round(1)
    df.insert(0, "CustomerID", [f"C{i:04d}" for i in range(1, len(df) + 1)])

    if add_issues:
        df = _add_data_quality_issues(df, rng)
    return df


def _add_data_quality_issues(df, rng, n_missing=10, n_duplicates=8):
    """Blank a few numeric cells and repeat a few whole rows (small amounts)."""
    df = df.copy()
    for _ in range(n_missing):
        row = int(rng.integers(0, len(df)))
        col = FEATURE_COLUMNS[int(rng.integers(0, len(FEATURE_COLUMNS)))]
        df.loc[row, col] = np.nan
    duplicates = df.iloc[rng.choice(len(df), size=n_duplicates, replace=False)]
    df = pd.concat([df, duplicates])
    return df.sort_values("CustomerID", kind="stable").reset_index(drop=True)


def generate_and_save(path=DATA_PATH, seed=SEED, overwrite=False):
    """Generate the dataset and save it as CSV. Refuses to overwrite unless told to."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use overwrite=True to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_customers(seed=seed)
    df.to_csv(path, index=False)
    return path


def ensure_dataset(path=DATA_PATH):
    """Create the CSV only if it does not exist yet. Never overwrites."""
    path = Path(path)
    if not path.exists():
        generate_and_save(path, overwrite=False)
    return path
