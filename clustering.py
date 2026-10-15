"""K-Means clustering and cluster summaries."""
import pandas as pd
from sklearn.cluster import KMeans

from src.data_loader import FEATURE_COLUMNS

MIN_K = 2
MAX_K = 8
RANDOM_STATE = 42   # makes the result reproducible
N_INIT = 10         # number of random starts; the best run is kept


def validate_k(k, n_samples):
    """Return k as an int, or raise ValueError with a readable message."""
    message = f"K must be a whole number between {MIN_K} and {MAX_K}."
    if isinstance(k, bool):
        raise ValueError(message)
    try:
        k_int = int(k)
        if isinstance(k, float) and k != k_int:
            raise ValueError(message)
    except (TypeError, ValueError):
        raise ValueError(message) from None
    if not MIN_K <= k_int <= MAX_K:
        raise ValueError(message)
    if k_int > n_samples:
        raise ValueError(f"K ({k_int}) cannot be larger than the number of records ({n_samples}).")
    return k_int


def run_kmeans(cleaned_df, scaled_x, k):
    """Run K-Means on the standardised features.

    Returns (copy of cleaned_df with a 'Cluster' column, fitted model).
    """
    if len(cleaned_df) != len(scaled_x):
        raise ValueError("Cleaned data and scaled matrix have different lengths.")
    k = validate_k(k, len(cleaned_df))
    try:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        model.fit(scaled_x)
    except Exception as error:  # report any sklearn problem in a readable way
        raise RuntimeError(f"Clustering failed: {error}") from error
    labeled = cleaned_df.copy()
    labeled["Cluster"] = model.labels_
    return labeled, model


# Words used when a cluster's average is clearly above / below the overall average.
_DESCRIPTIONS = {
    "Age": ("older customers", "younger customers"),
    "AnnualIncome": ("high income", "low income"),
    "SpendingScore": ("high spending score", "low spending score"),
    "PurchaseFrequency": ("frequent purchases", "infrequent purchases"),
    "AverageOrderValue": ("high order value", "low order value"),
    "MembershipYears": ("long membership", "short membership"),
}


def describe_cluster(cluster_means, overall_means, overall_stds, threshold=0.5):
    """Describe a cluster by comparing its averages with the overall averages.

    A feature is 'high' or 'low' only if the cluster average differs from the
    overall average by more than `threshold` standard deviations.
    """
    terms = []
    for feature in FEATURE_COLUMNS:
        std = overall_stds[feature]
        if std == 0:
            continue
        z = (cluster_means[feature] - overall_means[feature]) / std
        if z > threshold:
            terms.append(_DESCRIPTIONS[feature][0])
        elif z < -threshold:
            terms.append(_DESCRIPTIONS[feature][1])
    if not terms:
        return "Close to the overall average on all six features."
    return "Relatively " + ", ".join(terms) + " (compared with all customers)."


def summarize_clusters(labeled_df):
    """One row per cluster: size, percentage, feature averages and a data-driven profile."""
    if "Cluster" not in labeled_df.columns:
        raise ValueError("The data has no 'Cluster' column. Run clustering first.")
    overall_means = labeled_df[FEATURE_COLUMNS].mean()
    overall_stds = labeled_df[FEATURE_COLUMNS].std()
    rows = []
    for cluster_id, group in labeled_df.groupby("Cluster"):
        means = group[FEATURE_COLUMNS].mean()
        rows.append({
            "Cluster": int(cluster_id),
            "Customers": len(group),
            "Percentage": round(100 * len(group) / len(labeled_df), 1),
            "AvgAge": round(means["Age"], 2),
            "AvgAnnualIncome": round(means["AnnualIncome"], 2),
            "AvgSpendingScore": round(means["SpendingScore"], 2),
            "AvgPurchaseFrequency": round(means["PurchaseFrequency"], 2),
            "AvgOrderValue": round(means["AverageOrderValue"], 2),
            "AvgMembershipYears": round(means["MembershipYears"], 2),
            "Profile": describe_cluster(means, overall_means, overall_stds),
        })
    return pd.DataFrame(rows)
