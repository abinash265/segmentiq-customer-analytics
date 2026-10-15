"""Elbow Method: inertia for K = 2..8 on the current scaled dataset."""
import pandas as pd
from sklearn.cluster import KMeans

from src.clustering import MAX_K, MIN_K, N_INIT, RANDOM_STATE


def run_elbow(scaled_x, k_min=MIN_K, k_max=MAX_K):
    """Fit K-Means for each K and return a DataFrame with columns K and Inertia.

    Inertia = sum of squared distances from each point to its cluster centre.
    K values larger than the number of records are skipped.
    """
    n_samples = len(scaled_x)
    k_values = [k for k in range(k_min, k_max + 1) if k <= n_samples]
    if not k_values:
        raise ValueError("Not enough records to run the Elbow Method (need at least 2).")
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT).fit(scaled_x)
        rows.append({"K": k, "Inertia": round(float(model.inertia_), 4)})
    return pd.DataFrame(rows)
