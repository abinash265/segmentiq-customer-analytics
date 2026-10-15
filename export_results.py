"""Save analysis results as CSV files."""
from pathlib import Path

from src.data_loader import PROJECT_ROOT

OUTPUT_DIR = PROJECT_ROOT / "outputs"


class ExportError(Exception):
    """Raised with a readable message when a file cannot be written."""


def _write_csv(df, path):
    if df is None or len(df) == 0:
        raise ExportError("There are no results to export yet.")
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    except PermissionError:
        raise ExportError(f"Permission denied: {path}\nClose the file if it is open in Excel, or choose another location.") from None
    except OSError as error:
        raise ExportError(f"Could not save the file: {path}\n{error}") from error
    return path


def export_segments(labeled_df, path=OUTPUT_DIR / "customer_segments.csv"):
    """Cleaned customer records with their cluster assignment."""
    return _write_csv(labeled_df, path)


def export_summary(summary_df, path=OUTPUT_DIR / "cluster_summary.csv"):
    """Per-cluster statistics and profiles."""
    return _write_csv(summary_df, path)


def export_elbow(elbow_df, path=OUTPUT_DIR / "elbow_results.csv"):
    """K and inertia values from the Elbow Method."""
    return _write_csv(elbow_df, path)
