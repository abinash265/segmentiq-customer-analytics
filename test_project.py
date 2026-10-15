"""Automated tests (unittest). They use small generated data and temp folders,
so the real data/customers.csv and outputs/ folder are never touched."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import clustering, data_loader, evaluation, export_results, preprocessing
from src.data_loader import FEATURE_COLUMNS


def small_frame():
    """Tiny hand-made dataset: 4 rows, with one duplicate and one missing value."""
    return pd.DataFrame({
        "CustomerID": ["C1", "C2", "C2", "C3", "C4"],
        "Age": [25, 40, 40, np.nan, 55],
        "AnnualIncome": [30, 80, 80, 50, 90],
        "SpendingScore": [70, 20, 20, 50, 30],
        "PurchaseFrequency": [40, 10, 10, 25, 12],
        "AverageOrderValue": [40, 120, 120, 70, 130],
        "MembershipYears": [1, 6, 6, 3, 9],
    })


class TestProject(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = data_loader.generate_customers(seed=42)
        cls.cleaned, cls.scaled, cls.scaler, cls.summary = preprocessing.prepare_data(cls.raw)

    # 1-2: dataset generation
    def test_01_dataset_has_required_columns(self):
        for column in data_loader.REQUIRED_COLUMNS:
            self.assertIn(column, self.raw.columns)

    def test_02_customer_ids_unique_in_base_dataset(self):
        base = data_loader.generate_customers(seed=42, add_issues=False)
        self.assertEqual(len(base), 500)
        self.assertTrue(base["CustomerID"].is_unique)

    def test_dataset_is_reproducible(self):
        again = data_loader.generate_customers(seed=42)
        pd.testing.assert_frame_equal(self.raw, again)

    # 3-4: cleaning
    def test_03_duplicates_removed(self):
        cleaned, _, _, summary = preprocessing.prepare_data(small_frame())
        self.assertEqual(summary["duplicate_rows"], 1)
        self.assertEqual(len(cleaned), 4)
        self.assertTrue(cleaned["CustomerID"].is_unique)

    def test_04_missing_values_imputed_with_median(self):
        cleaned, _, _, summary = preprocessing.prepare_data(small_frame())
        self.assertEqual(summary["missing_values"], 1)
        self.assertFalse(cleaned[FEATURE_COLUMNS].isna().any().any())
        self.assertEqual(cleaned.loc[cleaned["CustomerID"] == "C3", "Age"].iloc[0], 40)  # median of 25, 40, 55

    def test_invalid_values_are_handled(self):
        df = small_frame()
        df["SpendingScore"] = df["SpendingScore"].astype(object)  # a CSV with text in the column
        df.loc[0, "AnnualIncome"] = -5
        df.loc[1, "SpendingScore"] = "abc"
        cleaned, _, _, summary = preprocessing.prepare_data(df)
        self.assertEqual(summary["invalid_values"], 2)
        self.assertFalse(cleaned[FEATURE_COLUMNS].isna().any().any())

    def test_source_dataframe_not_modified(self):
        df = small_frame()
        before = df.copy()
        preprocessing.prepare_data(df)
        pd.testing.assert_frame_equal(df, before)

    def test_missing_column_rejected(self):
        with self.assertRaises(ValueError):
            preprocessing.prepare_data(small_frame().drop(columns=["Age"]))

    # 5-6: scaling and features
    def test_05_scaled_values_are_finite(self):
        self.assertTrue(np.isfinite(self.scaled).all())
        np.testing.assert_allclose(self.scaled.mean(axis=0), 0, atol=1e-8)

    def test_06_customer_id_not_a_feature(self):
        self.assertNotIn("CustomerID", FEATURE_COLUMNS)
        self.assertEqual(self.scaled.shape, (len(self.cleaned), 6))

    # 7-9: K-Means
    def test_07_one_label_per_record(self):
        labeled, _ = clustering.run_kmeans(self.cleaned, self.scaled, 4)
        self.assertEqual(len(labeled), len(self.cleaned))
        self.assertNotIn("Cluster", self.cleaned.columns)  # original left untouched

    def test_08_labels_are_valid_ids(self):
        labeled, _ = clustering.run_kmeans(self.cleaned, self.scaled, 4)
        self.assertTrue(set(labeled["Cluster"]).issubset(set(range(4))))

    def test_09_summary_has_required_statistics(self):
        labeled, _ = clustering.run_kmeans(self.cleaned, self.scaled, 3)
        summary = clustering.summarize_clusters(labeled)
        for column in ["Cluster", "Customers", "Percentage", "AvgAge", "AvgAnnualIncome",
                       "AvgSpendingScore", "AvgPurchaseFrequency", "AvgOrderValue",
                       "AvgMembershipYears", "Profile"]:
            self.assertIn(column, summary.columns)
        self.assertEqual(summary["Customers"].sum(), len(labeled))
        self.assertAlmostEqual(summary["Percentage"].sum(), 100, delta=0.5)

    def test_kmeans_is_reproducible(self):
        a, _ = clustering.run_kmeans(self.cleaned, self.scaled, 4)
        b, _ = clustering.run_kmeans(self.cleaned, self.scaled, 4)
        self.assertTrue((a["Cluster"] == b["Cluster"]).all())

    # 10: elbow
    def test_10_elbow_covers_k_2_to_8(self):
        elbow = evaluation.run_elbow(self.scaled)
        self.assertEqual(list(elbow["K"]), list(range(2, 9)))
        self.assertTrue((elbow["Inertia"] > 0).all())
        self.assertTrue(elbow["Inertia"].is_monotonic_decreasing)

    def test_elbow_skips_k_larger_than_records(self):
        elbow = evaluation.run_elbow(self.scaled[:4])
        self.assertEqual(list(elbow["K"]), [2, 3, 4])

    # 11: export
    def test_11_exports_create_valid_csv_files(self):
        labeled, _ = clustering.run_kmeans(self.cleaned, self.scaled, 3)
        summary = clustering.summarize_clusters(labeled)
        elbow = evaluation.run_elbow(self.scaled)
        with tempfile.TemporaryDirectory() as folder:
            seg = export_results.export_segments(labeled, Path(folder) / "sub" / "seg.csv")
            summ = export_results.export_summary(summary, Path(folder) / "summary.csv")
            elb = export_results.export_elbow(elbow, Path(folder) / "elbow.csv")
            self.assertEqual(len(pd.read_csv(seg)), len(labeled))
            self.assertIn("Cluster", pd.read_csv(seg).columns)
            self.assertEqual(len(pd.read_csv(summ)), 3)
            self.assertEqual(list(pd.read_csv(elb)["K"]), list(range(2, 9)))

    def test_export_errors_are_readable(self):
        with self.assertRaises(export_results.ExportError):
            export_results.export_summary(None)
        with tempfile.TemporaryDirectory() as folder:
            blocker = Path(folder) / "file.txt"
            blocker.write_text("x")
            with self.assertRaises(export_results.ExportError):
                export_results.export_elbow(pd.DataFrame({"K": [2]}), blocker / "out.csv")

    # 12: invalid K
    def test_12_invalid_k_rejected(self):
        for bad in [1, 9, 0, -3, 2.5, "abc", None, True]:
            with self.subTest(k=bad):
                with self.assertRaises(ValueError):
                    clustering.validate_k(bad, 500)
        with self.assertRaises(ValueError):
            clustering.validate_k(5, 4)  # more clusters than records
        self.assertEqual(clustering.validate_k("3", 500), 3)

    # data file handling
    def test_csv_not_overwritten_without_permission(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "customers.csv"
            data_loader.ensure_dataset(path)             # creates it
            self.assertTrue(path.exists())
            first = path.read_text()
            data_loader.ensure_dataset(path)             # leaves it alone
            self.assertEqual(first, path.read_text())
            with self.assertRaises(FileExistsError):
                data_loader.generate_and_save(path)      # refuses to overwrite
            loaded = preprocessing.load_csv(path)
            self.assertGreater(len(loaded), 500)

    def test_missing_csv_gives_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            preprocessing.load_csv("does_not_exist.csv")


if __name__ == "__main__":
    unittest.main()
