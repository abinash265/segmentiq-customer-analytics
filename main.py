"""Entry point.

    python main.py                  open the GUI
    python main.py --generate-data  regenerate data/customers.csv (overwrites it!)
    python main.py --test           run the automated tests
"""
import argparse
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))   # so "import src..." works from any folder


def run_tests():
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(description="Customer Segmentation System (K-Means)")
    parser.add_argument("--generate-data", action="store_true",
                        help="regenerate data/customers.csv (replaces the existing file)")
    parser.add_argument("--test", action="store_true", help="run the automated tests")
    args = parser.parse_args()

    if args.test:
        return run_tests()

    if args.generate_data:
        from src import data_loader
        path = data_loader.generate_and_save(overwrite=True)
        print(f"Sample data generated: {path}")
        return 0

    try:
        import streamlit.web.cli as stcli
        import sys as _sys
        _sys.argv = ["streamlit", "run", str(ROOT / "app.py"), "--server.headless", "false"]
        stcli.main()
        return 0
    except ImportError as error:
        print(f"Could not start the web dashboard: {error}")
        print("Install dependencies with: python -m pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
