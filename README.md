# Customer Segmentation System Using K-Means Clustering

A Data Warehousing and Mining (DWM) mini-project: a Python desktop app (Tkinter) that groups customers
into segments with K-Means clustering, shows charts and statistics, and exports the results to CSV.

## Abstract
Businesses want to treat different kinds of customers differently. This project generates a synthetic
customer dataset (500 customers, saved as CSV), cleans it, standardises six numeric features, and applies
K-Means clustering. The user picks the number of clusters K (2-8) with the help of the Elbow Method, then
views a cluster scatter plot and a per-cluster summary table, and exports the results.

## Problem statement
Given customer records with no ready-made group labels, discover groups of similar customers
(unsupervised learning) and describe each group using its real statistics.

## Objectives
- Apply data preprocessing (duplicates, missing values, standardisation).
- Apply K-Means clustering with scikit-learn.
- Use the Elbow Method to help choose K.
- Present results in a simple Tkinter GUI and export them as CSV.
- Test the code with automated unit tests.

## Main features
- Dashboard, Dataset & Cleaning, Clustering & Visualisation, and Export tabs.
- Choose K from 2 to 8; clustering is recomputed every time you click **Run Clustering**.
- If you change K after clustering, a red message tells you the results are stale until you re-run.
- Elbow chart (inertia vs K) and AnnualIncome-vs-SpendingScore scatter plot, both embedded in the app.
- Cluster summary: size, % of customers, six averages, and a short profile generated from those averages.
- CSV export of segments, cluster summary and elbow results.
- 21 automated unit tests.

## Technology stack
Python 3.10+, Tkinter/ttk, pandas, NumPy, scikit-learn, Matplotlib, unittest.

## Folder structure
```
customer_segmentation_system/
  main.py            entry point (GUI, --generate-data, --test)
  gui.py             Tkinter interface
  requirements.txt
  README.md  report.md  viva_prep.md
  run_app.bat        double-click launcher for Windows
  .gitignore
  data/customers.csv       saved synthetic dataset
  src/
    __init__.py
    data_loader.py         dataset generation + saving
    preprocessing.py       loading, validation, cleaning, scaling
    clustering.py          K-Means, K validation, cluster summaries
    evaluation.py          Elbow Method
    export_results.py      CSV export
  tests/
    __init__.py
    test_project.py        unittest suite
  outputs/.gitkeep         exported CSV files go here
```

## Dataset description
`data/customers.csv` is synthetic (made with NumPy using a fixed seed, so it is reproducible). It has 500
base customers from six different behaviour patterns, plus a few deliberate problems for the cleaning step:
10 missing numeric values and 8 duplicate rows (508 rows in the file).

| Column | Meaning |
|---|---|
| CustomerID | Identifier (NOT used for clustering) |
| Age | Age in years |
| AnnualIncome | Income in thousands of the currency unit |
| SpendingScore | Integer 1-100 |
| PurchaseFrequency | Approximate purchases per year |
| AverageOrderValue | Average value per purchase |
| MembershipYears | Years as a customer |

The clustering features are the six columns except CustomerID. There is no target column: this is unsupervised learning.

## Data preprocessing
1. Load the CSV and check that the required columns exist (the file is never modified).
2. Count missing values and duplicate rows.
3. Remove exact duplicate records.
4. Treat invalid values (text, impossible numbers such as a negative income) as missing.
5. Fill missing values with the column **median** (robust to outliers).
6. Standardise the six features with `StandardScaler`. The scaler is fitted on the cleaned data of the
   current run, and the same scaled matrix is used for the Elbow Method and K-Means.

## K-Means explanation
K-Means picks K starting centres, assigns each customer to the nearest centre, moves each centre to the mean
of its customers, and repeats until the centres stop moving. We use `random_state=42` (reproducible) and
`n_init=10` (10 random starts; the best result is kept). Cluster numbers (0, 1, 2...) are arbitrary names,
not rankings.

## Elbow Method explanation
For K = 2 to 8 we fit K-Means and record the **inertia** (sum of squared distances from points to their
cluster centre). Inertia always falls as K grows. The "elbow" is where the drop becomes small. This
**helps estimate** a useful K but does not prove one objectively correct K; you choose K yourself.

## How to run on Windows (PowerShell)
Install Python 3.10+ from python.org first (tick "Add Python to PATH").

```powershell
cd path\to\customer_segmentation_system
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```
- Installing the packages needs internet once. After that the app works fully offline.
- If `python` is not found, use `py` instead (for example `py -m venv .venv`).
- If PowerShell blocks the activation script, open **Command Prompt** instead and run
  `.venv\Scripts\activate.bat` (no security settings need to change). Or skip activation and call
  `.venv\Scripts\python.exe main.py` directly.
- **VS Code:** File > Open Folder > choose `customer_segmentation_system`, open a terminal (Ctrl+`), run the commands above.
- **Double-click launch:** after the setup above, double-click `run_app.bat`. If `.venv` is missing it shows setup instructions.

## How to run the tests
```powershell
python main.py --test
```

## How to regenerate the sample data
The app uses the saved `data/customers.csv` and only creates it if it is missing. To replace it on purpose
(this overwrites the file), run:
```powershell
python main.py --generate-data
```
The **Regenerate sample data** button in the Dataset tab does the same after asking for confirmation.

## How to export results
Open the **Export Results** tab. Run clustering first for the segments and summary files, and run the Elbow
Method for the elbow file. A save dialog opens in the `outputs/` folder with these suggested names:
`customer_segments.csv`, `cluster_summary.csv`, `elbow_results.csv`.

## Free deployment / sharing
This is a **desktop** (Tkinter) app, so it cannot be hosted on a website the way a web app can. Free ways to
share it:
1. **GitHub (free, recommended).** Create a free GitHub repository, push this folder (`.gitignore` is included),
   and anyone can clone it and follow the Windows steps above. Put the repository link in your report.
2. **GitHub Releases (free).** Zip the project folder and attach it to a Release so your teacher can download it.
3. **Optional:** a Windows `.exe` with PyInstaller is possible, but file paths need small changes for a
   packaged app, so treat it as an extra and test it before relying on it.
4. **A live web link** (for example Streamlit Community Cloud, which is free) would need a separate web
   version of the interface, because Tkinter does not run in a browser. The `src/` modules can be reused as they are.

## Known limitations
- The data is synthetic, so the segments only demonstrate the method; they say nothing about real customers.
- K-Means assumes roughly round, similar-sized clusters and is affected by outliers and by the starting centres.
- The Elbow Method can be ambiguous; there is no single "correct" K.
- The scatter plot shows only two of the six features.
- Cluster profiles are interpretations of averages, not facts about customer personality or future behaviour.

## Screenshot instructions (Windows)
Do not use fake screenshots; capture your own from the running app.
1. Launch the app (`python main.py` or `run_app.bat`).
2. **Dashboard** tab: screenshot (Figure 1).
3. **Dataset & Cleaning** tab: screenshot showing the table and data-quality line (Figure 2).
4. **Clustering & Visualisation** tab: click **Run Elbow Method**, then pick a K and click **Run Clustering**.
   Screenshot the elbow chart (Figure 3), the scatter plot (Figure 4) and the cluster summary table (Figure 5).
5. **Export Results** tab: export the three files, then open `customer_segments.csv` in Excel and screenshot it (Figure 6).
6. Capture with **Win + Shift + S** (Snipping Tool), save as PNG, and insert into `report.md`/your Word report.

## SegmentIQ interactive business dashboard

The web interface is built with Streamlit and Plotly, with a responsive light workspace and a dark navigation sidebar. It includes:

- **Executive overview** — customer KPIs, data-readiness checks, and portfolio-level charts.
- **Segment explorer** — interactive segment map, segment-size mix, average profiles, and experiment ideas.
- **Customer explorer** — filter customer records by segment, customer ID, income, spending score, and order value; export filtered records.
- **Data & quality** — upload a CSV, validate the required schema, inspect data quality, download a template, and export cleaned data.
- **Reports & export** — export segmented customers, segment profiles, model metrics, data-quality reports, elbow results, and an analysis report.

### Start the dashboard

From the project folder, run:

```bash
python -m pip install -r requirements.txt
python main.py
```

On Windows, `run_app.bat` can be used after creating `.venv` and installing the requirements as described in the batch file. The dashboard opens in your browser and runs locally. Uploaded data is held in the current app session; this project does not include user accounts, a database, role-based permissions, or a production hosting configuration.

**Business-use note:** the included customer dataset is synthetic. For a real deployment, confirm data definitions and units, add appropriate authentication and access controls, and validate suggested marketing experiments against business outcomes before operational use.
