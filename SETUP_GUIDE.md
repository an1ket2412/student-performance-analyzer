# Student Performance Analyzer: Complete Setup and Run Guide

This zip is the whole project: code, data, saved model, notebooks, docs, the synthetic-data extension
and your project synopsis. Follow the steps in order. Every command is run from the project folder.

---

## STEP 1: What you need
- Python 3.10 or newer. Check: `python --version` (Windows: `py --version`).
- An internet connection for the one-time package install.
- About 200 MB of free space once the packages are installed.

## STEP 2: Unzip
Extract `Student_Performance_Analyzer_Complete.zip` (right-click -> Extract All). You get a folder
called `student-performance-analyzer`. Do not run anything from inside the zip viewer.

## STEP 3: Open a terminal in the project folder
- **Windows:** open the folder, click the address bar, type `powershell`, press Enter.
- **VS Code:** File -> Open Folder -> pick `student-performance-analyzer`, then Terminal -> New Terminal.
- **macOS/Linux:** `cd path/to/student-performance-analyzer`.

Check you are in the right place: listing the folder (`dir` or `ls`) must show `src`, `app`, `data`,
`models`, `notebooks`.

## STEP 4: Create a virtual environment
**Windows (PowerShell):**
```
python -m venv .venv
.venv\Scripts\activate
```
If activation is blocked, run `Set-ExecutionPolicy -Scope Process Bypass` and activate again.

**macOS / Linux:**
```
python3 -m venv .venv
source .venv/bin/activate
```
Your prompt now starts with `(.venv)`.

## STEP 5: Install the packages
```
pip install -r requirements-dev.txt
```
This takes a few minutes. Then check:
```
python -c "import pandas, sklearn, streamlit; print('OK')"
```

## STEP 6: Check the data is in place
- `data/raw/student-mat.csv` (395 real students, never edit it)
- `data/processed/student_clean.csv`
- `data/synthetic/student-mat-synthetic-5000.csv` (5,000 synthetic rows)
- `models/final_model.joblib` (saved model)

## STEP 7: Clean the data
```
python src/cleaning.py
```
Re-creates `data/processed/student_clean.csv` from the raw file.

## STEP 8: Check feature engineering
```
python src/features.py
```

## STEP 9: Train and evaluate the models
```
python src/training.py
```
Compares six models, fits the final Logistic Regression and re-saves `models/final_model.joblib`.
Expected on the real test set (79 students): accuracy about 58.2%, at-risk recall about 42.3%,
ROC-AUC about 0.602. If you see a scikit-learn version warning in the dashboard, this step fixes it.

## STEP 10: Run the dashboard
```
streamlit run app/app.py
```
Open http://localhost:8501 if it does not open by itself. If Streamlit asks for an email, press Enter.
Stop it with Ctrl+C.

## STEP 11: Run the notebooks (optional)
```
jupyter notebook
```
Open `notebooks/` and run 01_profile, 02_cleaning, 03_eda, 04_features, 05_modelling in order
(Kernel -> Restart and Run All).

## STEP 12: Run the synthetic-data scripts
Always use `python -m ...` for these (not `python scripts/...`).

Generate synthetic data (any size; change `--rows`):
```
python -m src.synthetic --input data/raw/student-mat.csv --rows 5000 --seed 42 --out data/synthetic/student-mat-synthetic-5000.csv
```
Check how close it is to the real data (writes `docs/synthetic_fidelity.json`):
```
python -m scripts.evaluate_synthetic
```
Check whether synthetic rows help (scored on real students only; takes a minute or two; writes
`docs/augmentation_results.json`):
```
python -m scripts.augmentation_experiment
```

---

## Folder map
```
src/            cleaning.py, features.py, training.py, synthetic.py
scripts/        evaluate_synthetic.py, augmentation_experiment.py
app/app.py      Streamlit dashboard
data/raw/       original UCI file (do not edit)
data/processed/ cleaned data
data/synthetic/ synthetic data + its README
models/         saved model
notebooks/      01-05 analysis notebooks
docs/           charter, data dictionary, report, synthetic results (json)
submission/     project synopsis (.docx)
SETUP_GUIDE.md  this file
```

## Troubleshooting
| Problem | Fix |
|---|---|
| `ModuleNotFoundError: src` | You are not in the project folder, or you ran a script file directly. Use `python -m scripts.<name>`. |
| `streamlit` or `jupyter` not found | The virtual environment is not active. Repeat Step 4's activate command. |
| `File not found: data/raw/student-mat.csv` | Run commands from the project folder. Paths are relative. |
| scikit-learn version warning in the app | Run `python src/training.py` once (Step 9). |
| `pip install` fails | Update pip: `python -m pip install --upgrade pip`, then retry. |

## Rules for the synthetic data
1. Never evaluate a model on synthetic rows; test on held-out REAL students only.
2. The shipped 5,000-row file was generated from all 395 real rows, so use it for demos and load tests.
   If you train with synthetic rows before an evaluation, generate them from the training split only.
3. Do not report EDA findings from synthetic data as real findings.