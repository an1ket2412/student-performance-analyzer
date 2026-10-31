# Student Performance Analyzer

**Early Academic Risk Identification & Intervention System**  
*Developed as part of the HCL Career Shaper / Data Science Training Program*

---

## 1. Project Title & Overview

**Student Performance Analyzer** is an end-to-end Machine Learning and analytical system built to identify secondary students at risk of academic failure in mathematics at the start of or early in the academic semester.

By systematically distinguishing between pre-outcome behavioral/socioeconomic signals and mid-term exam scores, the system enforces **zero target leakage**, ensuring that teachers receive actionable alerts early enough to deliver pedagogical support, tutoring, and counseling before final examinations.

---

## 2. Problem Statement

Secondary and higher-secondary educators frequently discover that a student is struggling academically only after midterm examinations or final report cards are published. At that late stage, corrective interventions (supplemental tutoring, study coaching, parental engagement) are far less effective because foundational learning gaps have compounded. 

This system solves the core pedagogical challenge:
> *Empower educators with an automated, pre-outcome early-warning system that flags at-risk students at term commencement, allowing timely, targeted academic support.*

---

## 3. Objective

1. Analyze educational and behavioral drivers of secondary mathematics performance.
2. Build, evaluate, and tune machine learning classifiers prioritizing **Recall on the Fail / At-Risk class**.
3. Enforce strict **quarantining of target-leakage attributes** (excluding $G1$ and $G2$).
4. Deploy an interactive, production-ready **Streamlit dashboard** for cohort visual analytics and real-time individual risk profiling.

---

## 4. Target Users

- **Class Teachers:** Receive early alerts to tailor classroom instruction, check in on homework completion, and assign peer mentors.
- **Academic Coordinators & Counselors:** Monitor cohort-wide risk distributions, plan remedial tutoring sessions, and allocate institutional support.
- **School Administrators:** Audit academic health and ensure equitable educational outcomes across student demographics.

---

## 5. Dataset

- **Source:** [UCI Machine Learning Repository — Student Performance Data Set](https://archive.ics.uci.edu/dataset/320/student+performance)
- **Course Subject:** Mathematics (`student-mat.csv`)
- **Dimensions:** 395 student observations $\times$ 33 attributes
- **Delimiter:** Semicolon (`;`)
- **Grading Scale:** Portuguese 0 to 20 integer scale
  - **Passing:** Final grade $G3 \ge 10$ (67.1% of cohort)
  - **Fail / At-Risk:** Final grade $G3 < 10$ (32.9% of cohort)
- **Target Leakage Precaution:** $G1$ (first period grade) and $G2$ (second period grade) are strongly correlated with $G3$ ($r > 0.80$), but are strictly **quarantined and excluded** from feature inputs to preserve true early-warning capability.

---

## 6. Data Preparation & Cleaning

- Encapsulated in reusable module `src/cleaning.py`.
- Verified 0 null values and 0 duplicate records across the raw data.
- Stripped whitespace from categorical attributes and validated boundary ranges ($15 \le \text{age} \le 22$, $0 \le \text{absences} \le 93$).
- Derived standardized binary targets: `passed` ($G3 \ge 10$) and `at_risk` ($G3 < 10$).
- Persisted cleaned dataset to `data/processed/student_clean.csv`.

---

## 7. Exploratory Data Analysis (EDA)

Key insights uncovered in `notebooks/03_eda.ipynb`:
1. **Past Academic Failures:** The single strongest pre-outcome predictor ($r = -0.36$). Zero past failures yields a 71.8% pass rate, dropping to $< 38\%$ with $\ge 1$ failure.
2. **Weekly Study Time:** Students dedicating $>10$ hours/week achieve an 81.5% pass rate compared to 56.7% for those studying $<2$ hours.
3. **Parental Education:** Students whose parents attained higher education pass at 77.2% vs. 55.6% for primary education.
4. **Behavioral Friction:** Elevated workday alcohol consumption (`Dalc` $\ge 4$) correlates with a 44–55% pass rate.
5. **Digital Equity:** Home internet access provides an ~15 percentage-point advantage in pass rate (69.3% vs. 53.7%).

---

## 8. Feature Engineering

Encapsulated in `src/features.py`:
- `parent_edu_avg`: Combined arithmetic mean of mother's and father's education $(Medu + Fedu) / 2$.
- `total_alc`: Weighted weekly alcohol consumption $(5 \times Dalc + 2 \times Walc) / 7$.
- `support_index`: Sum of school support, family support, and paid tutoring indicators (0 to 3).
- `absences_capped`: Absences clipped at 30 days to limit leverage of extreme outliers.
- `attendance_rate_est`: Estimated session attendance percentage assuming ~90 instructional periods.
- Standardized scaling via `StandardScaler` for all 18 numeric features and one-hot encoding via `OneHotEncoder(handle_unknown='ignore')` for all 17 categorical features using Scikit-learn `ColumnTransformer`.

---

## 9. Model Development & Comparison

Models trained on training partition (316 rows, 80%) using 5-fold Stratified Cross-Validation:

| Model Candidate | CV At-Risk Recall | CV At-Risk Precision | CV At-Risk F1 | CV Accuracy | CV ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Balanced)** | **56.8%** | **50.2%** | **53.1%** | **67.7%** | **0.695** |
| Random Forest (Balanced) | 51.9% | 54.3% | 52.8% | 69.6% | 0.712 |
| Decision Tree (Balanced) | 45.1% | 53.7% | 48.2% | 68.3% | 0.634 |
| Logistic Regression (Baseline) | 35.6% | 50.5% | 41.6% | 67.7% | 0.681 |
| Gradient Boosting | 32.8% | 54.5% | 40.3% | 69.0% | 0.657 |
| Majority Class Baseline | 0.0% | 0.0% | 0.0% | 67.1% | 0.500 |

*Tuning Decision:* **Logistic Regression Balanced** was selected as the final production model due to its superior recall on the minority at-risk class and clear interpretability.

---

## 10. Evaluation on Untouched Test Set

Final evaluation on the untouched holdout test partition (79 students, 20%):

- **Accuracy:** 58.2%
- **At-Risk Recall (Class 0):** 42.3% (at default 0.50 threshold) $\rightarrow$ **61.5%** at calibrated threshold 0.40
- **At-Risk Precision:** 37.9% $\rightarrow$ 40.0% at threshold 0.40
- **At-Risk F1-Score:** 40.0% $\rightarrow$ 48.5% at threshold 0.40
- **ROC-AUC Score:** 0.602
- **Confusion Matrix (at default 0.50 cutoff):**
  - True At-Risk Identified (True Negatives): 11
  - Missed At-Risk (False Negatives from risk perspective): 15
  - False Alarms (False Positives from risk perspective): 18
  - True Passing (True Positives): 35

---

## 11. Streamlit Dashboard

The interactive application (`app/app.py`) provides:
- **Cohort Overview & KPI Cards:** Live metrics on total students, pass rate, average absences, and at-risk count.
- **Visual Analytics:** Interactive multi-select filters on school, study time, gender, and address type with automated chart updates.
- **Individual Risk Assessment Form:** Accepts pre-outcome student inputs, runs the exact training feature engineering pipeline, and outputs:
  - Predicted class (Pass vs. At-Risk)
  - Calibrated risk probability
  - Risk Tier:
    - **Low Risk:** $< 35\%$
    - **Moderate Risk:** $35\% \le \text{Risk} < 55\%$
    - **High Risk:** $\ge 55\%$
  - Concrete pedagogical action guidance for educators.

---

## 12. Project Structure

```text
student-performance-analyzer/
│
├── data/
│   ├── raw/
│   │   └── student-mat.csv
│   └── processed/
│       └── student_clean.csv
│
├── notebooks/
│   ├── 01_profile.ipynb
│   ├── 02_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_features.ipynb
│   └── 05_modelling.ipynb
│
├── src/
│   ├── __init__.py
│   ├── cleaning.py
│   ├── features.py
│   └── training.py
│
├── models/
│   └── final_model.joblib
│
├── app/
│   └── app.py
│
├── docs/
│   ├── project_charter.md
│   ├── data_source.md
│   ├── data_dictionary.md
│   └── report.md
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 13. Installation & Environment Setup

### Prerequisites
- Python 3.9+ (tested on Python 3.12 and 3.13)
- Git

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/example/student-performance-analyzer.git
   cd student-performance-analyzer
   ```

2. **Create and activate a virtual environment:**
   - **macOS / Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - **Windows:**
     ```cmd
     python -m venv .venv
     .venv\Scripts\activate
     ```

3. **Install project dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 14. How to Run

### Execute Data Cleaning:
```bash
python src/cleaning.py
```

### Execute Feature Engineering Verification:
```bash
python src/features.py
```

### Train and Evaluate Models:
```bash
python src/training.py
```

### Launch Interactive Streamlit Dashboard:
```bash
streamlit run app/app.py
```
Then open `http://localhost:8501` in your web browser.

### Launch Jupyter Notebooks:
```bash
jupyter lab
# or
jupyter notebook
```
Navigate to `notebooks/` and run notebooks `01` through `05` sequentially.

---

## 15. Limitations & Ethical Considerations

1. **Sample Size:** $N = 395$ observations from two regional Portuguese secondary schools. Findings may not directly generalize to distinct national school systems without local recalibration.
2. **Pre-Outcome Feasibility:** Without midterm exam marks ($G1, G2$), predicting academic risk purely from demographic and enrollment attributes is inherently noisy.
3. **Non-Causal Interpretation:** Identified features represent statistical associations, not definitive causes of student failure.
4. **Responsible Usage:** The system is an early-support alert mechanism; it must never be utilized for automated punitive or disciplinary decisions.

---

## 16. Future Scope

1. Incorporate fine-grained weekly homework submission and Learning Management System (LMS) digital engagement telemetry.
2. Expand to multi-subject longitudinal tracking (Mathematics, Portuguese, Science).
3. Implement Bayesian updating to adjust student risk continuously as the term progresses.

---

## 17. Synthetic Data Extension

To work around the small sample (395 real students), the project includes a **synthetic** dataset of 5,000 rows generated from `data/raw/student-mat.csv` with a Gaussian copula (`src/synthetic.py`). It contains no real students. The synthetic rows are for training augmentation, app demos and load testing only, and are **never** used for evaluation.

| Item | Location |
|---|---|
| Generator | `src/synthetic.py` |
| 5,000-row synthetic file | `data/synthetic/student-mat-synthetic-5000.csv` |
| Fidelity / privacy checks | `scripts/evaluate_synthetic.py` -> `docs/synthetic_fidelity.json` |
| Augmentation experiment (scored on real students only) | `scripts/augmentation_experiment.py` -> `docs/augmentation_results.json` |
| Details and rules | `data/synthetic/README_SYNTHETIC.md` |

Key findings (5x5 repeated CV, real held-out students): ROC-AUC 0.679 -> 0.693 (not significant, p = 0.22); at-risk recall at the 0.40 threshold 70.2% -> 79.7% with similar precision. Synthetic rows mainly shift the operating point; they add no new real information.

Run from the project root with `python -m src.synthetic ...`, `python -m scripts.evaluate_synthetic` and `python -m scripts.augmentation_experiment`. See `SETUP_GUIDE.md` for every step.
