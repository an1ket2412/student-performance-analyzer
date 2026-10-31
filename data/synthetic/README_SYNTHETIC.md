# Synthetic extension of student-mat.csv

**This data is synthetic.** It was generated from the 395 real rows of UCI Student Performance
(Mathematics) with a Gaussian copula (`src/synthetic.py`, seed 42). It contains no real students.
Cite the original dataset (Cortez & Silva, 2008; UCI id 320, CC BY 4.0) and state that these rows
were derived from it. Never mix this folder with `data/raw/`.

## Files
- `student-mat-synthetic-5000.csv` - 5,000 rows, same 33 columns and `;` separator as the original.
- Regenerate any size: `python -m src.synthetic --input data/raw/student-mat.csv --rows N --seed S --out data/synthetic/<name>.csv`
- Checks: `python -m scripts.evaluate_synthetic` (-> `docs/synthetic_fidelity.json`);
  `python -m scripts.augmentation_experiment` (-> `docs/augmentation_results.json`).

## Fidelity (395 real rows vs 5,000 synthetic)
- Column distributions: mean total-variation distance 0.010 (worst: absences, 0.030).
- Pairwise rank correlations: mean absolute difference 0.031, worst pair 0.163. Correlations are
  weakened (failures-G3 -0.36 -> -0.24, Dalc-Walc 0.64 -> 0.50, Medu-Fedu 0.63 -> 0.52, G1-G3 0.88 -> 0.84).
- Fail rate (G3 < 10): 32.9% real vs 31.4% synthetic.
- Privacy: 0 exact copies of real rows; no synthetic row within 2 columns of a real row.
- Train on synthetic only, test on held-out real students (Stage-A features, 5x5 CV): AUC 0.679 vs 0.678
  when training on real data - synthetic rows add no information beyond the real rows.

## Does augmentation help? (always scored on REAL held-out students; synthetic made from training rows only)
- 5x5 repeated CV, real-only vs real + 5,000 synthetic (paired, 25 folds): ROC-AUC 0.679 -> 0.693
  (+0.015, not significant, Wilcoxon p = 0.22); at-risk recall at the 0.40 threshold 70.2% -> 79.7%
  (+9.5 points, 84% of folds improved), precision 43.3% -> 42.2%, accuracy 59.9% -> 57.3%.
  Folds overlap across repeats, so p-values are optimistic.
- Single 79-student project split (seed 7): AUC 0.602 -> 0.734, recall at 0.40 61.5% -> 88.5%.
  This is one small split and far noisier than the repeated CV; treat it as indicative only.
- Reading: augmentation mainly shifts the operating point toward flagging more at-risk students;
  there is no reliable gain in ranking quality.

## Rules
1. Never evaluate a model on synthetic rows; test only on held-out REAL students.
2. If synthetic rows are used for training, generate them from the training split only (the
   delivered 5,000-row file was generated from all 395 rows, so use it for demos and load tests,
   not for training before an evaluation).
3. Do not report EDA findings from synthetic data as real findings.
4. Good uses: app demos and load testing, batch-upload testing, pipeline development.
