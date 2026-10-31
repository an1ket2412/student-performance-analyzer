"""Does adding synthetic training rows help?  Always evaluated on REAL students only.
Synthetic rows are generated from the TRAINING rows of each split, never from test rows.
Run from project root:  python -m scripts.augmentation_experiment   (writes docs/augmentation_results.json)"""
import json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from src.cleaning import clean_student_data, load_raw_data
from src.features import engineer_features, get_feature_target_split, split_data
from src.synthetic import generate
from src.training import select_and_fit_final_model

raw = load_raw_data()
X, y = get_feature_target_split(engineer_features(clean_student_data(raw)))

def prep(df):
    return get_feature_target_split(engineer_features(clean_student_data(df)))

def metrics(model, Xte, yte):
    p_pass = model.predict_proba(Xte)[:, 1]
    out = {"auc": roc_auc_score(yte, p_pass)}
    for cut, name in ((0.5, "risk0.50"), (0.6, "risk0.40")):     # P(pass) cut-off = 1 - risk threshold
        pred = (p_pass >= cut).astype(int)
        out[f"accuracy_{name}"] = accuracy_score(yte, pred)
        out[f"at_risk_recall_{name}"] = recall_score(yte, pred, pos_label=0)
        out[f"at_risk_precision_{name}"] = precision_score(yte, pred, pos_label=0, zero_division=0)
    return out

def fit_eval(Xtr, ytr, Xte, yte, n_syn, seed):
    if n_syn:
        Xs, ys = prep(generate(raw.loc[Xtr.index].reset_index(drop=True), n_syn, seed))
        Xtr = pd.concat([Xtr, Xs], ignore_index=True); ytr = pd.concat([ytr, ys], ignore_index=True)
    return metrics(select_and_fit_final_model(Xtr, ytr), Xte, yte)

Xtr, Xte, ytr, yte = split_data(X, y)
single = {str(n): fit_eval(Xtr, ytr, Xte, yte, n, seed=7) for n in (0, 1000, 5000)}
print("Single project split (n_test = %d):" % len(yte)); print(pd.DataFrame(single).T.round(3).to_string())

rows0, rows1 = [], []
for rep in range(5):
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y)):
        a = fit_eval(X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te], 0, 0)
        b = fit_eval(X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te], 5000, rep * 10 + k)
        rows0.append(a); rows1.append(b)
A, B = pd.DataFrame(rows0), pd.DataFrame(rows1)
diff = B - A
paired = {m: {"real_only_mean": float(A[m].mean()), "augmented_mean": float(B[m].mean()),
              "mean_diff": float(diff[m].mean()), "folds_improved_pct": float((diff[m] > 0).mean() * 100),
              "wilcoxon_p": float(wilcoxon(diff[m]).pvalue)} for m in A.columns}
print("\n5x5 repeated CV, real-only vs +5000 synthetic (paired over 25 folds):")
print(pd.DataFrame(paired).T.round(3).to_string())
print("\nNote: folds overlap across repeats, so p-values are optimistic.")
json.dump({"single_split": single, "repeated_cv_paired": paired}, open("docs/augmentation_results.json", "w"), indent=2)
