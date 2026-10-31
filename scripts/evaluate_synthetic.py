"""Fidelity / privacy / train-on-synthetic-test-on-real checks.  Run from project root:
   python -m scripts.evaluate_synthetic        (writes docs/synthetic_fidelity.json)"""
import json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.synthetic import generate, encode

REAL = "data/raw/student-mat.csv"
SYN = "data/synthetic/student-mat-synthetic-5000.csv"
real = pd.read_csv(REAL, sep=";"); syn = pd.read_csv(SYN, sep=";")
isnum = pd.api.types.is_numeric_dtype
res = {}

tv = {}
for c in real.columns:
    p = real[c].value_counts(normalize=True); q = syn[c].value_counts(normalize=True)
    idx = p.index.union(q.index)
    tv[c] = 0.5 * (p.reindex(idx, fill_value=0) - q.reindex(idx, fill_value=0)).abs().sum()
tv = pd.Series(tv).sort_values(ascending=False)
res["tv_mean"] = round(float(tv.mean()), 3); res["tv_worst"] = [tv.index[0], round(float(tv.iloc[0]), 3)]

cr, _ = encode(real); cs, _ = encode(syn)
Rr, Rs = spearmanr(cr).correlation, spearmanr(cs).correlation
iu = np.triu_indices_from(Rr, 1)
res["corr_mean_abs_diff"] = round(float(np.nanmean(np.abs(Rr[iu] - Rs[iu]))), 3)
res["corr_max_abs_diff"] = round(float(np.nanmax(np.abs(Rr[iu] - Rs[iu]))), 3)
res["fail_rate_real_vs_syn"] = [round(float((real.G3 < 10).mean()), 3), round(float((syn.G3 < 10).mean()), 3)]

R_, S_ = real.to_numpy(), syn.to_numpy()
real_set = set(map(tuple, R_))
res["exact_copies_of_real_rows"] = int(sum(tuple(r) in real_set for r in S_))
def min_hamming(A, B, same=False):
    out = []
    for k in range(0, len(A), 250):
        d = (A[k:k+250, None, :] != B[None, :, :]).sum(2)
        if same:
            for t in range(d.shape[0]): d[t, k + t] = 99
        out.append(d.min(1))
    return np.concatenate(out)
d_syn = min_hamming(S_, R_)
res["syn_rows_within_2_columns_of_a_real_row_pct"] = round(float((d_syn <= 2).mean() * 100), 2)

feats = [c for c in real.columns if c not in ("G1", "G2", "G3")]
y = (real.G3 < 10).astype(int).to_numpy()
def pipe():
    num = [c for c in feats if isnum(real[c])]; cat = [c for c in feats if not isnum(real[c])]
    return Pipeline([("p", ColumnTransformer([("n", StandardScaler(), num),
                                              ("c", OneHotEncoder(handle_unknown="ignore"), cat)])),
                     ("m", LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5))])
auc_rr, auc_sr, n = [], [], 0
for rep in range(5):
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(real, y):
        rtr, rte = real.iloc[tr], real.iloc[te]
        auc_rr.append(roc_auc_score(y[te], pipe().fit(rtr[feats], y[tr]).predict_proba(rte[feats])[:, 1]))
        sy = generate(rtr, 5000, seed=rep * 10 + n); n += 1
        auc_sr.append(roc_auc_score(y[te], pipe().fit(sy[feats], (sy.G3 < 10).astype(int)).predict_proba(rte[feats])[:, 1]))
res["tstr_auc_real_to_real"] = [round(float(np.mean(auc_rr)), 3), round(float(np.std(auc_rr)), 3)]
res["tstr_auc_synthetic_to_real"] = [round(float(np.mean(auc_sr)), 3), round(float(np.std(auc_sr)), 3)]
json.dump(res, open("docs/synthetic_fidelity.json", "w"), indent=2)
print(json.dumps(res, indent=2))
