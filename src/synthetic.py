"""
Synthetic student generator (Gaussian copula) for student-mat.csv.

Preserves each column's marginal distribution and the pairwise rank correlations of the real data.
It does NOT add real information: every synthetic row is a statistical echo of the 395 real students.
Use synthetic rows for load-testing, demos, and (carefully) training augmentation.
NEVER evaluate a model on synthetic rows; report metrics on real held-out students only.

Usage (from project root):
  python -m src.synthetic --input data/raw/student-mat.csv --rows 5000 --seed 42 --out data/synthetic/student-mat-synthetic-5000.csv
"""
import argparse, csv
import numpy as np, pandas as pd
from scipy.stats import norm, rankdata


def encode(df):
    """Integer-code every column. Nominal columns are ordered by frequency (arbitrary but fixed)."""
    codes, labels = {}, {}
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            order = df[c].value_counts().index.tolist()
            labels[c] = order
            codes[c] = df[c].map({v: i for i, v in enumerate(order)}).to_numpy()
        else:
            codes[c] = df[c].to_numpy()
    return pd.DataFrame(codes), labels


def fit_copula(codes):
    n = len(codes)
    Z = np.column_stack([norm.ppf((rankdata(codes[c], method="average") - 0.5) / n) for c in codes.columns])
    R = np.corrcoef(Z.T)
    w, v = np.linalg.eigh(R)                     # force positive semi-definite
    R = (v * np.clip(w, 1e-6, None)) @ v.T
    d = np.sqrt(np.diag(R)); R = R / np.outer(d, d)
    sorted_cols = {c: np.sort(codes[c].to_numpy()) for c in codes.columns}
    return R, sorted_cols


def sample(R, sorted_cols, columns, labels, rows, rng):
    Zn = rng.multivariate_normal(np.zeros(len(columns)), R, size=rows, method="eigh")
    U = norm.cdf(Zn)
    out = {}
    for j, c in enumerate(columns):
        s = sorted_cols[c]
        x = s[np.minimum((U[:, j] * len(s)).astype(int), len(s) - 1)]
        out[c] = np.array(labels[c])[x.astype(int)] if c in labels else x
    return pd.DataFrame(out, columns=columns)


def generate(real, rows, seed):
    codes, labels = encode(real)
    R, sc = fit_copula(codes)
    return sample(R, sc, list(real.columns), labels, rows, np.random.default_rng(seed))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--rows", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    real = pd.read_csv(a.input, sep=";")
    syn = generate(real, a.rows, a.seed)
    assert list(syn.columns) == list(real.columns)
    syn.to_csv(a.out, sep=";", index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"wrote {len(syn)} synthetic rows to {a.out}")
