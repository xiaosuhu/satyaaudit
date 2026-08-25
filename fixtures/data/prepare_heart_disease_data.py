"""Prepare UCI Heart Disease (Cleveland) dataset for use in SatyaRepro testing.

Source: processed.cleveland.data (comma-separated, no header, '?' = missing)
Output: heart_disease.csv
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

_DIR = Path(__file__).parent
_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target",
]

df = pd.read_csv(
    _DIR / "processed.cleveland.data",
    header=None,
    names=_COLUMNS,
    na_values="?",
)

out = _DIR / "heart_disease.csv"
df.to_csv(out, index=False)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")
print(f"Missing : {df.isna().sum().sum()} total")
print()
print(df.isna().sum()[df.isna().sum() > 0].to_string())
print(f"\nSaved to: {out.resolve()}")
