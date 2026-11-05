"""Minimal non-Streamlit sanity check of saved artifacts and sample predictions."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
summary = json.loads((ART / "summary.json").read_text(encoding="utf-8"))
data = pd.read_csv(ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
features = data.drop(columns=["customerID", "Churn"])
assert len(data) == summary["dataset_n"] == 7043
assert summary["n_test"] == 1409
for fname in ["model.joblib", "rf_reference.joblib"]:
    model = joblib.load(ART / fname)
    proba = model.predict_proba(features.iloc[:3])[:, 1]
    assert len(proba) == 3 and np.isfinite(proba).all(), fname
    assert ((proba >= 0) & (proba <= 1)).all(), fname
    print(f"OK: {fname}: probabilities={proba.round(4).tolist()}")
print("OK: dataset, exported pipelines, and summary.json")
