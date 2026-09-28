"""Package the ten fitted models needed for manuscript/SI comparisons."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "rf_rfe_10models"
MODEL_DIR = ROOT / "models" / "rf_rfe_10models"
OUTPUT = ROOT / "models" / "ten_model_comparison_bundle.joblib"


def slug(name):
    return name.lower().replace(" ", "")


def main():
    selected = json.loads((REPORT / "rfecv_selected_features.json").read_text(encoding="utf-8"))
    performance = pd.read_csv(REPORT / "ten_model_oof_performance.csv").sort_values("r2", ascending=False)
    order = performance["model"].tolist()
    models = [{"name": name, "model": joblib.load(MODEL_DIR / f"{slug(name)}_best.joblib")} for name in order]
    bundle = {
        "selected_features": selected,
        "models": models,
        "model_order": order,
        "final_model": "ExtraTrees",
        "descriptor_backend": "pymatgen + matminer standard composition featurizers",
        "feature_selection": "RF-RFECV maximum mean five-fold R2",
        "hyperparameter_search": "exhaustive discrete GridSearchCV",
        "stacking_used_as_final_model": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, OUTPUT, compress=3)
    print(json.dumps({k: v for k, v in bundle.items() if k != "models"}, indent=2))
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
