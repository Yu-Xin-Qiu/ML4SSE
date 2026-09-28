"""Package the selected RF-RFECV ExtraTrees model as the final predictor."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "rf_rfe_10models"
SOURCE_MODEL = ROOT / "models" / "rf_rfe_10models" / "extratrees_best.joblib"
OUTPUT = ROOT / "models" / "final_extratrees_composition_model.joblib"


def main():
    selected = json.loads((REPORT / "rfecv_selected_features.json").read_text(encoding="utf-8"))
    model = joblib.load(SOURCE_MODEL)
    metrics = pd.read_csv(REPORT / "ten_model_oof_performance.csv")
    row = metrics.loc[metrics["model"].eq("ExtraTrees")].iloc[0]
    bundle = {
        "model_name": "ExtraTrees",
        "model": model,
        "selected_features": selected,
        "descriptor_backend": "pymatgen + matminer standard composition featurizers",
        "feature_selection": "RF-RFECV maximum mean five-fold R2",
        "validation": "KFold(n_splits=5, shuffle=True, random_state=42)",
        "reported_oof_r2": float(row["r2"]),
        "reported_oof_mae_log10": float(row["mae_log10"]),
        "reported_oof_rmse_log10": float(row["rmse_log10"]),
        "target": "log10(conductivity_mS_cm)",
        "grid_x_step": 0.5,
        "grid_y_step": 0.25,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, OUTPUT, compress=3)
    print(json.dumps({k: v for k, v in bundle.items() if k != "model"}, indent=2))
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
