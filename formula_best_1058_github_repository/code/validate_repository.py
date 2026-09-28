"""Check the released data, descriptors, model bundle and key metrics."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    data = pd.read_csv(require(ROOT / "data" / "formula_best_1058.csv"))
    matrix = pd.read_csv(require(ROOT / "data" / "modeling_matrix_matminer_188.csv"))
    selected = pd.read_csv(require(ROOT / "reports" / "final_si_tables" / "SI_selected_116_descriptors.csv"))
    performance = pd.read_csv(require(ROOT / "reports" / "rf_rfe_10models" / "ten_model_oof_performance.csv"))
    bundle = joblib.load(require(ROOT / "models" / "final_extratrees_composition_model.joblib"))

    descriptor_columns = [c for c in matrix if c.startswith("matminer_")]
    best = performance.loc[performance["model"].eq("ExtraTrees")].iloc[0]
    checks = {
        "rows": len(data),
        "unique_composition_keys": data["composition_key"].nunique(),
        "source_counts": data["source_database"].value_counts().sort_index().to_dict(),
        "modeling_rows": len(matrix),
        "input_descriptors": len(descriptor_columns),
        "selected_descriptors": len(selected),
        "selected_unique": selected["descriptor"].nunique(),
        "bundle_model": bundle.get("model_name"),
        "bundle_selected_descriptors": len(bundle.get("selected_features", [])),
        "oof_r2": float(best["r2"]),
        "oof_mae_log10": float(best["mae_log10"]),
    }
    assert checks["rows"] == 1058
    assert checks["unique_composition_keys"] == 1058
    assert checks["source_counts"] == {"EES": 679, "OBELiX": 109, "OurCSV": 270}
    assert checks["modeling_rows"] == 1058
    assert checks["input_descriptors"] == 188
    assert checks["selected_descriptors"] == checks["selected_unique"] == 116
    assert checks["bundle_model"] == "ExtraTrees"
    assert checks["bundle_selected_descriptors"] == 116
    assert abs(checks["oof_r2"] - 0.777231621258534) < 1e-10
    assert abs(checks["oof_mae_log10"] - 0.6342786832515954) < 1e-10
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    print("Repository validation passed.")


if __name__ == "__main__":
    main()
