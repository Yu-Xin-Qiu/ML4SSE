"""Predict room-temperature ionic conductivity from chemical formulas only."""
from pathlib import Path
import argparse

import joblib
import pandas as pd
from pymatgen.core import Composition

from matminer_composition_features import featurize_new_formulas

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "final_extratrees_composition_model.joblib"


def predict(formulas):
    bad = []
    for formula in formulas:
        try:
            Composition(str(formula))
        except Exception:
            bad.append(formula)
    if bad:
        raise ValueError(f"Could not parse formulas: {bad}")

    bundle = joblib.load(MODEL_PATH)
    expected = bundle["selected_features"]
    new_features = featurize_new_formulas(formulas, expected_columns=expected)
    new_features = new_features.fillna(0.0)
    pred_log10 = bundle["model"].predict(new_features[expected].to_numpy(float))
    factor = 10 ** float(bundle.get("reported_oof_mae_log10", 0.6342786832515954))
    return pd.DataFrame({
        "formula": formulas,
        "model": bundle.get("model_name", "ExtraTrees"),
        "predicted_log10_mS_cm": pred_log10,
        "predicted_conductivity_mS_cm": 10 ** pred_log10,
        "approx_lower_mS_cm": (10 ** pred_log10) / factor,
        "approx_upper_mS_cm": (10 ** pred_log10) * factor,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("formulas", nargs="*", help="Formulas such as Li6PS5Cl")
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--formula-column", default="formula")
    parser.add_argument("--output", type=Path, default=ROOT / "predictions.csv")
    args = parser.parse_args()
    formulas = list(args.formulas)
    if args.input_csv:
        table = pd.read_csv(args.input_csv)
        formulas.extend(table[args.formula_column].dropna().astype(str).tolist())
    if not formulas:
        parser.error("Provide formulas or --input-csv")
    result = predict(formulas)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

