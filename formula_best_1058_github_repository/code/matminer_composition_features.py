"""Standard composition descriptors from pymatgen and matminer only."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from matminer.featurizers.composition import (
    BandCenter,
    ElementFraction,
    ElementProperty,
    IonProperty,
    Stoichiometry,
    TMetalFraction,
    ValenceOrbital,
)


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "formula_best_1058.csv"
OUTPUT = ROOT / "data" / "modeling_matrix_matminer_188.csv"
PROVENANCE = ROOT / "reports" / "si_tables" / "SI_matminer_descriptor_provenance.csv"
BUILD_REPORT = ROOT / "reports" / "si_tables" / "SI_matminer_feature_build.json"

META_COLUMNS = [
    "source_database", "source_label", "source_doi", "source_ref",
    "formula_raw", "formula_clean", "composition_key",
    "conductivity_mS_cm", "target_log10_mS_cm",
]


def featurizers():
    return [
        ("ElementProperty", ElementProperty.from_preset("magpie"), "magpie"),
        ("Stoichiometry", Stoichiometry(), "default"),
        ("ElementFraction", ElementFraction(), "default"),
        ("ValenceOrbital", ValenceOrbital(), "default"),
        ("IonProperty", IonProperty(fast=True), "fast=True"),
        ("BandCenter", BandCenter(), "default"),
        ("TMetalFraction", TMetalFraction(), "default"),
    ]


def _safe_citations(featurizer):
    try:
        citation = " || ".join(str(x).replace("\n", " ") for x in featurizer.citations())
        # matminer 0.10.1 returns a mistyped DOI for the Magpie paper.
        # The publisher DOI is 10.1038/npjcompumats.2016.28.
        return citation.replace(
            "10.1038/npjcompumats.2017.28",
            "10.1038/npjcompumats.2016.28",
        )
    except Exception:
        return ""


def _safe_implementors(featurizer):
    try:
        return " | ".join(str(x) for x in featurizer.implementors())
    except Exception:
        return ""


def featurize_formulas(formulas: pd.Series):
    compositions = [Composition(str(f)) for f in formulas]
    frames = []
    provenance = []
    errors = []
    versions = {
        "pymatgen": importlib.metadata.version("pymatgen"),
        "matminer": importlib.metadata.version("matminer"),
        "chemml": importlib.metadata.version("chemml"),
    }
    for family, featurizer, preset in featurizers():
        labels = list(featurizer.feature_labels())
        rows = []
        for index, composition in enumerate(compositions):
            try:
                values = featurizer.featurize(composition)
            except Exception as exc:
                values = [np.nan] * len(labels)
                errors.append({"row": index, "formula": str(formulas.iloc[index]),
                               "featurizer": family, "error": repr(exc)})
            rows.append(values)
        names = [f"matminer_{family}__{label}" for label in labels]
        frames.append(pd.DataFrame(rows, columns=names, index=formulas.index))
        citation = _safe_citations(featurizer)
        implementors = _safe_implementors(featurizer)
        for name, label in zip(names, labels):
            provenance.append({
                "descriptor": name,
                "library": "matminer",
                "library_version": versions["matminer"],
                "composition_parser": "pymatgen.core.Composition",
                "pymatgen_version": versions["pymatgen"],
                "featurizer_class": f"matminer.featurizers.composition.{family}",
                "preset_or_options": preset,
                "original_feature_label": label,
                "citation": citation,
                "implementors": implementors,
                "documentation": "https://hackingmaterials.lbl.gov/matminer/featurizer_summary.html",
                "custom_descriptor": False,
            })
    features = pd.concat(frames, axis=1).replace([np.inf, -np.inf], np.nan)
    return features, pd.DataFrame(provenance), errors, versions


def build_dataset(force=False):
    if OUTPUT.exists() and PROVENANCE.exists() and BUILD_REPORT.exists() and not force:
        return pd.read_csv(OUTPUT)
    raw = pd.read_csv(RAW)
    formulas = raw["formula_clean"].fillna(raw["formula_raw"]).astype(str)
    features, provenance, errors, versions = featurize_formulas(formulas)
    raw_feature_count = features.shape[1]
    nonconstant = [c for c in features if features[c].nunique(dropna=True) > 1]
    features = features[nonconstant]
    imputation = features.isna().sum()
    medians = features.median(numeric_only=True).fillna(0)
    features = features.fillna(medians).astype(float)

    output = pd.concat([raw[META_COLUMNS].reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    provenance = provenance[provenance.descriptor.isin(nonconstant)].copy()
    provenance["missing_values_imputed"] = provenance.descriptor.map(imputation).fillna(0).astype(int)
    provenance.to_csv(PROVENANCE, index=False, encoding="utf-8-sig")
    report = {
        "rows": len(raw), "raw_descriptor_count": raw_feature_count,
        "nonconstant_descriptor_count": len(nonconstant),
        "libraries": versions, "featurizers": [x[0] for x in featurizers()],
        "formula_parse_failures": 0,
        "featurization_error_count": len(errors), "featurization_errors": errors,
        "custom_descriptors_used": False,
        "output": str(OUTPUT.relative_to(ROOT)),
        "provenance": str(PROVENANCE.relative_to(ROOT)),
    }
    BUILD_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def featurize_new_formulas(formulas, expected_columns=None):
    series = pd.Series(list(formulas), name="formula_clean")
    features, _, errors, _ = featurize_formulas(series)
    if errors:
        raise ValueError(f"matminer featurization failed: {errors[:3]}")
    features = features.replace([np.inf, -np.inf], np.nan)
    if expected_columns is not None:
        for column in expected_columns:
            if column not in features:
                features[column] = 0.0
        features = features[list(expected_columns)]
    return features


if __name__ == "__main__":
    frame = build_dataset(force=True)
    print(json.dumps({"rows": len(frame), "columns": len(frame.columns), "output": str(OUTPUT)}, ensure_ascii=False))
