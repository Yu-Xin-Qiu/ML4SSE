"""RF-RFECV feature selection and ten-model comparison.

Inputs are limited to standard pymatgen/matminer composition descriptors.
All model comparisons use the same shuffled five-fold split.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_selection import RFECV
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "modeling_matrix_matminer_188.csv"
OUT = ROOT / "reports" / "rf_rfe_10models"
FIG = ROOT / "figures" / "rf_rfe_10models"
MODELS = ROOT / "models" / "rf_rfe_10models"
for folder in (OUT, FIG, MODELS):
    folder.mkdir(parents=True, exist_ok=True)

META = {
    "source_database", "source_label", "source_doi", "source_ref",
    "formula_raw", "formula_clean", "composition_key", "conductivity_mS_cm",
    "target_log10_mS_cm", "parsed_formula_json",
}
CV = KFold(n_splits=5, shuffle=True, random_state=42)


def metric_row(name, y, pred):
    mae = float(mean_absolute_error(y, pred))
    return {
        "model": name,
        "r2": float(r2_score(y, pred)),
        "mae_log10": mae,
        "rmse_log10": float(mean_squared_error(y, pred) ** 0.5),
        "mae_factor": float(10**mae),
    }


def nature_style():
    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 7.5, "axes.labelsize": 8,
        "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "out",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def boxed(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)


def save(fig, stem):
    fig.savefig(FIG / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_data():
    df = pd.read_csv(DATA)
    features = [
        c for c in df.columns
        if c not in META and pd.api.types.is_numeric_dtype(df[c])
        and df[c].nunique(dropna=True) > 1
    ]
    if len(features) != 188:
        raise ValueError(f"Expected 188 standard descriptors, found {len(features)}")
    return df, features, df["target_log10_mS_cm"].to_numpy(float)


def run_rfecv(df, features, y):
    selected_path = OUT / "rfecv_selected_features.json"
    result_path = OUT / "rfecv_feature_count_results.csv"
    ranking_path = OUT / "rfecv_feature_ranking.csv"
    model_path = MODELS / "rfecv_selector.joblib"
    if all(p.exists() for p in (selected_path, result_path, ranking_path, model_path)):
        return json.loads(selected_path.read_text(encoding="utf-8")), joblib.load(model_path)

    selector_model = RandomForestRegressor(
        n_estimators=400, max_features=0.50, min_samples_leaf=1,
        random_state=42, n_jobs=1,
    )
    rfecv = RFECV(
        estimator=selector_model,
        step=0.10,
        min_features_to_select=10,
        cv=CV,
        scoring="r2",
        n_jobs=1,
        importance_getter="auto",
    )
    rfecv.fit(df[features].to_numpy(float), y)
    selected = list(np.asarray(features)[rfecv.support_])
    selected_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame({
        "descriptor": features,
        "selected": rfecv.support_,
        "rfe_rank": rfecv.ranking_,
    }).sort_values(["rfe_rank", "descriptor"]).to_csv(ranking_path, index=False, encoding="utf-8-sig")
    one_dimensional = {
        key: value for key, value in rfecv.cv_results_.items()
        if np.asarray(value).ndim == 1
    }
    cvres = pd.DataFrame(one_dimensional)
    keep = [c for c in cvres.columns if c == "n_features" or c.startswith("split") or c in {"mean_test_score", "std_test_score"}]
    cvres[keep].to_csv(result_path, index=False, encoding="utf-8-sig")
    np.savez_compressed(
        OUT / "rfecv_support_ranking_trajectory.npz",
        **{key: value for key, value in rfecv.cv_results_.items() if np.asarray(value).ndim > 1},
    )
    joblib.dump(rfecv, model_path)

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(cvres["n_features"], cvres["mean_test_score"], marker="o", ms=3.5, lw=1.0, color="#3C5488")
    ax.fill_between(cvres["n_features"], cvres["mean_test_score"] - cvres["std_test_score"],
                    cvres["mean_test_score"] + cvres["std_test_score"], color="#3C5488", alpha=0.18)
    ax.axvline(len(selected), color="#E64B35", ls="--", lw=0.9)
    ax.set_xlabel("Number of descriptors retained by RF-RFECV")
    ax.set_ylabel(r"Mean five-fold $R^2$")
    ax.text(0.98, 0.04, f"Selected = {len(selected)}", transform=ax.transAxes, ha="right", va="bottom")
    boxed(ax); fig.tight_layout(); save(fig, "rf_rfecv_optimal_feature_count")
    return selected, rfecv


def model_spaces():
    return {
        "Ridge": (
            make_pipeline(StandardScaler(), Ridge()),
            {"ridge__alpha": np.logspace(-3, 3, 13)},
        ),
        "ElasticNet": (
            make_pipeline(StandardScaler(), ElasticNet(max_iter=30000, random_state=42)),
            {"elasticnet__alpha": [1e-4,1e-3,1e-2,1e-1,1.0], "elasticnet__l1_ratio": [0.1,0.3,0.5,0.7,0.9]},
        ),
        "KNN": (
            make_pipeline(StandardScaler(), KNeighborsRegressor()),
            {"kneighborsregressor__n_neighbors": [3,5,7,9,13,17,23,29],
             "kneighborsregressor__weights": ["uniform", "distance"],
             "kneighborsregressor__p": [1, 2]},
        ),
        "SVR": (
            make_pipeline(StandardScaler(), SVR(kernel="rbf")),
            {"svr__C": [0.3,1,3,10,30], "svr__gamma": [1e-4,1e-3,1e-2,0.1],
             "svr__epsilon": [0.05,0.2]},
        ),
        "RandomForest": (
            RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=1),
            {"max_features": [0.2,0.4,0.6,0.8], "min_samples_leaf": [1,2,4],
             "max_depth": [None,20]},
        ),
        "ExtraTrees": (
            ExtraTreesRegressor(n_estimators=400, random_state=42, n_jobs=1),
            {"max_features": [0.2,0.4,0.6,0.8], "min_samples_leaf": [1,2,4],
             "max_depth": [None,20]},
        ),
        "AdaBoost": (
            AdaBoostRegressor(random_state=42),
            {"n_estimators": [100,250,500,750], "learning_rate": [0.01,0.05,0.2,0.8],
             "loss": ["linear","exponential"]},
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(n_estimators=400, random_state=42),
            {"learning_rate": [0.02,0.05,0.1,0.15], "max_depth": [1,2,3],
             "min_samples_leaf": [1,4], "subsample": [0.75,1.0]},
        ),
        "XGBoost": (
            XGBRegressor(n_estimators=600, objective="reg:squarederror", random_state=42,
                         n_jobs=1, tree_method="hist"),
            {"learning_rate": [0.02,0.05,0.1,0.15], "max_depth": [2,4,6],
             "subsample": [0.75,1.0], "colsample_bytree": [0.6,1.0]},
        ),
        "LightGBM": (
            LGBMRegressor(n_estimators=600, random_state=42, n_jobs=1, verbose=-1,
                          force_col_wise=True, deterministic=True),
            {"learning_rate": [0.02,0.05,0.1,0.15], "num_leaves": [15,31,63],
             "colsample_bytree": [0.6,1.0], "reg_lambda": [0.1,5.0]},
        ),
    }


def slug(name):
    return name.lower().replace(" ", "_")


def run_models(df, selected, y):
    x = df[selected].to_numpy(float)
    all_predictions = {}
    metrics = []
    search_frames = []
    best_rows = []

    for name, (estimator, space) in model_spaces().items():
        key = slug(name)
        model_file = MODELS / f"{key}_best.joblib"
        pred_file = OUT / f"{key}_oof.csv"
        search_file = OUT / f"{key}_search.csv"
        best_file = OUT / f"{key}_best.json"
        if all(p.exists() for p in (model_file, pred_file, search_file, best_file)):
            best = joblib.load(model_file)
            pred = pd.read_csv(pred_file)["oof_prediction"].to_numpy(float)
            search_df = pd.read_csv(search_file)
            best_info = json.loads(best_file.read_text(encoding="utf-8"))
        else:
            search = GridSearchCV(
                estimator, space, scoring="r2", cv=CV,
                n_jobs=1, refit=True, return_train_score=False,
                error_score="raise",
            )
            search.fit(x, y)
            best = search.best_estimator_
            pred = cross_val_predict(clone(best), x, y, cv=CV, n_jobs=1)
            search_df = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
            best_info = {"model": name, "best_params": search.best_params_,
                         "best_mean_cv_r2": float(search.best_score_),
                         "search_type": "exhaustive GridSearchCV",
                         "n_grid_points": int(len(search.cv_results_["params"]))}
            joblib.dump(best, model_file)
            pd.DataFrame({"oof_prediction": pred}).to_csv(pred_file, index=False)
            search_df.to_csv(search_file, index=False, encoding="utf-8-sig")
            best_file.write_text(json.dumps(best_info, ensure_ascii=False, indent=2), encoding="utf-8")
        all_predictions[name] = pred
        metrics.append(metric_row(name, y, pred))
        search_df.insert(0, "model", name)
        search_frames.append(search_df)
        best_rows.append({"model": name, "best_mean_cv_r2": best_info["best_mean_cv_r2"],
                          "best_params": json.dumps(best_info["best_params"], ensure_ascii=False)})
        print(name, metrics[-1], flush=True)

    perf = pd.DataFrame(metrics).sort_values("r2", ascending=False).reset_index(drop=True)
    perf.to_csv(OUT / "ten_model_oof_performance.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(best_rows).sort_values("best_mean_cv_r2", ascending=False).to_csv(
        OUT / "ten_model_best_hyperparameters.csv", index=False, encoding="utf-8-sig")
    pd.concat(search_frames, ignore_index=True).to_csv(
        OUT / "ten_model_all_search_results.csv", index=False, encoding="utf-8-sig")

    trace = df[["source_database", "source_doi", "formula_raw", "formula_clean", "composition_key"]].copy()
    trace["observed_log10_mS_cm"] = y
    for name, pred in all_predictions.items():
        trace[f"pred_{slug(name)}_log10_mS_cm"] = pred
    trace.to_csv(OUT / "ten_model_oof_predictions.csv", index=False, encoding="utf-8-sig")
    return all_predictions, perf


def figures(y, predictions, perf):
    colors = plt.cm.viridis(np.linspace(0.15, 0.90, len(perf)))
    fig, ax = plt.subplots(figsize=(6.3, 3.3))
    ordered = perf.sort_values("r2")
    ax.barh(np.arange(len(ordered)), ordered["r2"], color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(np.arange(len(ordered)), ordered["model"])
    ax.set_xlabel(r"Five-fold OOF $R^2$")
    boxed(ax); fig.tight_layout(); save(fig, "ten_model_oof_r2_comparison")

    names = perf["model"].tolist()
    fig, axes = plt.subplots(2, 5, figsize=(14.0, 5.7), sharex=True, sharey=True)
    lo = min(y.min(), *(predictions[n].min() for n in names)) - 0.15
    hi = max(y.max(), *(predictions[n].max() for n in names)) + 0.15
    for ax, name in zip(axes.flat, names):
        pred = predictions[name]
        row = perf.set_index("model").loc[name]
        ax.scatter(y, pred, s=7, alpha=0.48, edgecolor="none", color="#3C5488", rasterized=True)
        ax.plot([lo, hi], [lo, hi], color="black", lw=0.7, ls="--")
        ax.set_title(f"{name}\n$R^2$={row.r2:.3f}, MAE={row.mae_log10:.3f}")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); boxed(ax)
    for ax in axes[-1]: ax.set_xlabel(r"Observed log$_{10}\sigma$")
    for ax in axes[:, 0]: ax.set_ylabel(r"Predicted log$_{10}\sigma$")
    fig.tight_layout(); save(fig, "ten_model_oof_regression_scatter")

def main():
    nature_style()
    df, features, y = load_data()
    selected, rfecv = run_rfecv(df, features, y)
    predictions, perf = run_models(df, selected, y)
    figures(y, predictions, perf)
    manifest = {
        "rows": len(df), "input_descriptors": len(features),
        "selected_descriptors": len(selected),
        "selector": "RFECV(RandomForestRegressor, step=0.10, 5-fold R2)",
        "cv": "KFold(n_splits=5, shuffle=True, random_state=42)",
        "models": list(model_spaces()),
        "best_individual": perf.iloc[0].to_dict(),
        "final_model": "ExtraTrees",
        "stacking_used_as_final_model": False,
        "descriptor_backend": "pymatgen.core.Composition + matminer standard featurizers",
        "custom_descriptors_used": False,
    }
    (OUT / "rf_rfe_10model_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
