"""Generate publication figures and source data for the composition-only model.

Outputs are written as 600-dpi PNG and vector PDF files. The script uses only
the literature-only database (EES, OBELiX and curated data-our.csv).
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

import shap

from pymatgen.core import Composition
from matminer_composition_features import featurize_new_formulas


warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "modeling_matrix_matminer_188.csv"
OOF = ROOT / "data" / "final_matminer_extratrees_oof_predictions.csv"
MODEL = ROOT / "models" / "final_matminer_extratrees.joblib"
FIG_MAIN = ROOT / "figures" / "nature_main"
FIG_SI = ROOT / "figures" / "nature_si"
SOURCE = ROOT / "reports" / "figure_source_data"
for folder in (FIG_MAIN, FIG_SI, SOURCE):
    folder.mkdir(parents=True, exist_ok=True)

BLUE = "#3C5488"
RED = "#E64B35"
GREEN = "#00A087"
ORANGE = "#F39B7F"
PURPLE = "#8491B4"
GREY = "#7E7E7E"
PALETTE = [BLUE, GREEN, ORANGE, PURPLE, RED, GREY]


def nature_style():
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.transparent": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def panel_label(ax, label):
    """Panel letters are intentionally omitted for standalone publication figures."""
    return None

def clean_axes(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)

def save_figure(fig, folder, stem):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

def numeric_features(df):
    excluded = {"conductivity_mS_cm", "target_log10_mS_cm"}
    return [column for column in df.columns
            if column not in excluded and pd.api.types.is_numeric_dtype(df[column])]

def ranked_features(df, features):
    model = ExtraTreesRegressor(n_estimators=800, random_state=42, n_jobs=1,
                                min_samples_leaf=1, max_features=0.35)
    model.fit(df[features].to_numpy(float), df["target_log10_mS_cm"].to_numpy(float))
    return pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

def figure_a_distribution(df):
    values = df["target_log10_mS_cm"].to_numpy(float)
    pd.DataFrame({"log10_conductivity_mS_cm": values}).to_csv(
        SOURCE / "figure_a_log10_conductivity_distribution.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(3.5, 2.65))
    sns.histplot(values, bins=32, stat="density", color=BLUE, alpha=0.82,
                 edgecolor="white", linewidth=0.35, ax=ax)
    sns.kdeplot(values, color=RED, linewidth=1.25, ax=ax)
    median = float(np.median(values))
    ax.axvline(median, color="black", lw=0.8, ls="--")
    ax.text(median, ax.get_ylim()[1] * 0.96, f"Median = {median:.2f}",
            ha="right", va="top", fontsize=6.5)
    ax.set_xlabel(r"log$_{10}$ conductivity (mS cm$^{-1}$)")
    ax.set_ylabel("Probability density")
    clean_axes(ax)
    panel_label(ax, "a")
    fig.tight_layout()
    save_figure(fig, FIG_MAIN, "a_conductivity_distribution")


PERIODIC_POSITIONS = {
    "H": (1, 1), "He": (18, 1),
    "Li": (1, 2), "Be": (2, 2), "B": (13, 2), "C": (14, 2), "N": (15, 2), "O": (16, 2), "F": (17, 2), "Ne": (18, 2),
    "Na": (1, 3), "Mg": (2, 3), "Al": (13, 3), "Si": (14, 3), "P": (15, 3), "S": (16, 3), "Cl": (17, 3), "Ar": (18, 3),
    "K": (1, 4), "Ca": (2, 4), "Sc": (3, 4), "Ti": (4, 4), "V": (5, 4), "Cr": (6, 4), "Mn": (7, 4), "Fe": (8, 4), "Co": (9, 4), "Ni": (10, 4), "Cu": (11, 4), "Zn": (12, 4), "Ga": (13, 4), "Ge": (14, 4), "As": (15, 4), "Se": (16, 4), "Br": (17, 4), "Kr": (18, 4),
    "Rb": (1, 5), "Sr": (2, 5), "Y": (3, 5), "Zr": (4, 5), "Nb": (5, 5), "Mo": (6, 5), "Tc": (7, 5), "Ru": (8, 5), "Rh": (9, 5), "Pd": (10, 5), "Ag": (11, 5), "Cd": (12, 5), "In": (13, 5), "Sn": (14, 5), "Sb": (15, 5), "Te": (16, 5), "I": (17, 5), "Xe": (18, 5),
    "Cs": (1, 6), "Ba": (2, 6), "Hf": (4, 6), "Ta": (5, 6), "W": (6, 6), "Re": (7, 6), "Os": (8, 6), "Ir": (9, 6), "Pt": (10, 6), "Au": (11, 6), "Hg": (12, 6), "Tl": (13, 6), "Pb": (14, 6), "Bi": (15, 6), "Po": (16, 6), "At": (17, 6), "Rn": (18, 6),
    "La": (3, 8), "Ce": (4, 8), "Pr": (5, 8), "Nd": (6, 8), "Pm": (7, 8), "Sm": (8, 8), "Eu": (9, 8), "Gd": (10, 8), "Tb": (11, 8), "Dy": (12, 8), "Ho": (13, 8), "Er": (14, 8), "Tm": (15, 8), "Yb": (16, 8), "Lu": (17, 8),
}


def figure_b_periodic_table(df):
    parsed = [Composition(str(f)) for f in df["formula_clean"].fillna(df["formula_raw"])]
    counts = {element: int(sum(element in composition for composition in parsed))
              for element in PERIODIC_POSITIONS}
    source = pd.DataFrame({"element": counts.keys(), "formula_count": counts.values()})
    source.to_csv(SOURCE / "figure_b_element_frequency.csv", index=False)

    positive = [v for v in counts.values() if v > 0]
    norm = LogNorm(vmin=1, vmax=max(positive))
    cmap = mpl.colormaps["YlGnBu"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for element, (group, period) in PERIODIC_POSITIONS.items():
        count = counts[element]
        color = cmap(norm(count)) if count > 0 else "#F2F2F2"
        y = 9 - period
        ax.add_patch(Rectangle((group - 1, y - 1), 0.92, 0.88,
                               facecolor=color, edgecolor="white", linewidth=0.45))
        ax.text(group - 0.54, y - 0.38, element, ha="center", va="center",
                fontsize=6.2, fontweight="bold")
        if count > 0:
            ax.text(group - 0.54, y - 0.72, str(count), ha="center", va="center", fontsize=4.6)
    ax.text(0.0, 1.0, "Lanthanides", fontsize=5.3, va="center")
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.026, pad=0.015)
    cbar.set_label("Number of formulas", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6, width=0.6, length=2)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 8.2)
    ax.set_aspect("equal")
    ax.axis("off")
    panel_label(ax, "b")
    fig.tight_layout()
    save_figure(fig, FIG_MAIN, "b_periodic_table_element_distribution")


def correlation_heatmap(corr, folder, stem, figsize, label=None, tick_size=5.5):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, cmap="vlag", center=0, vmin=-1, vmax=1, square=True,
                xticklabels=True, yticklabels=True, cbar_kws={"label": "Spearman r", "shrink": 0.72}, ax=ax)
    ax.tick_params(axis="x", labelrotation=90, labelsize=tick_size, length=0)
    ax.tick_params(axis="y", labelrotation=0, labelsize=tick_size, length=0)
    if label:
        panel_label(ax, label)
    fig.tight_layout()
    save_figure(fig, folder, stem)


def figure_c_correlations(df, ranking, all_features, selected_features):
    top10 = list(ranking.head(10).index)
    corr10 = df[top10].corr(method="spearman")
    corr10.to_csv(SOURCE / "figure_c_top10_feature_correlations.csv")
    correlation_heatmap(corr10, FIG_MAIN, "c_top10_feature_correlations", (4.35, 3.8), "c", 6.2)



def model_specs(top120, top80):
    specs = [
        ("Ridge", top120, make_pipeline(StandardScaler(), RidgeCV(alphas=[10.0]))),
        ("HistGradientBoosting", top120, HistGradientBoostingRegressor(max_iter=600, learning_rate=0.06, max_leaf_nodes=63, l2_regularization=0.05, random_state=42)),
        ("Random forest", top120, RandomForestRegressor(n_estimators=500, max_features=0.20, min_samples_leaf=1, random_state=42, n_jobs=1)),
        ("LightGBM", top120, LGBMRegressor(n_estimators=700, learning_rate=0.05, num_leaves=15, colsample_bytree=0.8, reg_lambda=1.0, random_state=42, n_jobs=1, verbose=-1, force_col_wise=True, deterministic=True)),
        ("ExtraTrees", top120, ExtraTreesRegressor(n_estimators=500, max_features=0.60, min_samples_leaf=1, random_state=42, n_jobs=1)),
    ]
    return specs


def figure_d_models(df, ranking):
    from sklearn.base import clone

    y = df["target_log10_mS_cm"].to_numpy(float)
    top120 = list(ranking.head(120).index)
    top80 = list(ranking.head(80).index)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for fold, (train, test) in enumerate(cv.split(df), 1):
        for name, columns, estimator in model_specs(top120, top80):
            model = clone(estimator)
            model.fit(df.iloc[train][columns].to_numpy(float), y[train])
            pred = model.predict(df.iloc[test][columns].to_numpy(float))
            rows.append({"model": name, "fold": fold, "r2": r2_score(y[test], pred), "mae_log10": mean_absolute_error(y[test], pred)})
    scores = pd.DataFrame(rows)
    scores.to_csv(SOURCE / "figure_d_model_performance_by_fold.csv", index=False)
    order = scores.groupby("model")["r2"].mean().sort_values().index.tolist()
    summary = scores.groupby("model").agg(mean_r2=("r2", "mean"), sd_r2=("r2", "std"), mean_mae=("mae_log10", "mean"), sd_mae=("mae_log10", "std")).loc[order].reset_index()
    summary.to_csv(SOURCE / "figure_d_model_performance_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(4.5, 3.1))
    x = np.arange(len(summary))
    ax.bar(x, summary["mean_r2"], yerr=summary["sd_r2"], capsize=2.5,
           color=[PALETTE[i % len(PALETTE)] for i in range(len(summary))],
           edgecolor="black", linewidth=0.45, error_kw={"lw": 0.7})
    ax.set_xticks(x, summary["model"], rotation=35, ha="right")
    ax.set_ylabel(r"Five-fold $R^2$ (mean $\pm$ s.d.)")
    ax.set_ylim(min(-0.05, summary["mean_r2"].min() - 0.1), 0.9)
    clean_axes(ax)
    panel_label(ax, "d")
    fig.tight_layout()
    save_figure(fig, FIG_MAIN, "d_model_performance_with_error_bars")


def figure_e_scatter():
    pred = pd.read_csv(OOF)
    y = pred["target_log10_mS_cm"].to_numpy(float)
    p = pred["oof_pred_log10_mS_cm"].to_numpy(float)
    r2 = r2_score(y, p)
    mae = mean_absolute_error(y, p)
    rmse = math.sqrt(mean_squared_error(y, p))
    pred.to_csv(SOURCE / "figure_e_best_model_oof_predictions.csv", index=False)
    lo, hi = min(y.min(), p.min()) - 0.15, max(y.max(), p.max()) + 0.15
    fig, ax = plt.subplots(figsize=(3.5, 3.25))
    ax.scatter(y, p, s=11, alpha=0.58, color=BLUE, edgecolor="none", rasterized=True)
    ax.plot([lo, hi], [lo, hi], color="black", lw=0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Observed log$_{10}$ conductivity (mS cm$^{-1}$)")
    ax.set_ylabel(r"Predicted log$_{10}$ conductivity (mS cm$^{-1}$)")
    ax.text(0.04, 0.96, f"$R^2$ = {r2:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}",
            transform=ax.transAxes, ha="left", va="top")
    clean_axes(ax)
    panel_label(ax, "e")
    fig.tight_layout()
    save_figure(fig, FIG_MAIN, "e_best_model_oof_scatter")


def ensemble_shap(df, selected_features):
    bundle = joblib.load(MODEL)
    rng = np.random.default_rng(42)
    n = min(500, len(df))
    sample_idx = np.sort(rng.choice(len(df), size=n, replace=False))
    values = df.iloc[sample_idx][selected_features].copy()
    shap_total = pd.DataFrame(0.0, index=np.arange(n), columns=selected_features)
    for item in bundle["models"]:
        columns = item["feature_columns"]
        explainer = shap.TreeExplainer(item["model"])
        current = explainer.shap_values(df.iloc[sample_idx][columns].to_numpy(float), check_additivity=False)
        shap_total.loc[:, columns] += np.asarray(current) / len(bundle["models"])
    return values.reset_index(drop=True), shap_total


def shap_beeswarm(values, shap_values, features, folder, stem, figsize, label=None, ysize=5.8):
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=figsize)
    cmap = mpl.colormaps["coolwarm"]
    for row, feature in enumerate(features[::-1]):
        x = shap_values[feature].to_numpy(float)
        v = values[feature].to_numpy(float)
        ranks = pd.Series(v).rank(pct=True).to_numpy(float)
        jitter = rng.normal(0, 0.095, size=len(x))
        ax.scatter(x, row + jitter, c=ranks, cmap=cmap, vmin=0, vmax=1,
                   s=7 if len(features) <= 15 else 3.5, alpha=0.7,
                   edgecolor="none", rasterized=True)
    ax.axvline(0, color="#777777", lw=0.55)
    ax.set_yticks(range(len(features)), features[::-1])
    ax.tick_params(axis="y", labelsize=ysize, length=0)
    ax.set_xlabel(r"SHAP value (log$_{10}$ mS cm$^{-1}$)")
    clean_axes(ax)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_ticks([0, 1], labels=["Low", "High"])
    cbar.set_label("Feature value")
    if label:
        panel_label(ax, label)
    fig.tight_layout()
    save_figure(fig, folder, stem)


def figure_f_shap(df, selected_features, all_features):
    values, shap_values = ensemble_shap(df, selected_features)
    importance = shap_values.abs().mean().sort_values(ascending=False)
    source = pd.concat({"feature_value": values, "shap_value": shap_values}, axis=1)
    source.to_csv(SOURCE / "figure_f_shap_values_selected_features.csv", index=False)
    importance.rename("mean_abs_shap").to_csv(SOURCE / "figure_f_shap_importance.csv")
    top10 = list(importance.head(10).index)
    shap_beeswarm(values, shap_values, top10, FIG_MAIN, "f_shap_top10_features", (4.4, 3.8), "f", 6.3)


def formula_pair(chalcogen, halogen, x, y):
    parts = ["Li5.5", "P", f"S{4.5-x:.4g}"]
    if x > 1e-12:
        parts.append(f"{chalcogen}{x:.4g}")
    if 1.5 - y > 1e-12:
        parts.append(f"Cl{1.5-y:.4g}")
    if y > 1e-12:
        parts.append(f"{halogen}{y:.4g}")
    return "".join(parts)


def predict_formulas(formulas):
    for formula in formulas:
        Composition(str(formula))
    bundle = joblib.load(MODEL)
    features = featurize_new_formulas(formulas, expected_columns=bundle["all_descriptor_columns"])
    features = features.fillna(pd.Series(bundle.get("feature_medians", {}))).fillna(0.0)
    component = []
    for item in bundle["models"]:
        component.append(item["model"].predict(features[item["feature_columns"]].to_numpy(float)))
    log_pred = np.vstack(component).mean(axis=0)
    return log_pred, 10 ** log_pred


def figure_g_dopant_matrix():
    chalcogens = ["O", "Se", "Te"]
    halogens = ["F", "Br", "I"]
    x_values = np.linspace(0.25, 4.5, 18)
    y_values = np.linspace(0.10, 1.5, 15)
    formulas, metadata = [], []
    for chalcogen in chalcogens:
        for halogen in halogens:
            for x in x_values:
                for y in y_values:
                    formulas.append(formula_pair(chalcogen, halogen, x, y))
                    metadata.append({"chalcogen_dopant": chalcogen, "halogen_dopant": halogen, "x": x, "y": y})
    pred_log, pred = predict_formulas(formulas)
    source = pd.DataFrame(metadata)
    source["formula"] = formulas
    source["predicted_log10_mS_cm"] = pred_log
    source["predicted_mS_cm"] = pred
    source.to_csv(SOURCE / "figure_g_dopant_pair_grid_predictions.csv", index=False)
    summary = source.groupby(["halogen_dopant", "chalcogen_dopant"]).agg(
        geometric_mean_mS_cm=("predicted_log10_mS_cm", lambda x: float(10 ** x.mean())),
        arithmetic_mean_mS_cm=("predicted_mS_cm", "mean"),
        median_mS_cm=("predicted_mS_cm", "median"),
    ).reset_index()
    summary.to_csv(SOURCE / "figure_g_dopant_pair_means.csv", index=False)
    matrix = summary.pivot(index="halogen_dopant", columns="chalcogen_dopant", values="geometric_mean_mS_cm").loc[halogens, chalcogens]
    fig, ax = plt.subplots(figsize=(3.55, 2.9))
    sns.heatmap(matrix, cmap="mako", annot=True, fmt=".2f", linewidths=0.5,
                linecolor="white", cbar_kws={"label": r"Geometric mean (mS cm$^{-1}$)"}, ax=ax)
    ax.set_xlabel("Chalcogen-site dopant")
    ax.set_ylabel("Halogen-site dopant")
    ax.tick_params(length=0)
    panel_label(ax, "g")
    fig.tight_layout()
    save_figure(fig, FIG_MAIN, "g_dopant_pair_mean_predictions")


def figure_h_contour():
    x_values = np.linspace(0, 4.5, 91)
    y_values = np.linspace(0, 1.5, 61)
    formulas, metadata = [], []
    for y in y_values:
        for x in x_values:
            formulas.append(formula_pair("Se", "Br", x, y))
            metadata.append({"x_Se": x, "y_Br": y})
    pred_log, pred = predict_formulas(formulas)
    source = pd.DataFrame(metadata)
    source["formula"] = formulas
    source["predicted_log10_mS_cm"] = pred_log
    source["predicted_mS_cm"] = pred
    source.to_csv(SOURCE / "figure_h_Se_Br_contour_predictions.csv", index=False)
    z = source["predicted_log10_mS_cm"].to_numpy().reshape(len(y_values), len(x_values))
    best = source.iloc[int(np.argmax(source["predicted_mS_cm"].to_numpy()))]
    pd.DataFrame([best]).to_csv(SOURCE / "figure_h_predicted_maximum.csv", index=False)
    fig, ax = plt.subplots(figsize=(4.0, 3.05))
    levels = np.linspace(float(np.nanmin(z)), float(np.nanmax(z)), 18)
    contour = ax.contourf(x_values, y_values, z, levels=levels, cmap="viridis", extend="both")
    lines = ax.contour(x_values, y_values, z, levels=levels[::3], colors="white", linewidths=0.35, alpha=0.7)
    ax.clabel(lines, inline=True, fontsize=4.7, fmt="%.1f")
    ax.scatter([best["x_Se"]], [best["y_Br"]], marker="*", s=55,
               facecolor=RED, edgecolor="white", linewidth=0.55, zorder=4,
               clip_on=False)
    cbar = fig.colorbar(contour, ax=ax, pad=0.025)
    cbar.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))
    cbar.set_label(r"Predicted log$_{10}$ conductivity (mS cm$^{-1}$)")
    ax.set_xlabel(r"Se content, $x$")
    ax.set_ylabel(r"Br content, $y$")
    ax.set_title(r"Li$_{5.5}$PS$_{4.5-x}$Se$_x$Cl$_{1.5-y}$Br$_y$", pad=4)
    panel_label(ax, "h")
    fig.tight_layout()
    save_figure(fig, FIG_MAIN, "h_Li5.5PS4.5-xSexCl1.5-yBry_contour")


def write_manifest(df, all_features, selected_features, ranking):
    manifest = {
        "rows": int(len(df)),
        "sources": {str(k): int(v) for k, v in df["source_database"].value_counts().items()},
        "features_before_selection": len(all_features),
        "features_after_selection": len(selected_features),
        "top10_features": list(ranking.head(10).index),
        "figure_format": "600-dpi PNG plus vector PDF",
        "font": "Arial",
        "dopant_pair_average": "geometric mean over x=0.25..4.5 and y=0.10..1.5",
    }
    (SOURCE / "figure_generation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    nature_style()
    df = pd.read_csv(DATA)
    allowed = {"EES", "OBELiX", "OurCSV"}
    unexpected = set(df["source_database"].dropna().unique()) - allowed
    if unexpected:
        raise ValueError(f"Unexpected data sources: {sorted(unexpected)}")
    all_features = numeric_features(df)
    ranking = ranked_features(df, all_features)
    bundle = joblib.load(MODEL)
    selected_features = list(dict.fromkeys(c for item in bundle["models"] for c in item["feature_columns"]))

    figure_a_distribution(df)
    figure_b_periodic_table(df)
    figure_c_correlations(df, ranking, all_features, selected_features)
    figure_d_models(df, ranking)
    figure_e_scatter()
    figure_f_shap(df, selected_features, all_features)
    figure_g_dopant_matrix()
    figure_h_contour()
    write_manifest(df, all_features, selected_features, ranking)
    print(json.dumps({
        "done": True,
        "rows": len(df),
        "main_figures": len(list(FIG_MAIN.glob("*.png"))),
        "si_figures": len(list(FIG_SI.glob("*.png"))),
        "source_files": len(list(SOURCE.glob("*"))),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()






