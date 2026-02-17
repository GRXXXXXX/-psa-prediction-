"""
PSA Gray Zone Malignancy Prediction - Visualization Suite
Generates 6 publication-quality plots from trained model results.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select specific columns from X (needed for unpickling the saved model)."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, self.columns]

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
RESULTS_DIR = BASE_DIR / "results"
DATA_PATH = BASE_DIR / "完整版数据.xlsx"

COLORS = {
    "combined": "#E74C3C",
    "f-PSA/t-PSA(%)": "#3498DB",
    "PHI": "#2ECC71",
    "PSAD": "#F39C12",
    "P2PSA（pg/ml）": "#9B59B6",
    "计算得前列腺体积（ml）": "#1ABC9C",
}
FEATURE_DISPLAY = {
    "f-PSA/t-PSA(%)": "f/t PSA",
    "PHI": "PHI",
    "PSAD": "PSAD",
    "P2PSA（pg/ml）": "P2PSA",
    "计算得前列腺体积（ml）": "Prostate Vol",
}


def load_artifacts():
    with open(MODEL_DIR / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    cv_probs = np.load(RESULTS_DIR / "cv_probs.npy")
    y_true = np.load(RESULTS_DIR / "y_true.npy")
    df = pd.read_excel(DATA_PATH)
    return config, cv_probs, y_true, df


def get_feature_cols(config):
    """Get the feature column names based on config version."""
    if "feature_names_5" in config:
        return config["feature_names_5"]
    if "feature_names_4" in config:
        return config["feature_names_4"]
    return config.get("feature_names", [])


def get_display_feature_cols(config):
    """Get only the 4 primary biomarkers for display (exclude derived features)."""
    primary = ["f-PSA/t-PSA(%)", "PHI", "PSAD", "P2PSA（pg/ml）"]
    all_cols = get_feature_cols(config)
    extra = [c for c in all_cols if c not in primary]
    return primary + extra


def plot_auc_comparison(config):
    single_aucs = config["single_indicator_aucs"]
    combined_auc = config["cv_auc"]
    feature_names = list(single_aucs.keys())

    labels = [FEATURE_DISPLAY.get(n, n) for n in feature_names] + ["Combined\nModel"]
    values = [single_aucs[n] for n in feature_names] + [combined_auc]
    colors = [COLORS.get(n, "#95A5A6") for n in feature_names] + [COLORS["combined"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax.set_ylabel("AUC")
    ax.set_title("AUC Comparison: Single Indicators vs Combined Model")
    ax.set_ylim(0, 1.08)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random (0.5)")
    ax.legend(loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(RESULTS_DIR / "auc_comparison.png")
    plt.close(fig)
    print("  Saved: auc_comparison.png")


def plot_roc_curves(config, cv_probs, y_true, df):
    single_aucs = config["single_indicator_aucs"]
    feature_names = list(single_aucs.keys())

    fig, ax = plt.subplots(figsize=(7, 7))

    for name in feature_names:
        vals = df[name].values.astype(float)
        auc_val = roc_auc_score(y_true, vals)
        if auc_val < 0.5:
            vals = -vals
            auc_val = roc_auc_score(y_true, vals)
        fpr, tpr, _ = roc_curve(y_true, vals)
        display_name = FEATURE_DISPLAY.get(name, name)
        ax.plot(fpr, tpr, color=COLORS.get(name, "#95A5A6"), linewidth=1.5,
                label=f"{display_name} (AUC={auc_val:.3f})")

    fpr_c, tpr_c, _ = roc_curve(y_true, cv_probs)
    combined_auc = roc_auc_score(y_true, cv_probs)
    ax.plot(fpr_c, tpr_c, color=COLORS["combined"], linewidth=2.5,
            label=f"Combined Model (AUC={combined_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    ax.set_xlabel("1 - Specificity (False Positive Rate)")
    ax.set_ylabel("Sensitivity (True Positive Rate)")
    ax.set_title("ROC Curves: Single Indicators & Combined Model")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")

    fig.savefig(RESULTS_DIR / "roc_curves.png")
    plt.close(fig)
    print("  Saved: roc_curves.png")


def plot_optimal_cutoff(config, cv_probs, y_true):
    cutoff = config["optimal_cutoff"]
    fpr, tpr, thresholds = roc_curve(y_true, cv_probs)
    combined_auc = roc_auc_score(y_true, cv_probs)

    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_fpr, best_tpr = fpr[best_idx], tpr[best_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(fpr, tpr, color=COLORS["combined"], linewidth=2,
             label=f"Combined Model (AUC={combined_auc:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax1.scatter(best_fpr, best_tpr, color=COLORS["combined"], s=120,
                zorder=5, edgecolors="black", linewidths=1.5)
    ax1.annotate(
        f"Optimal Cutoff = {cutoff:.3f}\nSens = {best_tpr:.3f}\nSpec = {1 - best_fpr:.3f}",
        xy=(best_fpr, best_tpr),
        xytext=(best_fpr + 0.15, best_tpr - 0.15),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="black"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"),
    )
    ax1.set_xlabel("1 - Specificity (False Positive Rate)")
    ax1.set_ylabel("Sensitivity (True Positive Rate)")
    ax1.set_title("ROC Curve with Optimal Cutoff (Youden's J)")
    ax1.legend(loc="lower right")
    ax1.set_aspect("equal")

    valid = thresholds < 1.0
    thresh_plot = thresholds[valid]
    tpr_plot = tpr[valid]
    spec_plot = 1 - fpr[valid]

    ax2.plot(thresh_plot, tpr_plot, color="#E74C3C", linewidth=2, label="Sensitivity")
    ax2.plot(thresh_plot, spec_plot, color="#3498DB", linewidth=2, label="Specificity")
    ax2.axvline(x=cutoff, color="gray", linestyle="--", alpha=0.7, label=f"Cutoff = {cutoff:.3f}")

    cross_idx = np.argmin(np.abs(tpr_plot - spec_plot))
    ax2.scatter(thresh_plot[cross_idx], tpr_plot[cross_idx], color="black", s=80, zorder=5)

    ax2.set_xlabel("Probability Threshold")
    ax2.set_ylabel("Rate")
    ax2.set_title("Sensitivity & Specificity vs Threshold")
    ax2.legend(loc="center right")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.02, 1.05)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "optimal_cutoff.png")
    plt.close(fig)
    print("  Saved: optimal_cutoff.png")


def plot_calibration_curve(cv_probs, y_true):
    fig, ax = plt.subplots(figsize=(7, 7))

    prob_true, prob_pred = calibration_curve(y_true, cv_probs, n_bins=8, strategy="uniform")

    ax.plot(prob_pred, prob_true, "o-", color=COLORS["combined"], linewidth=2,
            markersize=6, label="Combined Model")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfectly Calibrated")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title("Calibration Curve")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")

    fig.savefig(RESULTS_DIR / "calibration_curve.png")
    plt.close(fig)
    print("  Saved: calibration_curve.png")


def plot_performance_summary(config, cv_probs, y_true):
    cutoff = config["optimal_cutoff"]
    y_pred = (cv_probs >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "AUC": config["cv_auc"],
        "Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "PPV": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "NPV": tn / (tn + fn) if (tn + fn) > 0 else 0,
        "Accuracy": (tp + tn) / (tp + tn + fp + fn),
    }

    labels = list(metrics.keys())
    values = list(metrics.values())
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles += [angles[0]]

    fig = plt.figure(figsize=(14, 6))

    ax_radar = fig.add_subplot(121, polar=True)
    ax_radar.plot(angles, values_plot, "o-", color=COLORS["combined"], linewidth=2, markersize=6)
    ax_radar.fill(angles, values_plot, color=COLORS["combined"], alpha=0.15)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(labels, fontsize=10)
    ax_radar.set_ylim(0, 1.05)
    ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_radar.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="gray")
    ax_radar.set_title(f"Performance Radar - {config['best_model_name']}", pad=20)

    ax_table = fig.add_subplot(122)
    ax_table.axis("off")
    cell_text = [[f"{v:.4f}"] for v in values]
    table = ax_table.table(
        cellText=cell_text,
        rowLabels=labels,
        colLabels=["Value"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#E74C3C")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == -1:
            cell.set_facecolor("#F8F8F8")
            cell.set_text_props(fontweight="bold")

    ax_table.set_title(f"Cutoff = {cutoff:.4f}", fontsize=12, pad=20)

    fig.suptitle("Model Performance Summary", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "performance_summary.png")
    plt.close(fig)
    print("  Saved: performance_summary.png")


def plot_feature_importance(config):
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import VotingClassifier, StackingClassifier

    model = joblib.load(MODEL_DIR / "best_model.pkl")

    feature_cols = get_feature_cols(config)
    n_input = len(feature_cols)
    display_names = [FEATURE_DISPLAY.get(n, n) for n in feature_cols]

    fig, ax = plt.subplots(figsize=(7, 5))

    def _extract_importances(m, n_raw):
        """Extract importances, mapping poly/selected features back to raw inputs."""
        if isinstance(m, (VotingClassifier, StackingClassifier)):
            agg = np.zeros(n_raw)
            count = 0
            fitted = m.estimators_ if hasattr(m, "estimators_") else []
            for est in fitted:
                imp = _extract_importances(est, n_raw)
                if imp is not None:
                    s = np.sum(imp)
                    agg += imp / s if s > 0 else imp
                    count += 1
            return agg / count if count > 0 else None

        if isinstance(m, Pipeline):
            clf = m.named_steps.get("clf", None)
            if clf is None:
                return None

            has_sel = "sel" in m.named_steps
            has_poly = "poly" in m.named_steps

            if has_sel:
                sel_cols = m.named_steps["sel"].columns
                n_selected = len(sel_cols)
            else:
                sel_cols = list(range(n_raw))
                n_selected = n_raw

            if has_poly:
                poly = m.named_steps["poly"]
                if hasattr(clf, "coef_"):
                    coefs = np.abs(clf.coef_[0])
                    powers = poly.powers_
                    imp_sel = np.zeros(n_selected)
                    for j, row in enumerate(powers):
                        for fi in range(n_selected):
                            if row[fi] > 0:
                                imp_sel[fi] += coefs[j]
                elif hasattr(clf, "feature_importances_"):
                    raw = clf.feature_importances_
                    powers = poly.powers_
                    imp_sel = np.zeros(n_selected)
                    for j, row in enumerate(powers):
                        for fi in range(n_selected):
                            if row[fi] > 0:
                                imp_sel[fi] += raw[j]
                else:
                    return None
            else:
                if hasattr(clf, "feature_importances_"):
                    imp_sel = clf.feature_importances_[:n_selected]
                elif hasattr(clf, "coef_"):
                    imp_sel = np.abs(clf.coef_[0])[:n_selected]
                else:
                    return None

            imp = np.zeros(n_raw)
            for i, col_idx in enumerate(sel_cols):
                if i < len(imp_sel):
                    imp[col_idx] = imp_sel[i]
            return imp

        if hasattr(m, "feature_importances_"):
            return m.feature_importances_[:n_raw]
        if hasattr(m, "coef_"):
            return np.abs(m.coef_[0])[:n_raw]
        return None

    importances = _extract_importances(model, n_input)
    if importances is None:
        print("  Skipped: feature_importance.png (model has no importances)")
        return

    sorted_idx = np.argsort(importances)
    colors = [COLORS.get(feature_cols[i], "#95A5A6") for i in sorted_idx]

    ax.barh([display_names[i] for i in sorted_idx], importances[sorted_idx],
            color=colors, edgecolor="white", linewidth=1.5)

    for i, (val, idx) in enumerate(zip(importances[sorted_idx], sorted_idx)):
        ax.text(val + max(importances) * 0.02, i, f"{val:.3f}",
                va="center", fontsize=11, fontweight="bold")

    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Feature Importance - {config['best_model_name']}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_importance.png")
    plt.close(fig)
    print("  Saved: feature_importance.png")


def main():
    print("=" * 60)
    print("PSA Gray Zone - Generating Visualizations")
    print("=" * 60)

    config, cv_probs, y_true, df = load_artifacts()
    print(f"Model: {config['best_model_name']} | AUC: {config['cv_auc']}")
    print()

    print("[1/6] AUC Comparison Bar Chart...")
    plot_auc_comparison(config)

    print("[2/6] ROC Curves...")
    plot_roc_curves(config, cv_probs, y_true, df)

    print("[3/6] Optimal Cutoff Plot...")
    plot_optimal_cutoff(config, cv_probs, y_true)

    print("[4/6] Calibration Curve...")
    plot_calibration_curve(cv_probs, y_true)

    print("[5/6] Performance Summary...")
    plot_performance_summary(config, cv_probs, y_true)

    print("[6/6] Feature Importance...")
    plot_feature_importance(config)

    print(f"\nAll plots saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
