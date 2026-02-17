"""
PSA Gray Zone Malignancy Prediction - Streamlit UI
Input biomarker values to get malignancy probability prediction.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select specific columns from X (needed for unpickling the saved model)."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, self.columns]

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
RESULTS_DIR = BASE_DIR / "results"

FEATURE_RANGES = {
    "f-PSA/t-PSA(%)": {"min": 0.0, "max": 1.0, "default": 0.15, "step": 0.01,
                        "label": "f/t PSA (%)", "help": "f-PSA / t-PSA ratio"},
    "PHI": {"min": 0.0, "max": 150.0, "default": 35.0, "step": 0.1,
            "label": "PHI", "help": "Prostate Health Index"},
    "PSAD": {"min": 0.0, "max": 0.5, "default": 0.12, "step": 0.001,
             "label": "PSAD", "help": "PSA Density (ng/ml/ml)", "format": "%.3f"},
    "P2PSA（pg/ml）": {"min": 0.0, "max": 100.0, "default": 15.0, "step": 0.1,
                      "label": "P2PSA (pg/ml)", "help": "[-2]proPSA"},
    "计算得前列腺体积（ml）": {"min": 5.0, "max": 200.0, "default": 35.0, "step": 0.5,
                        "label": "Prostate Volume (ml)", "help": "Computed prostate volume"},
}


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_DIR / "best_model.pkl")
    with open(MODEL_DIR / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    return model, config


def main():
    st.set_page_config(
        page_title="PSA Gray Zone Prediction",
        page_icon="🔬",
        layout="wide",
    )

    st.title("🔬 PSA Gray Zone Malignancy Prediction")
    st.markdown(
        "Predict prostate biopsy outcome (malignant vs benign) using biomarkers "
        "in the PSA gray zone (4-10 ng/ml)."
    )

    model, config = load_model()
    cutoff = config["optimal_cutoff"]
    best_name = config["best_model_name"]

    st.sidebar.header("Model Info")
    st.sidebar.markdown(f"**Algorithm**: {best_name}")
    st.sidebar.markdown(f"**CV AUC**: {config['cv_auc']:.4f}")
    st.sidebar.markdown(f"**Optimal Cutoff**: {cutoff:.4f}")
    st.sidebar.markdown(f"**Training Samples**: 100 (18 malignant, 82 benign)")

    uses_5 = config.get("uses_5_features", False)
    feature_keys = list(FEATURE_RANGES.keys())
    if not uses_5:
        feature_keys = feature_keys[:4]

    st.header("Input Biomarker Values")
    n_cols = len(feature_keys)
    cols = st.columns(n_cols)
    feature_values = []

    for i, feat_name in enumerate(feature_keys):
        meta = FEATURE_RANGES[feat_name]
        with cols[i]:
            fmt = meta.get("format", None)
            kwargs = {}
            if fmt:
                kwargs["format"] = fmt
            val = st.number_input(
                meta["label"],
                min_value=meta["min"],
                max_value=meta["max"],
                value=meta["default"],
                step=meta["step"],
                help=meta["help"],
                **kwargs,
            )
            feature_values.append(val)

    st.markdown("---")

    if st.button("🔍 Predict", type="primary", use_container_width=True):
        X_input = np.array(feature_values).reshape(1, -1)
        prob = model.predict_proba(X_input)[0, 1]
        is_malignant = prob >= cutoff

        col_result, col_detail = st.columns([1, 1])

        with col_result:
            if is_malignant:
                st.error(f"### ⚠️ Malignant (High Risk)")
                st.metric("Malignancy Probability", f"{prob:.1%}")
            else:
                st.success(f"### ✅ Benign (Low Risk)")
                st.metric("Malignancy Probability", f"{prob:.1%}")

        with col_detail:
            st.markdown("**Classification Details**")
            st.markdown(f"- **Probability**: {prob:.4f}")
            st.markdown(f"- **Cutoff Threshold**: {cutoff:.4f}")
            st.markdown(f"- **Result**: {'Malignant' if is_malignant else 'Benign'}")
            st.markdown(f"- **Model**: {best_name}")

        st.progress(min(prob, 1.0))

    st.markdown("---")
    st.header("Model Visualizations")

    plot_files = {
        "AUC Comparison": "auc_comparison.png",
        "ROC Curves": "roc_curves.png",
        "Optimal Cutoff": "optimal_cutoff.png",
        "Calibration Curve": "calibration_curve.png",
        "Performance Summary": "performance_summary.png",
        "Feature Importance": "feature_importance.png",
    }

    tabs = st.tabs(list(plot_files.keys()))
    for tab, (name, filename) in zip(tabs, plot_files.items()):
        with tab:
            img_path = RESULTS_DIR / filename
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)
            else:
                st.warning(f"Plot not found: {filename}. Run generate_plots.py first.")

    st.markdown("---")
    st.caption(
        "⚠️ This tool is for research purposes only and should not replace clinical judgment. "
        "Model trained on 100 PSA gray zone cases with RepeatedStratifiedKFold cross-validation."
    )


if __name__ == "__main__":
    main()
