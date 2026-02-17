"""
PSA Gray Zone Malignancy Prediction - Desktop GUI
Lightweight tkinter app for EXE packaging.
"""

import json
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

import joblib
import numpy as np
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

FEATURES = [
    {"key": "f-PSA/t-PSA(%)", "label": "f/t PSA (%)", "min": 0.0, "max": 1.0, "default": 0.15},
    {"key": "PHI", "label": "PHI", "min": 0.0, "max": 150.0, "default": 35.0},
    {"key": "PSAD", "label": "PSAD (ng/ml/ml)", "min": 0.0, "max": 0.5, "default": 0.12},
    {"key": "P2PSA", "label": "P2PSA (pg/ml)", "min": 0.0, "max": 100.0, "default": 15.0},
    {"key": "Volume", "label": "Prostate Volume (ml)", "min": 5.0, "max": 200.0, "default": 35.0},
]


class PSAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PSA Gray Zone Malignancy Prediction")
        self.root.geometry("520x580")
        self.root.resizable(False, False)

        self.model, self.config = self._load_model()
        self.cutoff = self.config["optimal_cutoff"]
        self.entries = []

        self._build_ui()

    def _load_model(self):
        model = joblib.load(MODEL_DIR / "best_model.pkl")
        with open(MODEL_DIR / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        return model, config

    def _build_ui(self):
        title = tk.Label(self.root, text="PSA Gray Zone Malignancy Prediction",
                         font=("Arial", 16, "bold"), pady=10)
        title.pack()

        subtitle = tk.Label(self.root,
                            text=f"Model: {self.config['best_model_name']}  |  "
                                 f"CV AUC: {self.config['cv_auc']:.3f}  |  "
                                 f"Cutoff: {self.cutoff:.4f}",
                            font=("Arial", 10), fg="gray")
        subtitle.pack()

        sep = ttk.Separator(self.root, orient="horizontal")
        sep.pack(fill="x", padx=20, pady=10)

        input_label = tk.Label(self.root, text="Input Biomarker Values",
                               font=("Arial", 13, "bold"))
        input_label.pack(pady=(0, 5))

        uses_5 = self.config.get("uses_5_features", False)
        feature_list = FEATURES if uses_5 else FEATURES[:4]

        input_frame = tk.Frame(self.root)
        input_frame.pack(padx=40, pady=5, fill="x")

        for i, feat in enumerate(feature_list):
            row = tk.Frame(input_frame)
            row.pack(fill="x", pady=4)

            lbl = tk.Label(row, text=feat["label"], font=("Arial", 11), width=20, anchor="w")
            lbl.pack(side="left")

            var = tk.StringVar(value=str(feat["default"]))
            entry = ttk.Entry(row, textvariable=var, width=15, font=("Arial", 11))
            entry.pack(side="left", padx=(10, 0))

            hint = tk.Label(row, text=f"({feat['min']} - {feat['max']})",
                            font=("Arial", 9), fg="gray")
            hint.pack(side="left", padx=(8, 0))

            self.entries.append((feat, var))

        sep2 = ttk.Separator(self.root, orient="horizontal")
        sep2.pack(fill="x", padx=20, pady=15)

        btn = tk.Button(self.root, text="Predict", font=("Arial", 13, "bold"),
                        bg="#E74C3C", fg="white", activebackground="#C0392B",
                        activeforeground="white", width=20, height=2,
                        command=self._predict)
        btn.pack(pady=5)

        self.result_frame = tk.Frame(self.root)
        self.result_frame.pack(padx=40, pady=10, fill="x")

        self.result_label = tk.Label(self.result_frame, text="", font=("Arial", 14, "bold"))
        self.result_label.pack()

        self.prob_label = tk.Label(self.result_frame, text="", font=("Arial", 12))
        self.prob_label.pack()

        self.detail_label = tk.Label(self.result_frame, text="", font=("Arial", 10),
                                     fg="gray", justify="left")
        self.detail_label.pack(pady=(5, 0))

        footer = tk.Label(self.root,
                          text="For research purposes only. Not a substitute for clinical judgment.",
                          font=("Arial", 8), fg="gray")
        footer.pack(side="bottom", pady=5)

    def _predict(self):
        values = []
        for feat, var in self.entries:
            try:
                v = float(var.get())
            except ValueError:
                messagebox.showerror("Input Error",
                                     f"Invalid value for {feat['label']}. Please enter a number.")
                return
            if v < feat["min"] or v > feat["max"]:
                messagebox.showwarning("Range Warning",
                                       f"{feat['label']} = {v} is outside the expected range "
                                       f"({feat['min']} - {feat['max']}).")
            values.append(v)

        X_input = np.array(values).reshape(1, -1)
        prob = self.model.predict_proba(X_input)[0, 1]
        is_malignant = prob >= self.cutoff

        if is_malignant:
            self.result_label.config(text="Malignant (High Risk)", fg="#E74C3C")
        else:
            self.result_label.config(text="Benign (Low Risk)", fg="#27AE60")

        self.prob_label.config(text=f"Malignancy Probability: {prob:.1%}")

        self.detail_label.config(
            text=f"Probability: {prob:.4f}  |  Cutoff: {self.cutoff:.4f}  |  "
                 f"Model: {self.config['best_model_name']}"
        )


def main():
    root = tk.Tk()
    PSAApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
