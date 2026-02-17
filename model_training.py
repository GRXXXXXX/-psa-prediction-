"""
PSA Gray Zone Malignancy Prediction - Model Training Pipeline v5
Strategy: Stable Optuna (RepeatedCV objective) + 5 features + Cross-ensemble
Target: AUC > 0.90
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "完整版数据.xlsx"
MODEL_DIR = BASE_DIR / "model"
RESULTS_DIR = BASE_DIR / "results"

FEATURE_COLS_4 = ["f-PSA/t-PSA(%)", "PHI", "PSAD", "P2PSA（pg/ml）"]
FEATURE_COLS_5 = ["f-PSA/t-PSA(%)", "PHI", "PSAD", "P2PSA（pg/ml）", "计算得前列腺体积（ml）"]
TARGET_COL = "穿刺结果（恶性/良性）"
RANDOM_STATE = 42
N_OPTUNA_TRIALS = 200

# Optuna uses 2-repeat 5-fold for stable optimization (10 evals/trial)
OPTUNA_CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)
# Final evaluation: 20-repeat 5-fold for very stable estimates
EVAL_CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=RANDOM_STATE)


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select specific columns from X so all models share the same input."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, self.columns]


def engineer_features(X_raw):
    """Domain features from 4 biomarkers (for app.py compatibility)."""
    ft = X_raw[:, 0]
    phi = X_raw[:, 1]
    psad = X_raw[:, 2]
    p2 = X_raw[:, 3]
    eps = 1e-8
    features = [
        ft, phi, psad, p2,
        np.log1p(phi),
        np.log1p(p2),
        phi * psad,
        phi * (1 - ft),
        p2 * psad,
        p2 / (ft + eps),
        phi / (ft + eps),
        phi ** 2,
        phi * psad * (1 - ft),
    ]
    return np.column_stack(features)


def load_data():
    df = pd.read_excel(DATA_PATH)
    X5 = df[FEATURE_COLS_5].values.astype(float)
    y = (df[TARGET_COL] == "恶性").astype(int).values
    return X5, y, df


def compute_single_indicator_aucs(X5, y):
    aucs = {}
    for i, name in enumerate(FEATURE_COLS_5):
        vals = X5[:, i]
        auc = roc_auc_score(y, vals)
        if auc < 0.5:
            auc = roc_auc_score(y, -vals)
        aucs[name] = round(auc, 4)
    return aucs


def find_optimal_cutoff(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx], tpr[best_idx], fpr[best_idx]


def make_4feat_poly_pipeline(clf, degree=2, interact_only=True):
    """4 original features → poly → classifier. Input: X5 (5 cols)."""
    return Pipeline([
        ("sel", ColumnSelector([0, 1, 2, 3])),
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=degree, interaction_only=interact_only,
                                     include_bias=False)),
        ("clf", clf),
    ])


def make_5feat_pipeline(clf):
    """All 5 features → classifier. Input: X5."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


def make_5feat_poly_pipeline(clf, degree=2, interact_only=True):
    """All 5 features + poly → classifier. Input: X5."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=degree, interaction_only=interact_only,
                                     include_bias=False)),
        ("clf", clf),
    ])


def optuna_optimize(name, X, y, make_model_fn, n_trials=N_OPTUNA_TRIALS):
    """Generic Optuna optimization with stable RepeatedCV objective."""
    def objective(trial):
        model = make_model_fn(trial)
        scores = cross_val_score(model, X, y, cv=OPTUNA_CV, scoring="roc_auc")
        return np.mean(scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials)
    best_model = make_model_fn(study.best_trial)
    return best_model, study.best_value


def get_all_strategies(sw):
    """Define all model strategies. Each returns (name, make_model_fn, desc)."""
    strategies = []

    # === A: 4-feature + poly (v2 winner approach) ===
    def lr_4poly(trial):
        C = trial.suggest_float("C", 1e-3, 50.0, log=True)
        deg = trial.suggest_int("degree", 2, 3)
        inter = trial.suggest_categorical("interact_only", [True, False])
        return make_4feat_poly_pipeline(
            LogisticRegression(C=C, class_weight="balanced", max_iter=5000,
                               random_state=RANDOM_STATE),
            degree=deg, interact_only=inter,
        )
    strategies.append(("LR-4Poly", lr_4poly, "LR + Poly on 4 features"))

    def svm_4poly(trial):
        C = trial.suggest_float("C", 1e-2, 200.0, log=True)
        gamma = trial.suggest_float("gamma", 1e-4, 10.0, log=True)
        deg = trial.suggest_int("degree", 2, 3)
        return make_4feat_poly_pipeline(
            SVC(C=C, kernel="rbf", gamma=gamma, class_weight="balanced",
                probability=True, random_state=RANDOM_STATE),
            degree=deg, interact_only=True,
        )
    strategies.append(("SVM-4Poly", svm_4poly, "SVM + Poly on 4 features"))

    def gbm_4poly(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 30, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 4),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 20),
        }
        return make_4feat_poly_pipeline(
            GradientBoostingClassifier(**p, random_state=RANDOM_STATE),
            degree=2, interact_only=True,
        )
    strategies.append(("GBM-4Poly", gbm_4poly, "GBM + Poly on 4 features"))

    def xgb_4poly(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 30, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 4),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        }
        return make_4feat_poly_pipeline(
            XGBClassifier(**p, scale_pos_weight=sw, eval_metric="logloss",
                          random_state=RANDOM_STATE),
            degree=2, interact_only=True,
        )
    strategies.append(("XGB-4Poly", xgb_4poly, "XGBoost + Poly on 4 features"))

    # === B: 5 features + poly (adding prostate volume) ===
    def lr_5poly(trial):
        C = trial.suggest_float("C", 1e-3, 50.0, log=True)
        return make_5feat_poly_pipeline(
            LogisticRegression(C=C, class_weight="balanced", max_iter=5000,
                               random_state=RANDOM_STATE),
            degree=2, interact_only=True,
        )
    strategies.append(("LR-5Poly", lr_5poly, "LR + Poly on 5 features"))

    def svm_5poly(trial):
        C = trial.suggest_float("C", 1e-2, 200.0, log=True)
        gamma = trial.suggest_float("gamma", 1e-4, 10.0, log=True)
        return make_5feat_poly_pipeline(
            SVC(C=C, kernel="rbf", gamma=gamma, class_weight="balanced",
                probability=True, random_state=RANDOM_STATE),
            degree=2, interact_only=True,
        )
    strategies.append(("SVM-5Poly", svm_5poly, "SVM + Poly on 5 features"))

    # === C: 5 features, tree-based (no poly needed) ===
    def gbm_5(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 30, 400),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
        return make_5feat_pipeline(
            GradientBoostingClassifier(**p, random_state=RANDOM_STATE),
        )
    strategies.append(("GBM-5", gbm_5, "GBM on 5 features"))

    def xgb_5(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 30, 400),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
        return make_5feat_pipeline(
            XGBClassifier(**p, scale_pos_weight=sw, eval_metric="logloss",
                          random_state=RANDOM_STATE),
        )
    strategies.append(("XGB-5", xgb_5, "XGBoost on 5 features"))

    def lgbm_5(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 30, 400),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 3, 25),
            "num_leaves": trial.suggest_int("num_leaves", 4, 31),
        }
        return make_5feat_pipeline(
            LGBMClassifier(**p, scale_pos_weight=sw, random_state=RANDOM_STATE, verbose=-1),
        )
    strategies.append(("LGBM-5", lgbm_5, "LightGBM on 5 features"))

    def rf_5(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
        return make_5feat_pipeline(
            RandomForestClassifier(**p, class_weight="balanced",
                                    random_state=RANDOM_STATE),
        )
    strategies.append(("RF-5", rf_5, "RandomForest on 5 features"))

    return strategies


def evaluate_model(model, X, y):
    """Evaluate with heavy repeated CV."""
    scores = cross_val_score(model, X, y, cv=EVAL_CV, scoring="roc_auc")
    return np.mean(scores), np.std(scores)


def get_cv_probs(model, X, y):
    """Get cross-validated probabilities for plot generation."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    return probs


def main():
    print("=" * 70)
    print("PSA Gray Zone - v5: Stable Optuna + Extended Features + Ensembles")
    print("=" * 70)

    print("\n[1/7] Loading data...")
    X5, y, df = load_data()
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    sw = n_neg / n_pos
    print(f"  Samples: {len(y)} (Malignant: {n_pos}, Benign: {n_neg})")
    print(f"  Features: {FEATURE_COLS_5}")

    print("\n[2/7] Single indicator AUCs...")
    single_aucs = compute_single_indicator_aucs(X5, y)
    for name, auc in single_aucs.items():
        print(f"  {name}: {auc:.4f}")

    print(f"\n[3/7] Optuna optimization ({N_OPTUNA_TRIALS} trials x 10 strategies)...")
    print(f"  Optuna CV: RepeatedStratifiedKFold(5, 2) for stable objective")
    strategies = get_all_strategies(sw)
    models = {}

    for name, make_fn, desc in strategies:
        print(f"  {name:15s} ({desc})...", end="", flush=True)
        model, best_auc = optuna_optimize(name, X5, y, make_fn)
        models[name] = model
        print(f" Optuna AUC: {best_auc:.4f}")

    print(f"\n[4/7] Evaluating all models (RepeatedCV 5-fold x 20 repeats)...")
    results = {}
    for name, model in models.items():
        auc_mean, auc_std = evaluate_model(model, X5, y)
        results[name] = {"auc_mean": round(auc_mean, 4), "auc_std": round(auc_std, 4)}

    print("\n" + "-" * 55)
    print(f"{'Model':<16} {'AUC':>8} {'Std':>8}")
    print("-" * 55)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["auc_mean"]):
        print(f"  {name:<14} {r['auc_mean']:>8.4f} {r['auc_std']:>8.4f}")
    print("-" * 55)

    print(f"\n[5/7] Building ensembles...")
    ranked = sorted(results.keys(), key=lambda k: -results[k]["auc_mean"])
    ensemble_results = {}

    for n in [3, 4, 5, 6]:
        top = ranked[:n]
        if len(top) < n:
            continue
        ens_name = f"Voting-{n}"
        ens = VotingClassifier(
            estimators=[(name, models[name]) for name in top], voting="soft",
        )
        auc_mean, auc_std = evaluate_model(ens, X5, y)
        ensemble_results[ens_name] = {
            "auc_mean": round(auc_mean, 4),
            "auc_std": round(auc_std, 4),
            "model": ens,
        }

    # Weighted voting (top-5)
    top5 = ranked[:5]
    weights = [results[n]["auc_mean"] for n in top5]
    ens = VotingClassifier(
        estimators=[(name, models[name]) for name in top5],
        voting="soft", weights=weights,
    )
    auc_mean, auc_std = evaluate_model(ens, X5, y)
    ensemble_results["WVoting-5"] = {
        "auc_mean": round(auc_mean, 4),
        "auc_std": round(auc_std, 4),
        "model": ens,
    }

    # Stacking
    for n in [3, 4, 5]:
        top = ranked[:n]
        if len(top) < n:
            continue
        ens_name = f"Stack-{n}"
        ens = StackingClassifier(
            estimators=[(name, models[name]) for name in top],
            final_estimator=LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=3000,
                random_state=RANDOM_STATE,
            ),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        )
        auc_mean, auc_std = evaluate_model(ens, X5, y)
        ensemble_results[ens_name] = {
            "auc_mean": round(auc_mean, 4),
            "auc_std": round(auc_std, 4),
            "model": ens,
        }

    print("\n  Ensemble results:")
    for name, r in sorted(ensemble_results.items(), key=lambda x: -x[1]["auc_mean"]):
        print(f"    {name:<16} AUC = {r['auc_mean']:.4f} +/- {r['auc_std']:.4f}")

    # Merge all results
    all_results = {**results}
    for k, v in ensemble_results.items():
        all_results[k] = {"auc_mean": v["auc_mean"], "auc_std": v["auc_std"]}

    print(f"\n[6/7] Final ranking...")
    print("\n" + "-" * 70)
    print(f"{'Rank':<5} {'Model':<20} {'AUC':>8} {'Std':>8}")
    print("-" * 70)
    ranked_all = sorted(all_results.items(), key=lambda x: -x[1]["auc_mean"])
    for i, (name, r) in enumerate(ranked_all, 1):
        print(f"  {i:<3}  {name:<18} {r['auc_mean']:>8.4f} {r['auc_std']:>8.4f}")
    print("-" * 70)

    best_name = ranked_all[0][0]
    best_auc = ranked_all[0][1]["auc_mean"]
    print(f"\n  Best: {best_name} (AUC = {best_auc:.4f})")

    print(f"\n[7/7] Training final model & saving...")
    if best_name in ensemble_results:
        best_model = ensemble_results[best_name]["model"]
    else:
        best_model = models[best_name]

    best_model.fit(X5, y)

    cv_probs = get_cv_probs(best_model, X5, y)
    cv_auc = roc_auc_score(y, cv_probs)
    cutoff, _, _ = find_optimal_cutoff(y, cv_probs)

    # Get metrics at optimal cutoff
    y_pred = (cv_probs >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc = (tp + tn) / len(y)

    print(f"  CV AUC (5-fold partition): {cv_auc:.4f}")
    print(f"  Optimal cutoff: {cutoff:.4f}")
    print(f"  Sensitivity: {sens:.4f}, Specificity: {spec:.4f}, Accuracy: {acc:.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    joblib.dump(best_model, MODEL_DIR / "best_model.pkl")

    # Top 6 for display
    display_results = {}
    for name, r in ranked_all[:6]:
        display_results[name] = r

    config = {
        "best_model_name": best_name,
        "optimal_cutoff": round(float(cutoff), 4),
        "feature_names_4": FEATURE_COLS_4,
        "feature_names_5": FEATURE_COLS_5,
        "uses_feature_engineering": False,
        "uses_5_features": True,
        "cv_auc": round(float(cv_auc), 4),
        "sensitivity": round(float(sens), 4),
        "specificity": round(float(spec), 4),
        "accuracy": round(float(acc), 4),
        "model_results": display_results,
        "single_indicator_aucs": single_aucs,
        "class_distribution": {"malignant": int(n_pos), "benign": int(n_neg)},
    }
    with open(MODEL_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    np.save(RESULTS_DIR / "cv_probs.npy", cv_probs)
    np.save(RESULTS_DIR / "y_true.npy", y)
    np.save(RESULTS_DIR / "X_data.npy", X5[:, :4])

    print(f"\n  Saved: model/best_model.pkl, model/config.json")
    print("\nDone! Run generate_plots.py next.")


if __name__ == "__main__":
    main()
