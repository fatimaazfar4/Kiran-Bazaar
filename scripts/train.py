"""End-to-end training: features -> split -> CV tuning -> selection -> test -> save.

Run from the repo root:  python scripts/train.py
"""
from __future__ import annotations

import json
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score, brier_score_loss, classification_report,
                             confusion_matrix, f1_score, mean_absolute_error, precision_recall_curve,
                             precision_score, r2_score, recall_score, roc_auc_score, roc_curve,
                             root_mean_squared_error)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit

# Harmless library notices (SVC `probability` deprecation, joblib config propagation) would flood the console.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kiranbazaar import config as C  # noqa: E402
from kiranbazaar import features as F  # noqa: E402
from kiranbazaar import modeling as M  # noqa: E402

FIG = C.FIG_DIR
FIG.mkdir(parents=True, exist_ok=True)
C.MODELS_DIR.mkdir(exist_ok=True)
C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "reports"
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
CV = TimeSeriesSplit(n_splits=5)


def search(pipe, space, kind, scoring):
    if space is None:
        return pipe
    if kind == "grid":
        return GridSearchCV(pipe, space, cv=CV, scoring=scoring, n_jobs=-1)
    return RandomizedSearchCV(pipe, space, n_iter=20, cv=CV, scoring=scoring, n_jobs=-1,
                              random_state=C.RANDOM_STATE)


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(thr[i]), float(f1[i])


def clf_metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "f1": f1_score(y, yhat, zero_division=0), "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat, zero_division=0)}


def main():
    t0 = time.time()
    prices, markets = F.load_raw()
    feats = F.build_features(prices, markets)
    feats.to_csv(C.FEATURES_CSV, index=False)
    data = F.labelled(feats)
    tr, va, te = F.time_split(data)
    X = M.FEATURES
    Xtr, ytr = tr[X], tr[C.TARGET_CLF].astype(int)
    Xva, yva = va[X], va[C.TARGET_CLF].astype(int)
    Xte, yte = te[X], te[C.TARGET_CLF].astype(int)
    split_info = {name: {"rows": len(d), "start": str(d.date.min().date()), "end": str(d.date.max().date()),
                         "spike_rate": round(float(d[C.TARGET_CLF].mean()), 4)}
                  for name, d in [("train", tr), ("validation", va), ("test", te)]}
    print("split:", json.dumps(split_info, indent=1))

    # ------------------------------------------------------------------ classification
    rows, fitted, val_probs = [], {}, {}
    # Baseline 1: always predict the training prior (no skill)
    base = M.baseline_classifier().fit(Xtr, ytr)
    pb = base.predict_proba(Xva)[:, 1]
    rows.append({"model": "Baseline: class prior", "cv_roc_auc": np.nan, "train_roc_auc": 0.5,
                 **{f"val_{k}": v for k, v in clf_metrics(yva, pb, 0.5).items()}, "best_params": "-", "fit_s": 0})
    # Baseline 2: persistence rule ("it jumped last month, so it will jump again")
    persist_va = (va["ret_1"].fillna(0) > C.SPIKE_THRESHOLD).astype(float).values
    rows.append({"model": "Baseline: persistence rule", "cv_roc_auc": np.nan,
                 "train_roc_auc": roc_auc_score(ytr, (tr["ret_1"].fillna(0) > C.SPIKE_THRESHOLD)),
                 **{f"val_{k}": v for k, v in clf_metrics(yva, persist_va, 0.5).items()}, "best_params": "-",
                 "fit_s": 0})

    for name, (pipe, space, kind) in M.classifier_zoo().items():
        s = time.time()
        est = search(pipe, space, kind, "roc_auc").fit(Xtr, ytr)
        model = est.best_estimator_
        p = model.predict_proba(Xva)[:, 1]
        val_probs[name] = p
        fitted[name] = (model, est.best_params_)
        thr, _ = best_f1_threshold(yva, p)
        m = clf_metrics(yva, p, thr)
        rows.append({"model": name, "cv_roc_auc": est.best_score_,
                     "train_roc_auc": roc_auc_score(ytr, model.predict_proba(Xtr)[:, 1]),
                     **{f"val_{k}": v for k, v in m.items()},
                     "best_params": json.dumps({k.replace("model__", ""): (round(v, 4) if isinstance(v, float) else v)
                                                for k, v in est.best_params_.items()}),
                     "fit_s": round(time.time() - s, 1)})
        print(f"{name:28s} cv={est.best_score_:.3f} val_auc={m['roc_auc']:.3f} val_f1={m['f1']:.3f} "
              f"({time.time() - s:.0f}s)")

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "classification_comparison_validation.csv", index=False)

    # ---- select on VALIDATION ROC-AUC, tune decision threshold on validation (max F1)
    cand = table[~table.model.str.startswith("Baseline")].sort_values("val_roc_auc", ascending=False)
    best_name = cand.iloc[0]["model"]
    best_model, best_params = fitted[best_name]
    print("selected:", best_name)

    # ---- probability calibration (Platt / sigmoid), decided on VALIDATION data.
    # class_weight="balanced" inflates predicted probabilities, so "60 % chance" would not mean 60 %.
    # CalibratedClassifierCV with forward-chaining folds maps scores back to real-world frequencies.
    def calibrated(est):
        return CalibratedClassifierCV(clone(est), method="sigmoid", cv=TimeSeriesSplit(n_splits=5))

    cal_tr = calibrated(best_model).fit(Xtr, ytr)
    p_raw_va, p_cal_va = val_probs[best_name], cal_tr.predict_proba(Xva)[:, 1]
    calib_info = {
        "method": "sigmoid (Platt scaling), CalibratedClassifierCV with 5-fold TimeSeriesSplit",
        "val_brier_raw": round(float(brier_score_loss(yva, p_raw_va)), 4),
        "val_brier_calibrated": round(float(brier_score_loss(yva, p_cal_va)), 4),
        "val_mean_prob_raw": round(float(p_raw_va.mean()), 4),
        "val_mean_prob_calibrated": round(float(p_cal_va.mean()), 4),
        "val_actual_spike_rate": round(float(yva.mean()), 4),
    }
    print("calibration (validation):", calib_info)
    threshold, _ = best_f1_threshold(yva, p_cal_va)
    print("decision threshold (F1-optimal on validation, calibrated):", round(threshold, 3))

    # ---- refit the calibrated model on train+validation, evaluate ONCE on test
    trva = pd.concat([tr, va])
    final = calibrated(best_model).fit(trva[X], trva[C.TARGET_CLF].astype(int))
    pte = final.predict_proba(Xte)[:, 1]
    test_m = clf_metrics(yte, pte, threshold)
    persist_te = (te["ret_1"].fillna(0) > C.SPIKE_THRESHOLD).astype(float).values
    test_table = pd.DataFrame([
        {"model": f"{best_name}, calibrated (final)", **clf_metrics(yte, pte, threshold)},
        {"model": "Baseline: persistence rule", **clf_metrics(yte, persist_te, 0.5)},
        {"model": "Baseline: class prior", **clf_metrics(yte, np.full(len(yte), ytr.mean()), 0.5)},
    ])
    test_table.to_csv(RESULTS / "classification_test.csv", index=False)
    print(test_table.round(3).to_string())
    report_txt = classification_report(yte, (pte >= threshold).astype(int), target_names=["no spike", "spike"])
    cm = confusion_matrix(yte, (pte >= threshold).astype(int))
    print(report_txt)

    # per-commodity test performance (error analysis)
    te_an = te.assign(prob=pte, pred=(pte >= threshold).astype(int))
    per_c = te_an.groupby("commodity").apply(
        lambda g: pd.Series({"rows": len(g), "spike_rate": g[C.TARGET_CLF].mean(),
                             "roc_auc": roc_auc_score(g[C.TARGET_CLF], g.prob) if g[C.TARGET_CLF].nunique() > 1 else np.nan,
                             "f1": f1_score(g[C.TARGET_CLF], g.pred, zero_division=0)}), include_groups=False)
    per_c.round(3).to_csv(RESULTS / "test_by_commodity.csv")

    # ------------------------------------------------------------------ figures
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, p in val_probs.items():
        fpr, tpr, _ = roc_curve(yva, p)
        ax[0].plot(fpr, tpr, lw=1.4, label=f"{name} ({roc_auc_score(yva, p):.2f})")
        pr, rc, _ = precision_recall_curve(yva, p)
        ax[1].plot(rc, pr, lw=1.4, label=name)
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(title="ROC curves (validation)", xlabel="False positive rate", ylabel="True positive rate")
    ax[1].axhline(yva.mean(), color="k", ls="--", lw=0.8, label="no-skill")
    ax[1].set(title="Precision-Recall (validation)", xlabel="Recall", ylabel="Precision")
    ax[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "10_roc_pr_validation.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 4))
    ConfusionMatrixDisplay(cm, display_labels=["no spike", "spike"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Test confusion matrix\n{best_name}, threshold={threshold:.2f}")
    fig.tight_layout(); fig.savefig(FIG / "11_confusion_matrix_test.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    for lab, p, col in [("before calibration", p_raw_va, "#c0504d"), ("after calibration", p_cal_va, "#2b5797")]:
        fy, fx = calibration_curve(yva, p, n_bins=10, strategy="quantile")
        ax.plot(fx, fy, marker="o", color=col, label=f"{lab} (Brier {brier_score_loss(yva, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfectly calibrated")
    ax.set(title=f"Calibration on validation: {best_name}", xlabel="predicted probability of spike",
           ylabel="actual share of spikes", xlim=(0, 1), ylim=(0, 1)); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "16_calibration_validation.png"); plt.close(fig)

    t = table[~table.model.str.startswith("Baseline")].sort_values("val_roc_auc")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    yy = np.arange(len(t))
    ax.barh(yy - 0.2, t.train_roc_auc, 0.4, label="train", color="#9db4d8")
    ax.barh(yy + 0.2, t.val_roc_auc, 0.4, label="validation", color="#2b5797")
    ax.set_yticks(yy, t.model); ax.set_xlim(0.5, 1.0); ax.set_xlabel("ROC-AUC")
    ax.set_title("Over/under-fitting check: train vs validation ROC-AUC"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "12_train_vs_val_auc.png"); plt.close(fig)

    pi = permutation_importance(final, Xte, yte, scoring="roc_auc", n_repeats=5, random_state=C.RANDOM_STATE,
                                n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=X).sort_values()
    imp.to_csv(RESULTS / "permutation_importance_test.csv")
    fig, ax = plt.subplots(figsize=(7, 6))
    imp.tail(15).plot.barh(ax=ax, color="#2b5797")
    ax.set(title="Permutation importance (drop in test ROC-AUC)", xlabel="mean decrease in ROC-AUC")
    fig.tight_layout(); fig.savefig(FIG / "13_permutation_importance.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    pc = per_c.sort_values("roc_auc")
    ax.barh(pc.index, pc.roc_auc, color="#2b5797"); ax.axvline(0.5, color="k", ls="--", lw=0.8)
    ax.set(title="Test ROC-AUC by commodity (error analysis)", xlabel="ROC-AUC", xlim=(0.3, 1))
    fig.tight_layout(); fig.savefig(FIG / "14_test_auc_by_commodity.png"); plt.close(fig)

    # ------------------------------------------------------------------ regression (expected % change)
    ytr_r, yva_r, yte_r = tr[C.TARGET_REG], va[C.TARGET_REG], te[C.TARGET_REG]
    reg_rows, reg_fit = [], {}
    for name, (pipe, space, kind) in M.regressor_zoo().items():
        est = search(pipe, space, kind, "neg_mean_absolute_error").fit(tr[X], ytr_r)
        model = est.best_estimator_ if hasattr(est, "best_estimator_") else est
        pv = model.predict(va[X])
        reg_fit[name] = model
        reg_rows.append({"model": name, "train_mae": mean_absolute_error(ytr_r, model.predict(tr[X])),
                         "val_mae": mean_absolute_error(yva_r, pv), "val_rmse": root_mean_squared_error(yva_r, pv),
                         "val_r2": r2_score(yva_r, pv),
                         "best_params": json.dumps(getattr(est, "best_params_", {}), default=str)})
        print(f"{name:28s} val_mae={reg_rows[-1]['val_mae']:.4f} r2={reg_rows[-1]['val_r2']:.3f}")
    reg_table = pd.DataFrame(reg_rows)
    reg_table.to_csv(RESULTS / "regression_comparison_validation.csv", index=False)
    reg_best = reg_table[~reg_table.model.str.startswith("Baseline")].sort_values("val_mae").iloc[0]["model"]
    reg_final = clone(reg_fit[reg_best]).fit(trva[X], trva[C.TARGET_REG])
    pr_te = reg_final.predict(Xte)
    base_te = np.full(len(yte_r), trva[C.TARGET_REG].mean())
    reg_test = pd.DataFrame([
        {"model": f"{reg_best} (final)", "mae": mean_absolute_error(yte_r, pr_te),
         "rmse": root_mean_squared_error(yte_r, pr_te), "r2": r2_score(yte_r, pr_te)},
        {"model": "Baseline: train mean", "mae": mean_absolute_error(yte_r, base_te),
         "rmse": root_mean_squared_error(yte_r, base_te), "r2": r2_score(yte_r, base_te)},
        {"model": "Baseline: no change (0%)", "mae": mean_absolute_error(yte_r, 0 * base_te),
         "rmse": root_mean_squared_error(yte_r, 0 * base_te), "r2": r2_score(yte_r, 0 * base_te)},
    ])
    reg_test.to_csv(RESULTS / "regression_test.csv", index=False)
    print(reg_test.round(4).to_string())

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(yte_r * 100, pr_te * 100, s=6, alpha=0.35, color="#2b5797")
    lim = [-20, 30]; ax.plot(lim, lim, "k--", lw=0.8); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set(title=f"Regression on test: {reg_best}", xlabel="actual next-month change (%)",
           ylabel="predicted change (%)")
    fig.tight_layout(); fig.savefig(FIG / "15_regression_test_scatter.png"); plt.close(fig)

    # ------------------------------------------------------------------ save
    joblib.dump(final, C.MODELS_DIR / "spike_classifier.joblib", compress=3)
    joblib.dump(reg_final, C.MODELS_DIR / "price_change_regressor.joblib", compress=3)
    meta = {
        "project": "Kiran Bazaar - next-month food price spike prediction (Pakistan)",
        "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "spike_threshold": C.SPIKE_THRESHOLD,
        "classifier": best_name, "classifier_params": {k: str(v) for k, v in best_params.items()},
        "calibration": calib_info,
        "decision_threshold": threshold,
        "regressor": reg_best,
        "features": X, "categorical_features": C.CATEGORICAL_FEATURES, "numeric_features": C.NUMERIC_FEATURES,
        "split": split_info,
        "test_metrics_classifier": {k: round(float(v), 4) for k, v in test_m.items()},
        "test_confusion_matrix": cm.tolist(),
        "test_metrics_regressor": {k: round(float(v), 4) for k, v in reg_test.iloc[0, 1:].items()},
        "top_features": imp.sort_values(ascending=False).head(10).round(4).to_dict(),
        "data_last_month": str(feats.date.max().date()),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }
    (C.MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    (RESULTS / "classification_report_test.txt").write_text(report_txt)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
