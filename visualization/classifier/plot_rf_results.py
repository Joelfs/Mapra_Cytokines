#!/usr/bin/env python3
"""
Visualization of the Random Forest results (rf_condition_classifier.py,
--split stratified, and rf_timepoint_classifier.py from ml_models/).

For each of the two models, an overview figure is produced with:
  1. Validation accuracy per CV fold (+ mean) compared to the
     test-holdout accuracy
  2. Most important (decisive) DRVI factors, via SHAP (mean |SHAP value|,
     TreeExplainer) — same approach as plot_xgb_results.py, so RF and
     XGBoost are directly comparable. Computed here (not by the training
     scripts): loads the saved RF model (rf_*.joblib) and rebuilds the
     CV-pool pseudobulk feature matrix straight from the split h5ad, then
     runs shap.TreeExplainer on it.
  3. Labeled confusion matrix (rows=true class, columns=prediction)

Additionally:
  - rf_accuracy_summary.csv: CV mean/std vs. test accuracy, both models
  - rf_shap_values_<name>.csv: mean |SHAP value| per factor and class
    (+ overall mean), CV-pool only
  - rf_condition_stratified_confusion_readable.csv /
    rf_timepoint_confusion_readable.csv: long-format table
    (true_label, predicted_label, count), from which statements like
    "CCS groundtruth classified as ACS: 1" can be read directly.

Reads already existing result files from ml_models/ for accuracy/confusion
(no model re-run needed) plus the saved *.joblib models and the split h5ad
for the SHAP computation:
  - rf_cv_fold_metrics_stratified.csv / rf_timepoint_cv_fold_metrics.csv
  - rf_test_holdout_report_stratified.txt / rf_timepoint_test_holdout_report.txt
  - rf_condition_classifier_stratified.joblib / rf_timepoint_classifier.joblib
  - data_for_practicum_post_integration_curated_activedims_split.h5ad

Usage:
    conda run -n mapra_cytokines python plot_rf_results.py
"""

import io
import os
import re

import joblib
import numpy as np
import pandas as pd
import scanpy as sc
import shap
import matplotlib.pyplot as plt

ML_DIR = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/ml_models"
OUT_DIR = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/visualization/classifier"
DATA_H5AD = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/data/data_for_practicum_post_integration_curated_activedims_split.h5ad"
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = {
    "condition_stratified": {
        "title": "Condition (ACS / CCS / non-CCS)",
        "cv_metrics": os.path.join(ML_DIR, "rf_cv_fold_metrics_stratified.csv"),
        "test_report": os.path.join(ML_DIR, "rf_test_holdout_report_stratified.txt"),
        "model_path": os.path.join(ML_DIR, "rf_condition_classifier_stratified.joblib"),
        "level": "patient",
        "holdout_col": "holdout_stratified",
        "fold_col": "cv_fold_stratified",
    },
    "timepoint": {
        "title": "Timepoint (TP1-TP4, ACS_sterile)",
        "cv_metrics": os.path.join(ML_DIR, "rf_timepoint_cv_fold_metrics.csv"),
        "test_report": os.path.join(ML_DIR, "rf_timepoint_test_holdout_report.txt"),
        "model_path": os.path.join(ML_DIR, "rf_timepoint_classifier.joblib"),
        "level": "sample",
        "holdout_col": "holdout_timepoint",
        "fold_col": "cv_fold_timepoint",
    },
}

TOP_N_FACTORS = 15


def parse_test_report(path):
    """Extracts test accuracy + confusion matrix from an rf_*_test_holdout_report*.txt"""
    with open(path) as f:
        text = f.read()

    acc_match = re.search(r"^\s*accuracy\s+([0-9.]+)\s+(\d+)\s*$", text, flags=re.MULTILINE)
    test_accuracy = float(acc_match.group(1))
    n_test = int(acc_match.group(2))

    cm_header_idx = text.index("Confusion matrix")
    cm_text = text[cm_header_idx:].split("\n", 1)[1]
    cm_df = pd.read_csv(io.StringIO(cm_text), sep=r"\s+", index_col=0)
    cm_df.index = cm_df.index.astype(str)
    cm_df.columns = cm_df.columns.astype(str)

    return test_accuracy, n_test, cm_df


def confusion_to_readable(cm_df):
    """Long format: true_label, predicted_label, count (+ is_correct)."""
    rows = []
    for true_label in cm_df.index:
        for pred_label in cm_df.columns:
            count = int(cm_df.loc[true_label, pred_label])
            rows.append({
                "true_label": true_label,
                "predicted_label": pred_label,
                "count": count,
                "is_correct": true_label == pred_label,
                "description": f"{true_label} groundtruth classified as {pred_label}",
            })
    return pd.DataFrame(rows)


def build_pseudobulk_features(adata, level, holdout_col, fold_col):
    """Rebuilds the same CV-pool pseudobulk matrix the training script used
    (patient-level for 'patient', sample.timepoint-level for 'sample'),
    restricted to rows with holdout != 'excluded'."""
    factors = list(adata.uns["X_drvi_active_dims"])
    sample_id = adata.obs["sample_id"].astype(str)
    group_key = sample_id if level == "sample" else sample_id.str.split(".", n=1).str[0]

    feat_df = pd.DataFrame(np.asarray(adata.obsm["X_drvi"]), columns=factors, index=adata.obs_names)
    feat_df["_group"] = group_key.values
    feat_df["holdout"] = adata.obs[holdout_col].astype(str).values
    feat_df["cv_fold"] = adata.obs[fold_col].values

    agg = feat_df.groupby("_group", observed=True).agg(
        {**{f: "mean" for f in factors}, "holdout": "first", "cv_fold": "first"}
    )
    agg = agg[agg["holdout"] != "excluded"]
    return agg, factors


def compute_shap_importances(name, cfg, adata):
    """SHAP feature importance (TreeExplainer, not Gini) on the CV-pool —
    same method as xgb_condition_classifier.py / plot_xgb_results.py."""
    print(f"\n=== SHAP values ({name}) ===")
    bundle = joblib.load(cfg["model_path"])
    model = bundle["model"]
    classes = bundle["classes"]

    pseudobulk, factors = build_pseudobulk_features(
        adata, cfg["level"], cfg["holdout_col"], cfg["fold_col"]
    )
    cv_mask = pseudobulk["holdout"] == "cv"
    X_cv = pseudobulk.loc[cv_mask, factors]
    print(f"CV-pool for SHAP: {X_cv.shape[0]} rows x {X_cv.shape[1]} factors")

    explainer = shap.TreeExplainer(model)
    shap_values = np.asarray(explainer.shap_values(X_cv))
    # multi-class -> one (n_samples, n_features) matrix per class, as a list,
    # OR a (n_samples, n_features, n_classes) array, depending on shap version.
    if shap_values.ndim == 3:
        # (n_samples, n_features, n_classes) -> (n_classes, n_samples, n_features)
        shap_values = np.moveaxis(shap_values, -1, 0)
    mean_abs_per_class = np.stack([np.abs(sv).mean(axis=0) for sv in shap_values])

    shap_df = pd.DataFrame(mean_abs_per_class.T, index=factors, columns=classes)
    shap_df["mean_abs_shap"] = shap_df.mean(axis=1)
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)

    out_path = os.path.join(OUT_DIR, f"rf_shap_values_{name}.csv")
    shap_df.to_csv(out_path)
    print(f"Saved: {out_path}")

    print("Top 10 factors (mean |SHAP value|):")
    print(shap_df["mean_abs_shap"].head(10))

    return shap_df["mean_abs_shap"]


def plot_model(name, cfg, adata):
    cv_metrics = pd.read_csv(cfg["cv_metrics"])
    importances = compute_shap_importances(name, cfg, adata)
    test_accuracy, n_test, cm_df = parse_test_report(cfg["test_report"])

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    fig.suptitle(f"Random Forest, {cfg['title']}", fontsize=14, fontweight="bold")

    # ── Panel 1: Validation (CV fold) vs. test accuracy ─────────────────────
    ax = axes[0]
    fold_labels = [f"Fold {int(f)}" for f in cv_metrics["fold"]]
    ax.bar(fold_labels, cv_metrics["accuracy"], color="#4C72B0", label="Validation (CV fold)")
    test_x = len(fold_labels)
    ax.bar([test_x], [test_accuracy], color="#C44E52",
           label=f"Test-Holdout (n={n_test})")
    ax.set_xticks(list(range(len(fold_labels))) + [test_x])
    ax.set_xticklabels(fold_labels + ["Test"], rotation=0)
    for i, v in enumerate(cv_metrics["accuracy"]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.text(test_x, test_accuracy + 0.02, f"{test_accuracy:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation vs. Test Accuracy")
    ax.legend(fontsize=8, loc="lower right")

    # ── Panel 2: Feature importances (SHAP, mean |SHAP value|) ──────────────
    ax = axes[1]
    top = importances.head(TOP_N_FACTORS).sort_values(ascending=True)
    ax.barh(top.index, top.values, color="#55A868")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(f"Top {len(top)} decisive DRVI factors (SHAP)")

    # ── Panel 3: Labeled confusion matrix ─────────────────────────────────────
    ax = axes[2]
    im = ax.imshow(cm_df.values, cmap="Blues")
    ax.set_xticks(range(len(cm_df.columns)))
    ax.set_xticklabels(cm_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(cm_df.index)))
    ax.set_yticklabels(cm_df.index)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion Matrix (test holdout)")
    vmax = cm_df.values.max()
    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            val = cm_df.values[i, j]
            color = "white" if val > vmax / 2 else "black"
            weight = "bold" if cm_df.index[i] == cm_df.columns[j] else "normal"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontweight=weight)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(OUT_DIR, f"rf_{name}_overview.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

    readable = confusion_to_readable(cm_df)
    readable_path = os.path.join(OUT_DIR, f"rf_{name}_confusion_readable.csv")
    readable.to_csv(readable_path, index=False)
    print(f"Saved: {readable_path}")

    misclassified = readable[(~readable["is_correct"]) & (readable["count"] > 0)]
    if not misclassified.empty:
        print("  Misclassifications in the test holdout:")
        for _, row in misclassified.iterrows():
            print(f"    {row['description']}: {row['count']}")

    return {
        "model": name,
        "n_cv_folds": len(cv_metrics),
        "test_accuracy": test_accuracy,
        "n_test": n_test,
    }


def main():
    print("=== Load data (backed, for SHAP pseudobulk rebuild) ===")
    adata = sc.read_h5ad(DATA_H5AD, backed="r")
    print(f"{adata.n_obs} cells, X_drvi shape: {adata.obsm['X_drvi'].shape}")

    summary_rows = [plot_model(name, cfg, adata) for name, cfg in MODELS.items()]
    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUT_DIR, "rf_accuracy_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
