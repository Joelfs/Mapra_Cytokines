#!/usr/bin/env python3
"""
Visualization of the MLP classifier results (mlp_timepoint_classifier.py,
and mlp_condition_classifier.py --split stratified, once it exists, from
ml_models/).

Same layout as plot_rf_results.py / plot_xgb_results.py, so all three
models are directly comparable:
  1. Validation accuracy per CV fold (+ mean) compared to the
     test-holdout accuracy
  2. Most important (decisive) DRVI factors -- via Permutation Importance
     (mean drop in f1_macro when a factor is shuffled), NOT SHAP: MLPs
     don't have a TreeExplainer equivalent, and permutation importance is
     the standard model-agnostic choice for neural nets. Read straight
     from the CSV the training script already wrote (same pattern as
     plot_xgb_results.py -- no recomputation here).
  3. Labeled confusion matrix (rows=true class, columns=prediction)

Additionally:
  - mlp_accuracy_summary.csv: test accuracy summary, both models
  - mlp_condition_stratified_confusion_readable.csv /
    mlp_timepoint_confusion_readable.csv: long-format table
    (true_label, predicted_label, count), from which statements like
    "CCS groundtruth classified as ACS: 1" can be read directly.

Reads exclusively already existing result files from ml_models/ (no model
re-run needed):
  - mlp_cv_fold_metrics_stratified.csv / mlp_timepoint_cv_fold_metrics.csv
  - mlp_feature_importances_stratified.csv / mlp_timepoint_feature_importances.csv
  - mlp_test_holdout_report_stratified.txt / mlp_timepoint_test_holdout_report.txt

Note: the condition-task files don't exist yet (mlp_condition_classifier.py
still needs to be written/run) -- this script skips any model whose files
are missing rather than crashing, so it already works for timepoint alone.

Usage:
    conda run -n mapra_cytokines python plot_mlp_results.py
"""

import io
import os
import re

import pandas as pd
import matplotlib.pyplot as plt

ML_DIR = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/ml_models"
OUT_DIR = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/visualization/classifier"
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = {
    "condition_stratified": {
        "title": "Condition (ACS / CCS / non-CCS)",
        "cv_metrics": os.path.join(ML_DIR, "mlp_cv_fold_metrics_stratified.csv"),
        "importances": os.path.join(ML_DIR, "mlp_feature_importances_stratified.csv"),
        "test_report": os.path.join(ML_DIR, "mlp_test_holdout_report_stratified.txt"),
    },
    "timepoint": {
        "title": "Timepoint (TP1-TP4, ACS_sterile)",
        "cv_metrics": os.path.join(ML_DIR, "mlp_timepoint_cv_fold_metrics.csv"),
        "importances": os.path.join(ML_DIR, "mlp_timepoint_feature_importances.csv"),
        "test_report": os.path.join(ML_DIR, "mlp_timepoint_test_holdout_report.txt"),
    },
}

TOP_N_FACTORS = 15


def parse_test_report(path):
    """Extracts test accuracy + confusion matrix from an mlp_*_test_holdout_report*.txt"""
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


def plot_model(name, cfg):
    missing = [p for p in (cfg["cv_metrics"], cfg["importances"], cfg["test_report"]) if not os.path.exists(p)]
    if missing:
        print(f"\n=== {name}: SKIPPED (missing files) ===")
        for p in missing:
            print(f"  not found: {p}")
        return None

    cv_metrics = pd.read_csv(cfg["cv_metrics"])
    importances = pd.read_csv(cfg["importances"], index_col=0)["importance"]
    test_accuracy, n_test, cm_df = parse_test_report(cfg["test_report"])

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    fig.suptitle(f"MLP, {cfg['title']}", fontsize=14, fontweight="bold")

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

    # ── Panel 2: Feature importances (Permutation Importance) ───────────────
    ax = axes[1]
    top = importances.sort_values(ascending=False).head(TOP_N_FACTORS).sort_values(ascending=True)
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
    out_path = os.path.join(OUT_DIR, f"mlp_{name}_overview.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

    readable = confusion_to_readable(cm_df)
    readable_path = os.path.join(OUT_DIR, f"mlp_{name}_confusion_readable.csv")
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
    summary_rows = [row for name, cfg in MODELS.items() if (row := plot_model(name, cfg)) is not None]
    if not summary_rows:
        print("\nNo models plotted -- no result files found yet.")
        return
    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUT_DIR, "mlp_accuracy_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()