#!/usr/bin/env python3
"""
Two comparison figures across the four classifier types (Logistic Regression,
Random Forest, XGBoost, MLP), for both prediction tasks (condition, timepoint):

1. feature_consensus.png -- how many of the 4 models rank a given active
   DRVI factor among their own top 10 most important features (by whatever
   importance measure that model uses: LR |coefficient|, RF impurity
   importance, XGB mean |SHAP|, MLP permutation importance). One panel per
   task, factors sorted by consensus count (how many models agree), stacked
   bar colored by which model(s) contributed.

2. classifier_score_comparison.png -- test-holdout accuracy + macro-F1 per
   model, at the SAME evaluation unit for all 4 models (one patient for
   condition, one sample/timepoint for timepoint) so the comparison is
   apples-to-apples even though training differs (LR trains on pseudobulked
   patient/sample rows; RF/XGB/MLP train per-cell and aggregate to
   patient/sample level via majority vote for this number). One panel per
   task.

Reads only already-computed result files -- no model is retrained or
re-evaluated here (MLP's patient/sample-level numbers were computed once by
mlp_patient_level_eval.py, which loads the saved .joblib and does NOT
retrain).
"""
import os
import re
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

ML_DIR = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/ml_models"
# Cell-level reruns (2026-09-07), replacing the earlier per-patient pseudobulk
# LR results -- matches the cell-level setup RF/XGB/MLP already moved to.
LOGISTIC_COND_DIR = "/vol/disk/ubuntu/master_practicum_cytokines/norman/linear_classifier_cellwise"
LOGISTIC_TP_DIR = "/vol/disk/ubuntu/master_practicum_cytokines/norman/logistic_timepoint_cellwise"
OUT_DIR = "/vol/disk/ubuntu/master_practicum_cytokines/lisa/classifier_comparison"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_ORDER = ["LR", "RF", "XGB", "MLP"]
# One blue family throughout, shading light-to-dark per model instead of
# unrelated hues -- the model identity still reads clearly (position + shade)
# without the plot looking like a mix of arbitrary colors.
MODEL_COLORS = {"LR": "#9ecae1", "RF": "#4292c6", "XGB": "#2171b5", "MLP": "#08306b"}
BAR_COLOR = "#2171b5"
TOP_N = 10

FEATURE_IMPORTANCE_FILES = {
    "condition": {
        "LR": os.path.join(LOGISTIC_COND_DIR, "feature_importances.csv"),
        "RF": os.path.join(ML_DIR, "rf_feature_importances_stratified.csv"),
        "XGB": os.path.join(ML_DIR, "xgb_feature_importances_stratified.csv"),
        "MLP": os.path.join(ML_DIR, "mlp_feature_importances_stratified.csv"),
    },
    "timepoint": {
        "LR": os.path.join(LOGISTIC_TP_DIR, "feature_importances.csv"),
        "RF": os.path.join(ML_DIR, "rf_timepoint_feature_importances.csv"),
        "XGB": os.path.join(ML_DIR, "xgb_timepoint_feature_importances.csv"),
        "MLP": os.path.join(ML_DIR, "mlp_timepoint_feature_importances.csv"),
    },
}

# Only LR, RF, and XGB actually compute SHAP values (mean |SHAP| per factor).
# MLP uses permutation importance instead -- not SHAP, so it's excluded here
# rather than mixed in with the real SHAP-based models.
SHAP_MODEL_ORDER = ["LR", "RF", "XGB"]
VIZ_DIR = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/visualization/classifier"
SHAP_FILES = {
    "condition": {
        "LR": os.path.join(LOGISTIC_COND_DIR, "shap_values.csv"),
        "RF": os.path.join(VIZ_DIR, "rf_shap_values_condition_stratified.csv"),
        "XGB": os.path.join(ML_DIR, "xgb_shap_values_stratified.csv"),
    },
    "timepoint": {
        "LR": os.path.join(LOGISTIC_TP_DIR, "shap_values.csv"),
        "RF": os.path.join(VIZ_DIR, "rf_shap_values_timepoint.csv"),
        "XGB": os.path.join(ML_DIR, "xgb_timepoint_shap_values.csv"),
    },
}


def parse_test_report(path):
    """accuracy + macro-f1 from a sklearn classification_report text block."""
    with open(path) as f:
        text = f.read()
    acc_match = re.search(r"^\s*accuracy\s+([0-9.]+)\s+(\d+)\s*$", text, flags=re.MULTILINE)
    accuracy = float(acc_match.group(1))
    n = int(acc_match.group(2))
    f1_match = re.search(r"^\s*macro avg\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)\s+\d+\s*$", text, flags=re.MULTILINE)
    f1_macro = float(f1_match.group(1))
    return accuracy, f1_macro, n


def load_scores():
    mlp_summary = pd.read_csv(os.path.join(ML_DIR, "mlp_patient_level_summary.csv")).set_index("task")

    rows = []
    lr_acc, lr_f1, lr_n = parse_test_report(os.path.join(LOGISTIC_COND_DIR, "test_holdout_report.txt"))
    rows.append({"task": "condition", "model": "LR", "accuracy": lr_acc, "f1_macro": lr_f1, "n": lr_n})
    rf_acc, rf_f1, rf_n = parse_test_report(os.path.join(ML_DIR, "rf_test_holdout_report_stratified.txt"))
    rows.append({"task": "condition", "model": "RF", "accuracy": rf_acc, "f1_macro": rf_f1, "n": rf_n})
    xgb_acc, xgb_f1, xgb_n = parse_test_report(os.path.join(ML_DIR, "xgb_test_holdout_report_stratified.txt"))
    rows.append({"task": "condition", "model": "XGB", "accuracy": xgb_acc, "f1_macro": xgb_f1, "n": xgb_n})
    m = mlp_summary.loc["condition"]
    rows.append({"task": "condition", "model": "MLP", "accuracy": m["accuracy"], "f1_macro": m["f1_macro"], "n": int(m["n"])})

    lr_acc, lr_f1, lr_n = parse_test_report(os.path.join(LOGISTIC_TP_DIR, "test_holdout_report.txt"))
    rows.append({"task": "timepoint", "model": "LR", "accuracy": lr_acc, "f1_macro": lr_f1, "n": lr_n})
    rf_acc, rf_f1, rf_n = parse_test_report(os.path.join(ML_DIR, "rf_timepoint_test_holdout_report.txt"))
    rows.append({"task": "timepoint", "model": "RF", "accuracy": rf_acc, "f1_macro": rf_f1, "n": rf_n})
    xgb_acc, xgb_f1, xgb_n = parse_test_report(os.path.join(ML_DIR, "xgb_timepoint_test_holdout_report.txt"))
    rows.append({"task": "timepoint", "model": "XGB", "accuracy": xgb_acc, "f1_macro": xgb_f1, "n": xgb_n})
    m = mlp_summary.loc["timepoint"]
    rows.append({"task": "timepoint", "model": "MLP", "accuracy": m["accuracy"], "f1_macro": m["f1_macro"], "n": int(m["n"])})

    return pd.DataFrame(rows)


METRIC_COLORS = {"accuracy": "#4C72B0", "f1_macro": "#C44E52"}


def plot_classifier_scores():
    df = load_scores()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, task, unit in zip(axes, ["condition", "timepoint"], ["patients", "samples"]):
        sub = df[df["task"] == task].set_index("model").reindex(MODEL_ORDER)
        x = np.arange(len(MODEL_ORDER))
        w = 0.35
        ax.bar(x - w / 2, sub["accuracy"], width=w, label="Accuracy", color=METRIC_COLORS["accuracy"])
        ax.bar(x + w / 2, sub["f1_macro"], width=w, label="Macro F1", color=METRIC_COLORS["f1_macro"])
        for i, m in enumerate(MODEL_ORDER):
            ax.text(i - w / 2, sub.loc[m, "accuracy"] + 0.02, f"{sub.loc[m, 'accuracy']:.2f}",
                    ha="center", fontsize=8)
            ax.text(i + w / 2, sub.loc[m, "f1_macro"] + 0.02, f"{sub.loc[m, 'f1_macro']:.2f}",
                    ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER)
        ax.set_ylim(0, 1.15)
        n = int(sub["n"].iloc[0])
        title = "Condition: ACS vs. CCS vs. non-CCS" if task == "condition" else "Timepoint: TP1 through TP4"
        ax.set_title(f"{title}\nHeld-out test set, n = {n} {unit}", fontsize=10, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        if task == "condition":
            ax.set_ylabel("Score")

    handles = [plt.Rectangle((0, 0), 1, 1, color=METRIC_COLORS["accuracy"], label="Accuracy"),
               plt.Rectangle((0, 0), 1, 1, color=METRIC_COLORS["f1_macro"], label="Macro F1")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Comparing the four classifiers: accuracy and macro F1 on the held-out test set",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    out_path = os.path.join(OUT_DIR, "classifier_score_comparison.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(OUT_DIR, "classifier_score_comparison.csv"), index=False)


def get_top_importance_sets(task):
    """Top-10 factors per model, by whatever importance measure that model
    computes: LR/RF/XGB use mean |SHAP value|, MLP uses permutation importance
    (it doesn't produce SHAP values)."""
    sets = {}
    for model, path in FEATURE_IMPORTANCE_FILES[task].items():
        s = pd.read_csv(path, index_col=0)["importance"]
        sets[model] = set(s.sort_values(ascending=False).head(TOP_N).index)
    return sets


def get_shap_top_sets(task):
    sets = {}
    for model, path in SHAP_FILES[task].items():
        s = pd.read_csv(path, index_col=0)["mean_abs_shap"]
        sets[model] = set(s.sort_values(ascending=False).head(TOP_N).index)
    return sets


def _sorted_labels(factors):
    return ", ".join(sorted(factors, key=lambda f: int(f.replace("DR", ""))))


def plot_shap_venn():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, task in zip(axes, ["condition", "timepoint"]):
        sets = get_shap_top_sets(task)
        lr, rf, xgb = sets["LR"], sets["RF"], sets["XGB"]
        v = venn3([lr, rf, xgb], set_labels=("LR", "RF", "XGB"),
                  set_colors=(MODEL_COLORS["LR"], MODEL_COLORS["RF"], MODEL_COLORS["XGB"]),
                  alpha=0.55, ax=ax)

        # region id -> the actual set of factors in just that region
        regions = {
            "100": lr - rf - xgb,
            "010": rf - lr - xgb,
            "001": xgb - lr - rf,
            "110": (lr & rf) - xgb,
            "101": (lr & xgb) - rf,
            "011": (rf & xgb) - lr,
            "111": lr & rf & xgb,
        }
        for region_id, factors in regions.items():
            label = v.get_label_by_id(region_id)
            if label is None:
                continue
            if factors:
                label.set_text(_sorted_labels(factors))
                label.set_fontsize(7)
            else:
                label.set_text("")

        for patch in v.patches:
            if patch is not None:
                patch.set_edgecolor("white")

        title = "Condition: ACS vs. CCS vs. non-CCS" if task == "condition" else "Timepoint: TP1 through TP4"
        ax.set_title(f"{title}\ntop-{TOP_N} factors by mean |SHAP|", fontsize=11, fontweight="bold")

    fig.suptitle("Which active DRVI factors do the SHAP-based models agree on?\n"
                 "Logistic regression, Random Forest, and XGBoost (MLP isn't shown here, since it doesn't use SHAP)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out_path = os.path.join(OUT_DIR, "shap_venn.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _intersections(sets, order=SHAP_MODEL_ORDER):
    """All non-empty intersections (by exact membership pattern), sorted by size desc."""
    from itertools import combinations
    rows = []
    for r in range(len(order), 0, -1):
        for combo in combinations(order, r):
            included = set(combo)
            excluded = set(order) - included
            factors = set.intersection(*[sets[m] for m in included])
            for m in excluded:
                factors = factors - sets[m]
            if factors:
                rows.append({"members": combo, "factors": factors, "size": len(factors)})
    # Sort by degree of overlap first (how many models agree), then by
    # factor count within that -- so "all 4 models agree" always comes
    # before any single-model bucket, regardless of which has more factors.
    rows.sort(key=lambda r: (-len(r["members"]), -r["size"]))
    return rows


def plot_shap_upset():
    fig = plt.figure(figsize=(13, 10.5))
    outer = fig.add_gridspec(2, 1, hspace=0.6)

    for oi, task in enumerate(["condition", "timepoint"]):
        sets = get_top_importance_sets(task)
        rows = _intersections(sets, order=MODEL_ORDER)
        n = len(rows)

        gs = outer[oi].subgridspec(2, 1, height_ratios=[2.2, 1.2], hspace=0.05)
        ax_bar = fig.add_subplot(gs[0])
        ax_dot = fig.add_subplot(gs[1], sharex=ax_bar)

        x = np.arange(n)
        heights = [r["size"] for r in rows]
        ax_bar.bar(x, heights, color=BAR_COLOR, width=0.6)
        per_line = 2
        wrapped = []
        for r in rows:
            names = sorted(r["factors"], key=lambda f: int(f.replace("DR", "")))
            lines = [", ".join(names[j:j + per_line]) for j in range(0, len(names), per_line)]
            wrapped.append("\n".join(lines))
        max_lines = max(w.count("\n") + 1 for w in wrapped)
        line_h = max(heights) * 0.32
        for i, (r, label) in enumerate(zip(rows, wrapped)):
            ax_bar.text(i, r["size"] + max(heights) * 0.05, label,
                        ha="center", va="bottom", fontsize=8)
        ax_bar.set_ylim(0, max(heights) + max_lines * line_h + max(heights) * 0.15)
        ax_bar.set_ylabel("# factors")
        ax_bar.spines[["top", "right"]].set_visible(False)
        ax_bar.set_xticks([])
        title = "Condition: ACS vs. CCS vs. non-CCS" if task == "condition" else "Timepoint: TP1 through TP4"
        ax_bar.set_title(f"{title}: top {TOP_N} active DRVI factors, by each model's own importance measure",
                          fontsize=11, fontweight="bold")

        for i, m in enumerate(MODEL_ORDER):
            ax_dot.axhline(i, color="#dddddd", linewidth=1, zorder=0)
        for i, r in enumerate(rows):
            member_ys = [MODEL_ORDER.index(m) for m in r["members"]]
            for m in MODEL_ORDER:
                y = MODEL_ORDER.index(m)
                on = m in r["members"]
                ax_dot.scatter([i], [y], s=140 if on else 60,
                                color=BAR_COLOR if on else "#e6e6e6", zorder=3)
            if len(member_ys) > 1:
                ax_dot.plot([i, i], [min(member_ys), max(member_ys)], color="#333333", linewidth=1.5, zorder=2)
        ax_dot.set_xlim(-0.6, n - 0.4)
        ax_dot.set_yticks(range(len(MODEL_ORDER)))
        ax_dot.set_yticklabels(MODEL_ORDER)
        ax_dot.set_ylim(-0.6, len(MODEL_ORDER) - 0.4)
        ax_dot.set_xticks([])
        ax_dot.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax_dot.tick_params(left=False)

    fig.suptitle("Do the four classifiers agree on which DRVI factors matter?",
                 fontsize=12, fontweight="bold")
    out_path = os.path.join(OUT_DIR, "shap_upset.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def load_top_shap_factors(task):
    per_model = {}
    for model, path in SHAP_FILES[task].items():
        s = pd.read_csv(path, index_col=0)["mean_abs_shap"]
        per_model[model] = set(s.sort_values(ascending=False).head(TOP_N).index)
    all_factors = sorted(set.union(*per_model.values()), key=lambda f: int(f.replace("DR", "")))
    consensus = pd.DataFrame({m: [f in per_model[m] for f in all_factors] for m in SHAP_MODEL_ORDER}, index=all_factors)
    consensus["count"] = consensus[SHAP_MODEL_ORDER].sum(axis=1)
    consensus = consensus.sort_values("count", ascending=False)
    return consensus


def plot_shap_consensus():
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharey=False)
    for ax, task in zip(axes, ["condition", "timepoint"]):
        consensus = load_top_shap_factors(task)
        x = np.arange(len(consensus))
        bottom = np.zeros(len(consensus))
        for m in SHAP_MODEL_ORDER:
            heights = consensus[m].astype(int).values
            ax.bar(x, heights, bottom=bottom, color=MODEL_COLORS[m], label=m, width=0.7)
            bottom += heights
        ax.set_xticks(x)
        ax.set_xticklabels(consensus.index, rotation=90, fontsize=7)
        ax.set_ylim(0, 3.5)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_ylabel("# models\n(top-10 by SHAP)")
        title = "Condition: ACS vs. CCS vs. non-CCS" if task == "condition" else "Timepoint: TP1 through TP4"
        n_any = (consensus["count"] > 0).sum()
        n_all3 = (consensus["count"] == 3).sum()
        ax.set_title(f"{title} -- {n_any} factors in ≥1 model's top {TOP_N} by mean |SHAP|, "
                     f"{n_all3} in all 3", fontsize=10, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[m], label=m) for m in SHAP_MODEL_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"How many models agree an active DRVI factor matters?\n"
                 f"(top-{TOP_N} by mean |SHAP value| -- LR, RF, XGB only; MLP has no SHAP, it uses permutation importance)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    out_path = os.path.join(OUT_DIR, "shap_consensus.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    plot_classifier_scores()
    plot_shap_upset()
