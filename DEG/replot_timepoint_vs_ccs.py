"""
Heatmap + faceted volcano plots for the ACS-per-timepoint-vs-CCS comparison
(Comparison 4 in deg_conditions.ipynb), all four cell-type groupings --
Scanorama-named (Harmony_named, DRVI_named) and raw Leiden clusters
(Harmony_leiden, DRVI_leiden; both clustered at resolution 1.0, per the
h5ad's own uns params -- same clustering used everywhere else in the
notebook). The leiden groupings were added back to allow direct comparison
against the named-type results.

Same plot style as sudenaz/fix_and_replot.py (top 5 up + top 5 down per
cluster by log2FoldChange, clusters-with-hits only, faceted per-cluster
volcano grid with gene labels), requested to match that look. Reads the
already-saved DESeq2 CSVs -- does not rerun DESeq2.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "/vol/disk/ubuntu/master_practicum_cytokines/lisa/deg_conditions"
PADJ_THRESH = 0.05
LFC_THRESH = 1.0
TOP_N_PER_DIRECTION = 5
MAX_LABELS_PER_PANEL = 8
GROUPINGS = ["Harmony_named", "DRVI_named", "Harmony_leiden", "DRVI_leiden"]
TIMEPOINTS = ["TP1M", "TP2M", "TP3M", "TP4M"]


def faceted_volcano(df, title, out_path):
    clusters = sorted(df["cell_type"].unique(), key=str)
    n = len(clusters)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

    for i, cluster in enumerate(clusters):
        ax = axes[i // ncols][i % ncols]
        sub = df[df["cell_type"] == cluster]
        not_sig = sub[~sub["significant"]]
        sig_up = sub[sub["significant"] & (sub["log2FoldChange"] > 0)]
        sig_down = sub[sub["significant"] & (sub["log2FoldChange"] < 0)]

        ax.scatter(not_sig["log2FoldChange"], -np.log10(not_sig["padj"].clip(lower=1e-300)),
                   s=6, color="lightgrey", alpha=0.5)
        ax.scatter(sig_up["log2FoldChange"], -np.log10(sig_up["padj"].clip(lower=1e-300)),
                   s=16, color="firebrick", label=f"up ({len(sig_up)})")
        ax.scatter(sig_down["log2FoldChange"], -np.log10(sig_down["padj"].clip(lower=1e-300)),
                   s=16, color="steelblue", label=f"down ({len(sig_down)})")

        labeled = pd.concat([sig_up, sig_down]).sort_values("padj").head(MAX_LABELS_PER_PANEL)
        for _, row in labeled.iterrows():
            ax.annotate(row["gene"], (row["log2FoldChange"], -np.log10(max(row["padj"], 1e-300))),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")

        ax.axvline(LFC_THRESH, color="black", linestyle="--", linewidth=0.6)
        ax.axvline(-LFC_THRESH, color="black", linestyle="--", linewidth=0.6)
        ax.axhline(-np.log10(PADJ_THRESH), color="black", linestyle="--", linewidth=0.6)
        ax.set_title(f"Cluster {cluster}", fontsize=10)
        ax.set_xlabel("log2FC")
        ax.set_ylabel("-log10(padj)")
        ax.legend(fontsize=7, loc="upper left")

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def make_heatmap(df, title, out_path, top_n=TOP_N_PER_DIRECTION):
    sig = df[df["significant"]].copy()
    if sig.empty:
        print(f"  (no significant genes -- skipping heatmap for {title})")
        return
    top_up = sig.sort_values("log2FoldChange", ascending=False).groupby("cell_type").head(top_n)
    top_down = sig.sort_values("log2FoldChange", ascending=True).groupby("cell_type").head(top_n)
    top_genes = pd.concat([top_up, top_down])["gene"].unique()

    pivot = df[df["gene"].isin(top_genes)].pivot_table(
        index="gene", columns="cell_type", values="log2FoldChange", aggfunc="mean"
    )
    clusters_with_hits = sig["cell_type"].unique()
    pivot = pivot[[c for c in pivot.columns if c in clusters_with_hits]]
    if pivot.empty:
        print(f"  (empty pivot -- skipping heatmap for {title})")
        return

    fig_w = max(5, 0.6 * pivot.shape[1] + 2)
    fig_h = max(4, 0.3 * pivot.shape[0] + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(pivot, cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "log2FC"})
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Cell Type")
    ax.set_ylabel("Gene")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


for grouping in GROUPINGS:
    for tp in TIMEPOINTS:
        comparison = f"ACS_{tp}_vs_CCS"
        csv_path = os.path.join(OUTPUT_DIR, f"{grouping}_{comparison}.csv")
        if not os.path.exists(csv_path):
            print(f"SKIP (missing): {csv_path}")
            continue
        df = pd.read_csv(csv_path)

        faceted_volcano(
            df, f"{grouping}: ACS {tp} vs CCS, per cluster",
            os.path.join(OUTPUT_DIR, f"volcano_{grouping}_{comparison}.png"),
        )
        make_heatmap(
            df, f"Top {TOP_N_PER_DIRECTION} up/down per cluster, {grouping}: ACS {tp} vs CCS",
            os.path.join(OUTPUT_DIR, f"heatmap_{grouping}_{comparison}.png"),
        )
        print(f"Replotted: {grouping} {comparison}")

print(f"\nDone. Plots in: {OUTPUT_DIR}")
