"""
Heatmap: y = raw Leiden cluster (harmony_leiden / drvi_leiden, current
resolution 1.0), x = cell_type_Scanorama label, color = % of that cluster's
cells carrying each Scanorama label.

Shows cluster purity directly (which clusters are one clean cell type vs.
mixed across several) instead of just the single majority-vote label used
earlier for cluster naming.

Backed mode -- only reads .obs columns, no need to touch the big expression
matrix, should run in seconds.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

DATA_PATH = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/data/data_for_practicum_post_integration.h5ad"
OUT_DIR = "."

CLUSTER_COLS = {
    "Harmony": "harmony_leiden",
    "DRVI": "drvi_leiden",
}
REFERENCE_COL = "cell_type_Scanorama"

print("Loading (backed, read-only)...")
adata = sc.read_h5ad(DATA_PATH, backed="r")
obs = adata.obs[[*CLUSTER_COLS.values(), REFERENCE_COL]].copy()

for grouping, cluster_col in CLUSTER_COLS.items():
    print(f"\n=== {grouping} ({cluster_col}) ===")

    def sort_key(idx):
        try:
            return sorted(idx, key=lambda x: int(x))
        except ValueError:
            return sorted(idx)

    contingency = pd.crosstab(obs[cluster_col].astype(str), obs[REFERENCE_COL].astype(str))
    contingency = contingency.loc[sort_key(contingency.index)]
    # order columns by overall abundance, most common cell type first
    contingency = contingency[contingency.sum(axis=0).sort_values(ascending=False).index]

    row_norm = contingency.div(contingency.sum(axis=1), axis=0) * 100

    fig_w = max(7, 0.55 * row_norm.shape[1] + 3)
    fig_h = max(5, 0.32 * row_norm.shape[0] + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(row_norm, annot=True, fmt=".0f", cmap="viridis", vmin=0, vmax=100, ax=ax,
                cbar_kws={"label": "% of cluster's cells"}, linewidths=0.3, linecolor="white",
                annot_kws={"fontsize": 6})
    ax.set_title(f"{grouping}: Leiden cluster composition by Scanorama cell type", fontsize=11)
    ax.set_xlabel("cell_type_Scanorama", fontsize=9)
    ax.set_ylabel(f"{cluster_col} (raw cluster)", fontsize=9)
    ax.tick_params(axis="x", labelsize=8, rotation=35)
    ax.tick_params(axis="y", labelsize=7, rotation=0)
    fig.tight_layout()
    out_path = f"{OUT_DIR}/leiden_vs_scanorama_{grouping}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

    # purity summary: majority label + purity % per cluster
    majority_label = row_norm.idxmax(axis=1)
    purity = row_norm.max(axis=1)
    summary = pd.DataFrame({
        "cluster": row_norm.index,
        "n_cells": contingency.sum(axis=1).values,
        "majority_scanorama_label": majority_label.values,
        "purity_pct": purity.round(1).values,
    })
    summary.to_csv(f"{OUT_DIR}/leiden_vs_scanorama_{grouping}_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Mean purity: {purity.mean():.1f}%   Worst cluster: {purity.min():.1f}%")

print("\nDone.")