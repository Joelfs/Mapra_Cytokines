"""
Cluster correspondence between resolution 1.0 (current harmony_leiden /
drvi_leiden) and reconstructed resolution 2.0 (harmony_leiden_res2 /
drvi_leiden_res2), from resolution2_leiden_labels.csv.

For each grouping (Harmony, DRVI):
  - contingency table: how many cells of each res1.0 cluster fall into each
    res2.0 cluster
  - row-normalized heatmap (% of each res1.0 cluster's cells per res2.0
    cluster) -- shows merges/splits at a glance
  - overall normalized + adjusted mutual information score (scalar summary)
  - best-match mapping: for each res1.0 cluster, which res2.0 cluster it
    corresponds to most -- saved to CSV, this defines the "matched cluster
    pairs" for the downstream gene-conservation comparison

Fast: only reads the small labels CSV, no h5ad / scanpy needed.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score

IN_PATH = "resolution2_leiden_labels.csv"
OUT_DIR = "."

GROUPINGS = {
    "Harmony": ("harmony_leiden", "harmony_leiden_res2"),
    "DRVI": ("drvi_leiden", "drvi_leiden_res2"),
}

labels = pd.read_csv(IN_PATH, index_col=0)
print(f"Loaded {len(labels)} cells")

for name, (res1_col, res2_col) in GROUPINGS.items():
    print(f"\n=== {name} ===")
    res1 = labels[res1_col].astype(str)
    res2 = labels[res2_col].astype(str)

    nmi = normalized_mutual_info_score(res1, res2)
    ami = adjusted_mutual_info_score(res1, res2)
    print(f"NMI: {nmi:.3f}   AMI: {ami:.3f}")

    contingency = pd.crosstab(res1, res2)
    # sort rows/cols by cluster id numerically where possible
    def sort_key(idx):
        try:
            return sorted(idx, key=lambda x: int(x))
        except ValueError:
            return sorted(idx)
    contingency = contingency.loc[sort_key(contingency.index), sort_key(contingency.columns)]

    row_norm = contingency.div(contingency.sum(axis=1), axis=0) * 100

    fig_w = max(8, 0.35 * contingency.shape[1] + 3)
    fig_h = max(6, 0.3 * contingency.shape[0] + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(row_norm, cmap="viridis", vmin=0, vmax=100, ax=ax,
                cbar_kws={"label": "% of res 1.0 cluster's cells"},
                linewidths=0.2, linecolor="white")
    ax.set_title(f"{name}: resolution 1.0 vs 2.0 cluster correspondence\n"
                 f"(NMI={nmi:.3f}, AMI={ami:.3f})", fontsize=11)
    ax.set_xlabel("Resolution 2.0 cluster (reconstructed)", fontsize=9)
    ax.set_ylabel("Resolution 1.0 cluster (current)", fontsize=9)
    ax.tick_params(axis="x", labelsize=6, rotation=90)
    ax.tick_params(axis="y", labelsize=6, rotation=0)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/correspondence_heatmap_{name}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved correspondence_heatmap_{name}.png")

    # best-match mapping: for each res1.0 cluster, the res2.0 cluster it overlaps most with
    best_match = row_norm.idxmax(axis=1)
    best_pct = row_norm.max(axis=1)
    mapping = pd.DataFrame({
        "res1_cluster": best_match.index,
        "res1_n_cells": contingency.sum(axis=1).values,
        "best_matching_res2_cluster": best_match.values,
        "overlap_pct": best_pct.round(1).values,
    })
    mapping.to_csv(f"{OUT_DIR}/cluster_correspondence_{name}.csv", index=False)
    print(f"Saved cluster_correspondence_{name}.csv")
    print(mapping.to_string(index=False))

print("\nDone.")