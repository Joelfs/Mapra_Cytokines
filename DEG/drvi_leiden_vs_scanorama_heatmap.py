"""
DRVI leiden (resolution 1.0, 19 clusters) vs Scanorama cell type: composition
heatmap. Each row (cluster) sums to 1 across columns (cell types) -- shows,
per cluster, what fraction of its cells carry each Scanorama label. This is
the same majority-vote crosstab used throughout this project's cluster-to-
celltype mapping, just as a heatmap instead of a text table.

From the LIVE h5ad (not the stale deg_metadata_shared.parquet).
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/data/data_for_practicum_post_integration.h5ad"
OUTPUT_DIR = "/vol/disk/ubuntu/master_practicum_cytokines/lisa/monocytes_cd14_harmony_vs_leiden"


def read_cat(f, name):
    g = f["obs"][name]
    cats = np.array([c.decode() if isinstance(c, bytes) else c for c in g["categories"][()]])
    codes = g["codes"][()]
    return pd.Categorical.from_codes(codes, categories=cats)


f = h5py.File(DATA_PATH, "r")
drvi_leiden = read_cat(f, "drvi_leiden").astype(int)
scan = read_cat(f, "cell_type_Scanorama")

df = pd.DataFrame({"drvi_leiden": drvi_leiden, "cell_type_Scanorama": scan})
ct = pd.crosstab(df["drvi_leiden"], df["cell_type_Scanorama"])
prop = ct.div(ct.sum(axis=1), axis=0)
prop.index = prop.index.astype(int)
prop = prop.sort_index()

# order columns by overall prevalence, for a cleaner read
col_order = prop.sum(axis=0).sort_values(ascending=False).index
prop = prop[col_order]

n_cells = ct.sum(axis=1).sort_index()

fig, ax = plt.subplots(figsize=(max(8, 0.7 * prop.shape[1] + 3), max(6, 0.35 * prop.shape[0] + 2)))
sns.heatmap(prop, cmap="Blues", vmin=0, vmax=1, annot=True, fmt=".2f", annot_kws={"fontsize": 7},
            linewidths=0.4, ax=ax, cbar_kws={"label": "Fraction of cluster's cells"})
ax.set_title("DRVI leiden (resolution 1.0, 19 clusters) vs Scanorama cell type", fontsize=12)
ax.set_xlabel("Scanorama cell type")
ax.set_ylabel("drvi_leiden cluster")
ax.set_yticklabels([f"{i} (n={n_cells[i]:,})" for i in prop.index], rotation=0, fontsize=8)
plt.xticks(rotation=35, ha="right")
fig.tight_layout()
out_path = f"{OUTPUT_DIR}/heatmap_drvi_leiden_vs_Scanorama_composition.png"
fig.savefig(out_path, bbox_inches="tight", dpi=150)
print(f"Saved: {out_path}")
