"""
Violin plots for the genes that came up as cluster-exclusive in the
Monocytes-CD14 Leiden-0-vs-Leiden-9 DEG comparison (TP2M vs CCS, cluster 9
run with no per-patient cell-count threshold):

  Cluster-9-only (inflammatory monocyte activation): PYCARD, IL1B, CXCL8,
    CYBB, MNDA, FPR1, S100A8, S100A9
  Cluster-0-only (likely T-cell contamination, not monocyte biology):
    IL7R, TCF7

log1p_norm expression, split by cluster (0 vs 9) and condition
(ACS_sterile vs CCS) -- shows whether the ACS-vs-CCS shift found by DESeq2
is visible at the single-cell level and genuinely confined to one cluster.

Loads only the needed genes via backed mode -- not the full expression
matrix.
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/data/data_for_practicum_post_integration.h5ad"
OUTPUT_DIR = "/vol/disk/ubuntu/master_practicum_cytokines/lisa/monocytes_cd14_harmony_vs_leiden"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GENES_CLUSTER9 = ["PYCARD", "IL1B", "CXCL8", "CYBB", "MNDA", "FPR1", "S100A8", "S100A9"]
GENES_CLUSTER0 = ["IL7R", "TCF7"]
GENES = GENES_CLUSTER9 + GENES_CLUSTER0

print("Loading (backed)...")
adata_backed = sc.read_h5ad(DATA_PATH, backed="r")
adata_backed.obs["harmony_leiden_int"] = adata_backed.obs["harmony_leiden"].astype(int)

adata_backed.obs["timepoint"] = adata_backed.obs["display_name"].astype(str).apply(
    lambda s: f"TP{s.split('.')[-1]}M" if "." in s else "TP0M"
)
condition_map = {
    "acs_w_o_infection": "ACS_sterile", "acs_w_infection": "ACS_infection",
    "acs_subacute": "ACS_subacute", "ccs": "CCS",
    "vollstaendiger_ausschluss": "non-CCS", "koronarsklerose": "Sclerosis",
}
adata_backed.obs["condition"] = adata_backed.obs["classification"].map(condition_map)

mask = (
    (adata_backed.obs["harmony_leiden_int"].isin([0, 9]))
    & (
        ((adata_backed.obs["condition"] == "ACS_sterile") & (adata_backed.obs["timepoint"] == "TP2M"))
        | (adata_backed.obs["condition"] == "CCS")
    )
)
sub = adata_backed[mask, GENES].to_memory()
print(f"{sub.n_obs:,} cells x {sub.n_vars} genes loaded")

expr = sub.layers["log1p_norm"]
if not isinstance(expr, np.ndarray):
    expr = expr.toarray()
expr_df = pd.DataFrame(expr, columns=sub.var_names, index=sub.obs_names)
expr_df["cluster"] = sub.obs["harmony_leiden_int"].map({0: "Leiden 0", 9: "Leiden 9"}).values
expr_df["condition"] = sub.obs["condition"].values

n = len(GENES)
ncols = 5
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

for i, gene in enumerate(GENES):
    ax = axes[i // ncols][i % ncols]
    plot_df = expr_df[[gene, "cluster", "condition"]].rename(columns={gene: "expr"})
    sns.violinplot(data=plot_df, x="cluster", y="expr", hue="condition",
                    split=True, inner="quartile", ax=ax, cut=0,
                    palette={"ACS_sterile": "#d62728", "CCS": "#1f77b4"})
    ax.set_title(gene, fontsize=11, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("log1p_norm expr" if i % ncols == 0 else "")
    ax.legend(fontsize=7, loc="upper right")

for j in range(n, nrows * ncols):
    axes[j // ncols][j % ncols].axis("off")

fig.suptitle("Monocytes - CD14: cluster-exclusive DEGs, TP2M (ACS vs CCS), Leiden 0 vs Leiden 9", fontsize=13)
fig.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "violin_MonocytesCD14_cluster0vs9_ACS_TP2M_vs_CCS.png")
fig.savefig(out_path, bbox_inches="tight", dpi=150)
print(f"Saved: {out_path}")
