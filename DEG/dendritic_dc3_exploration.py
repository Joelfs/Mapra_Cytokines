"""
Follow-up on the DRVI TP3M finding (Dendritic cells: S100A8, S100A9, CDKN1A
higher in infection) -- Lisa's note flags this as a possible DC3
subpopulation appearing (DC3 = monocyte-like dendritic cell subset, known to
carry S100A8/A9/monocyte-associated markers).

Subsets to just the Dendritic cells (cell_type_Scanorama == "Dendritic") and
recomputes a FRESH PCA/neighbors/UMAP on just that subset -- the global
embedding is dominated by broad cell-type differences and would likely wash
out a subtle sub-population within one cell type. Small subset (~1-1.3k
cells), should run in seconds.

Outputs:
  - umap_dendritic_by_timepoint.png
  - umap_dendritic_by_gene_<GENE>.png  (one per marker gene)
  - violin_dendritic_<GENE>_by_timepoint.png  (one per marker gene, split by
    sterile vs infection within each timepoint)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/data/data_for_practicum_post_integration.h5ad"
OUT_DIR = "."

CELL_TYPE_COL = "cell_type_Scanorama"
CONDITION_COL = "classification"
MARKER_GENES = ["S100A8", "S100A9", "CDKN1A"]

condition_map = {
    "acs_w_o_infection": "ACS_sterile",
    "acs_w_infection": "ACS_infection",
    "acs_subacute": "ACS_subacute",
    "ccs": "CCS",
    "vollstaendiger_ausschluss": "non-CCS",
    "koronarsklerose": "Sclerosis",
}


def parse_timepoint(display_name):
    s = str(display_name)
    if "." not in s:
        return "TP0M"
    return f"TP{s.split('.')[-1]}M"


print("Loading full data (need expression values for feature/violin plots)...")
adata_full = sc.read_h5ad(DATA_PATH)
print(f"Cells: {adata_full.n_obs:,}   Genes: {adata_full.n_vars:,}")

adata_full.obs["timepoint"] = adata_full.obs["display_name"].apply(parse_timepoint)
adata_full.obs["condition"] = adata_full.obs[CONDITION_COL].map(condition_map)

dc = adata_full[adata_full.obs[CELL_TYPE_COL] == "Dendritic"].copy()
print(f"\nDendritic cells: {dc.n_obs:,}")
print(dc.obs["condition"].value_counts(dropna=False).to_string())
print(dc.obs["timepoint"].value_counts(dropna=False).to_string())

missing = [g for g in MARKER_GENES if g not in dc.var_names]
if missing:
    print(f"\nWARNING: genes not found in var_names, skipping: {missing}")
present_genes = [g for g in MARKER_GENES if g in dc.var_names]

# Check whether .X looks already-normalized (log1p'd) or raw counts
x_max = dc.X.max()
print(f"\n.X max value: {x_max:.1f} ({'looks log-normalized' if x_max < 50 else 'looks like raw counts -- normalizing'})")
if x_max >= 50:
    dc.X = dc.layers["counts"].copy()
    sc.pp.normalize_total(dc, target_sum=1e4)
    sc.pp.log1p(dc)

# ---------- fresh sub-embedding on just the Dendritic cells ----------
print("\nRecomputing HVGs / PCA / neighbors / UMAP on the Dendritic subset...")
dc_hvg = dc.copy()
sc.pp.highly_variable_genes(dc_hvg, n_top_genes=2000, flavor="seurat")
dc_hvg = dc_hvg[:, dc_hvg.var["highly_variable"]].copy()
sc.pp.scale(dc_hvg, max_value=10)
sc.tl.pca(dc_hvg, n_comps=min(30, dc_hvg.n_obs - 1))
sc.pp.neighbors(dc_hvg, n_neighbors=15)
sc.tl.umap(dc_hvg)

dc.obsm["X_umap_dc"] = dc_hvg.obsm["X_umap"]
coords = dc.obsm["X_umap_dc"]

# ---------- UMAP colored by timepoint ----------
fig, ax = plt.subplots(figsize=(6, 5.5))
tps = sorted(dc.obs["timepoint"].dropna().unique())
colors = plt.cm.viridis(np.linspace(0, 1, len(tps)))
for tp, col in zip(tps, colors):
    mask = (dc.obs["timepoint"] == tp).values
    ax.scatter(coords[mask, 0], coords[mask, 1], s=14, alpha=0.8, color=col, label=tp)
ax.set_title("Dendritic cells only -- colored by timepoint", fontsize=11)
ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.legend(title="timepoint", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/umap_dendritic_by_timepoint.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("Saved umap_dendritic_by_timepoint.png")

# ---------- UMAP colored by each marker gene ----------
for gene in present_genes:
    expr = dc[:, gene].X
    expr = np.asarray(expr.todense()).flatten() if hasattr(expr, "todense") else np.asarray(expr).flatten()
    fig, ax = plt.subplots(figsize=(6, 5.5))
    sc_plot = ax.scatter(coords[:, 0], coords[:, 1], s=14, alpha=0.85, c=expr, cmap="viridis")
    ax.set_title(f"Dendritic cells only -- {gene} expression", fontsize=11)
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(sc_plot, ax=ax, shrink=0.8)
    cbar.set_label("log1p normalized expression", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/umap_dendritic_by_gene_{gene}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved umap_dendritic_by_gene_{gene}.png")

# ---------- violin plots: grouped by timepoint, split sterile vs infection ----------
dc_acs = dc[dc.obs["condition"].isin(["ACS_sterile", "ACS_infection"])].copy()
tp_order = ["TP1M", "TP2M", "TP3M", "TP4M"]

for gene in present_genes:
    expr = dc_acs[:, gene].X
    expr = np.asarray(expr.todense()).flatten() if hasattr(expr, "todense") else np.asarray(expr).flatten()
    plot_df = pd.DataFrame({
        "expression": expr,
        "timepoint": pd.Categorical(dc_acs.obs["timepoint"].values, categories=tp_order, ordered=True),
        "condition": dc_acs.obs["condition"].values,
    }).dropna(subset=["timepoint"])

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.violinplot(data=plot_df, x="timepoint", y="expression", hue="condition",
                    split=True, inner="quartile", ax=ax, order=tp_order,
                    hue_order=["ACS_sterile", "ACS_infection"], cut=0)
    ax.set_title(f"Dendritic cells: {gene} expression by timepoint (sterile vs. infection)", fontsize=11)
    ax.set_xlabel("Timepoint", fontsize=10)
    ax.set_ylabel(f"{gene} (log1p normalized)", fontsize=10)
    ax.legend(title="", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/violin_dendritic_{gene}_by_timepoint.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved violin_dendritic_{gene}_by_timepoint.png")

    n_per_group = plot_df.groupby(["timepoint", "condition"], observed=True).size()
    print(n_per_group.to_string())

print("\nDone.")