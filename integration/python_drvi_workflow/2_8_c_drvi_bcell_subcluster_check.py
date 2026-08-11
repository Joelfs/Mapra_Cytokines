#!/usr/bin/env python3
"""
Checks whether a conspicuous mini-cluster still exists within the (already
cleaned) B cells in the curated dataset: subclustering (Leiden on the DRVI
embedding) restricted to B cells, then donor composition and top marker
genes per cluster to identify small/donor-dominated clusters.

Usage:
    conda run -n mapra_cytokines python 2_8_c_drvi_bcell_subcluster_check.py
"""

import os

import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

CURATED_H5AD = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/data/data_for_practicum_post_integration_curated.h5ad"
OUTPUT_DIR = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/visualization/drvi_interpretation"
CELLTYPE_KEY = "cell_type_curated"
RESOLUTION = 1.0

print("=== Load data ===")
adata = sc.read_h5ad(CURATED_H5AD)
b = adata[adata.obs[CELLTYPE_KEY] == "B-cell"].copy()
print(f"B cells: {b.n_obs}")

print("\n=== Subclustering (Leiden on X_drvi) ===")
sc.pp.neighbors(b, use_rep="X_drvi", n_neighbors=15)
sc.tl.umap(b, random_state=0)
sc.tl.leiden(b, resolution=RESOLUTION, key_added="b_subcluster")

cluster_sizes = b.obs["b_subcluster"].value_counts().sort_index()
print("\nCluster sizes:")
print(cluster_sizes)

# ── Donor composition per cluster ────────────────────────────────────────────

patient_id = b.obs["sample_id"].astype(str).str.split(".", n=1).str[0]
b.obs["patient_id"] = patient_id

print("\n=== Donor dominance per cluster ===")
dominance_rows = []
for cl in cluster_sizes.index:
    sub = patient_id[b.obs["b_subcluster"] == cl]
    top_donor = sub.value_counts().idxmax()
    top_frac = sub.value_counts().iloc[0] / len(sub)
    dominance_rows.append({
        "cluster": cl, "n_cells": len(sub),
        "top_donor": top_donor, "top_donor_fraction": top_frac,
    })
    print(f"  Cluster {cl}: n={len(sub)}, dominant donor={top_donor} ({100*top_frac:.1f}%)")

dominance_df = pd.DataFrame(dominance_rows).sort_values("top_donor_fraction", ascending=False)
dominance_df.to_csv(os.path.join(OUTPUT_DIR, "bcell_subcluster_donor_dominance.csv"), index=False)

# ── Marker genes for the most conspicuous (smallest / most strongly
#    donor-dominated) cluster ─────────────────────────────────────────────────

suspect_cluster = dominance_df.iloc[0]["cluster"]
print(f"\n=== Marker genes for the most conspicuous cluster ({suspect_cluster}) ===")

b.X = b.layers["log1p_norm"]
sc.tl.rank_genes_groups(b, "b_subcluster", groups=[suspect_cluster], reference="rest", method="wilcoxon")
markers = sc.get.rank_genes_groups_df(b, group=suspect_cluster).head(20)
print(markers[["names", "logfoldchanges", "pvals_adj"]].to_string(index=False))
markers.to_csv(os.path.join(OUTPUT_DIR, f"bcell_subcluster_{suspect_cluster}_markers.csv"), index=False)

# ── Plot: UMAP of the B cells, colored by subcluster and by donor ───────────

sc.settings.set_figure_params(dpi=100, facecolor="white", frameon=False)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
sc.pl.umap(b, color="b_subcluster", ax=axes[0], show=False, title="B-cell subclusters", legend_loc="on data")
top_donors = patient_id.value_counts().head(8).index.tolist()
b.obs["patient_id_top"] = patient_id.where(patient_id.isin(top_donors), "other")
sc.pl.umap(b, color="patient_id_top", ax=axes[1], show=False, title="Donor (top 8 + other)")
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "bcell_subcluster_umap.png")
plt.savefig(out, bbox_inches="tight", dpi=150)
plt.close("all")
print(f"\nSaved: {out}")
print(f"Saved: bcell_subcluster_donor_dominance.csv")
print(f"Saved: bcell_subcluster_{suspect_cluster}_markers.csv")
