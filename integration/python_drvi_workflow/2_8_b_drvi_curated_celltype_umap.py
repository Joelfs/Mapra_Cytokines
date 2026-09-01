#!/usr/bin/env python3
"""
UMAP of the curated cell-type annotation (`cell_type_curated`) from
data_for_practicum_post_integration_curated.h5ad (2_8_a).

Uses the already existing DRVI UMAP from embed.h5ad (reduced to the
remaining cells) — no recomputation needed since only 2% of the cells
were removed.

Usage:
    conda run -n mapra_cytokines python 2_8_b_drvi_curated_celltype_umap.py
"""

import os

import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

CURATED_H5AD = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/data/data_for_practicum_post_integration_curated.h5ad"
OUTPUT_DIR = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/visualization/drvi_interpretation"
EMBED_H5AD = os.path.join(OUTPUT_DIR, "embed.h5ad")

print("=== Load data ===")
adata = sc.read_h5ad(CURATED_H5AD)
print(f"{adata.n_obs} cells")

embed = sc.read_h5ad(EMBED_H5AD)
umap_df = pd.DataFrame(embed.obsm["X_umap"], index=embed.obs_names)
adata.obsm["X_umap"] = umap_df.loc[adata.obs_names].values

sc.settings.set_figure_params(dpi=100, facecolor="white", frameon=False)
fig, ax = plt.subplots(figsize=(8, 7))
sc.pl.umap(adata, color="cell_type_curated", ax=ax, show=False,
           title=f"cell_type_curated (n={adata.n_obs} cells)", legend_loc="right margin",
           legend_fontsize=8, size=4)

out = os.path.join(OUTPUT_DIR, "umap_cell_type_curated.png")
plt.savefig(out, bbox_inches="tight", dpi=150)
plt.close("all")
print(f"Saved: {out}")
