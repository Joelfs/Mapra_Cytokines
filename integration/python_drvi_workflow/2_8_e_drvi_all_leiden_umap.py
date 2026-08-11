#!/usr/bin/env python3
"""
UMAP of the Leiden clusters (drvi_leiden, res=1.0) for ALL cells in the
curated dataset (after removal of the doublets/artifacts).

Usage:
    conda run -n mapra_cytokines python 2_8_e_drvi_all_leiden_umap.py
"""

import os

import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

CURATED_H5AD = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/data/data_for_practicum_post_integration_curated.h5ad"
OUTPUT_DIR = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/visualization/drvi_interpretation"
EMBED_H5AD = os.path.join(OUTPUT_DIR, "embed.h5ad")
LEIDEN_KEY = "drvi_leiden"

print("=== Load data ===")
adata = sc.read_h5ad(CURATED_H5AD)
print(f"{adata.n_obs} cells")

embed = sc.read_h5ad(EMBED_H5AD)
umap_df = pd.DataFrame(embed.obsm["X_umap"], index=embed.obs_names)
adata.obsm["X_umap"] = umap_df.loc[adata.obs_names].values

sizes = adata.obs[LEIDEN_KEY].value_counts().sort_index()
adata.obs[f"{LEIDEN_KEY}_label"] = adata.obs[LEIDEN_KEY].map(lambda c: f"{c} (n={sizes[c]})")

sc.settings.set_figure_params(dpi=100, facecolor="white", frameon=False)
fig, ax = plt.subplots(figsize=(8, 7))
sc.pl.umap(adata, color=f"{LEIDEN_KEY}_label", ax=ax, show=False,
           title=f"{LEIDEN_KEY} (res=1.0, n={adata.n_obs} cells)",
           legend_loc="right margin", legend_fontsize=8)

out = os.path.join(OUTPUT_DIR, "umap_all_leiden_curated.png")
plt.savefig(out, bbox_inches="tight", dpi=150)
plt.close("all")
print(f"Saved: {out}")
