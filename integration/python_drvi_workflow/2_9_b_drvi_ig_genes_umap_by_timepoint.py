#!/usr/bin/env python3
"""
UMAP grid: selected IGH*/IGL* genes x timepoints.

For each gene (row) and each timepoint (column), expression is shown on the
DRVI UMAP: background = all cells (gray), foreground = only cells from that
timepoint, colored by expression. The color scale is consistent per gene
across all timepoints (same vmin/vmax per row), so the panels within a row
are directly comparable.

Timepoint derived from sample_id (see 2_9_a): "m6.4" -> TP4,
"k9" (no dot) -> control.

Usage:
    conda run -n mapra_cytokines python 2_9_b_drvi_ig_genes_umap_by_timepoint.py \
        --genes IGLC2,IGLC3,IGHM,IGHG1,IGHG2,IGHG3,IGHD,IGHA1
"""

import argparse
import os

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    "--drvi-input",
    default="/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/data/data_for_practicum_post_integration.h5ad",
)
parser.add_argument(
    "--output-dir",
    default="/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/visualization/drvi_interpretation",
)
parser.add_argument("--genes", required=True, help="Comma-separated list of genes")
parser.add_argument("--timepoints", default="control,TP1,TP2,TP3,TP4",
                    help="Comma-separated list of timepoint columns to display")
parser.add_argument("--color-map", default="YlOrBr")
parser.add_argument("--vmax-percentile", type=float, default=99.5)
parser.add_argument("--out-name", default="ig_genes_umap_by_timepoint.png")
args = parser.parse_args()

genes = [g.strip() for g in args.genes.split(",") if g.strip()]

print("=== Load data ===")
adata = sc.read_h5ad(args.drvi_input)
adata.X = adata.layers["log1p_norm"]

embed_path = os.path.join(args.output_dir, "embed.h5ad")
embed = sc.read_h5ad(embed_path)
umap_df = pd.DataFrame(embed.obsm["X_umap"], index=embed.obs_names)
umap = umap_df.loc[adata.obs_names].values

missing = [g for g in genes if g not in adata.var_names]
if missing:
    print(f"WARNING: not found in the dataset, skipped: {missing}")
genes = [g for g in genes if g in adata.var_names]
if not genes:
    raise RuntimeError("None of the specified genes were found in the dataset.")

# ── Derive timepoint from sample_id ───────────────────────────────────────────

sample_id = adata.obs["sample_id"].astype(str)
split = sample_id.str.split(".", n=1, expand=True)
timepoint = np.where(split[1].notna(), "TP" + split[1], "control")
timepoints = [t.strip() for t in args.timepoints.split(",") if t.strip()]
timepoint = pd.Categorical(timepoint, categories=["control", "TP1", "TP2", "TP3", "TP4"], ordered=True)

n_cells_by_tp = pd.Series(timepoint).value_counts()
print("Cells per timepoint:")
print(n_cells_by_tp.reindex(timepoints))

# ── Plot grid ──────────────────────────────────────────────────────────────────

n_genes = len(genes)
n_tp = len(timepoints)
fig, axes = plt.subplots(n_genes, n_tp, figsize=(3.2 * n_tp, 3.0 * n_genes))
if n_genes == 1:
    axes = axes[np.newaxis, :]

for i, gene in enumerate(genes):
    expr = adata[:, gene].X
    expr = np.asarray(expr.todense()).flatten() if hasattr(expr, "todense") else np.asarray(expr).flatten()
    vmax = max(np.percentile(expr, args.vmax_percentile), 1e-6)

    for j, tp in enumerate(timepoints):
        ax = axes[i, j]
        mask = (timepoint == tp)

        ax.scatter(umap[:, 0], umap[:, 1], s=1, c="lightgray", alpha=0.4, linewidths=0, rasterized=True)
        sc_plot = ax.scatter(umap[mask, 0], umap[mask, 1], s=2, c=expr[mask],
                              cmap=args.color_map, vmin=0, vmax=vmax, linewidths=0, rasterized=True)
        if i == 0:
            ax.set_title(f"{tp}\n(n={int(mask.sum())})", fontsize=9)
        if j == 0:
            ax.set_ylabel(gene, fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(sc_plot, ax=axes[i, -1], fraction=0.046, pad=0.04)

plt.tight_layout()
out = os.path.join(args.output_dir, args.out_name)
plt.savefig(out, bbox_inches="tight", dpi=130)
plt.close("all")
print(f"\nSaved: {out}")
