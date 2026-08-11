#!/usr/bin/env python3
"""
Compare IGH*/IGL* genes across timepoints.

`sample_id` encodes patient.timepoint (e.g. "m6.4" -> patient m6, TP4;
controls like "k9" have no timepoint, just a single draw -> "control").
For each IGH*/IGL* gene, the mean log1p-normalized expression per timepoint
is computed and shown as a gene x timepoint heatmap.

Optionally restrict to a cell type with --celltype-filter (e.g. 'B-cell' or
'Plasma Blast'), since Ig genes are barely expressed outside of B cells/
plasma cells and the signal would otherwise be diluted.

Usage:
    conda run -n mapra_cytokines python 2_9_a_drvi_ig_genes_by_timepoint.py \
        --gene-prefixes IGH,IGL --celltype-filter B-cell
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
parser.add_argument("--gene-prefixes", default="IGH,IGL")
parser.add_argument("--celltype-key", default="cell_type_Scanorama")
parser.add_argument("--celltype-filter", default="",
                    help="Only cells of this cell type, e.g. 'B-cell'. Empty = all cells.")
parser.add_argument("--out-name", default="")
args = parser.parse_args()

print("=== Load data ===")
adata = sc.read_h5ad(args.drvi_input)
adata.X = adata.layers["log1p_norm"]

if args.celltype_filter:
    keep = (adata.obs[args.celltype_key] == args.celltype_filter).values
    print(f"Filter '{args.celltype_key}' == '{args.celltype_filter}': {keep.sum()} / {adata.n_obs} cells")
    adata = adata[keep].copy()
else:
    print(f"No cell-type filter — all {adata.n_obs} cells.")

# ── Derive timepoint from sample_id ───────────────────────────────────────────
# "m6.4" -> "TP4" ; "k9" (no dot) -> "control"

sample_id = adata.obs["sample_id"].astype(str)
split = sample_id.str.split(".", n=1, expand=True)
timepoint = np.where(split[1].notna(), "TP" + split[1], "control")
adata.obs["timepoint"] = pd.Categorical(
    timepoint, categories=["control", "TP1", "TP2", "TP3", "TP4"], ordered=True
)
print("\nCells per timepoint:")
print(adata.obs["timepoint"].value_counts().sort_index())

# ── Collect IGH*/IGL* genes ───────────────────────────────────────────────────

prefixes = [p.strip() for p in args.gene_prefixes.split(",") if p.strip()]
genes = sorted(g for g in adata.var_names if any(g.startswith(p) for p in prefixes))
print(f"\n{len(genes)} genes found for prefixes {prefixes}: {genes}")

# ── Mean expression per gene x timepoint ──────────────────────────────────────

expr = adata[:, genes].X
expr = expr.toarray() if hasattr(expr, "toarray") else np.asarray(expr)
expr_df = pd.DataFrame(expr, columns=genes, index=adata.obs_names)
expr_df["timepoint"] = adata.obs["timepoint"].values

mean_by_tp = expr_df.groupby("timepoint", observed=True)[genes].mean().T  # genes x timepoints
n_cells_by_tp = adata.obs["timepoint"].value_counts()

ct_suffix = f"_{args.celltype_filter.replace(' ', '_').replace('-', '')}" if args.celltype_filter else ""
out_csv = os.path.join(args.output_dir, f"ig_genes_by_timepoint{ct_suffix}.csv")
mean_by_tp.to_csv(out_csv)
print(f"\nSaved: {os.path.basename(out_csv)}")

# ── Heatmap ───────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, max(6, 0.22 * len(genes))))
im = ax.imshow(mean_by_tp.values, aspect="auto", cmap="YlOrBr")
ax.set_xticks(range(mean_by_tp.shape[1]))
ax.set_xticklabels([f"{c}\n(n={n_cells_by_tp.get(c, 0)})" for c in mean_by_tp.columns], fontsize=8)
ax.set_yticks(range(len(genes)))
ax.set_yticklabels(genes, fontsize=6)
ax.set_xlabel("Timepoint")
ax.set_ylabel("Gene")
title = f"IGH*/IGL* mean expression by timepoint"
if args.celltype_filter:
    title += f"  [{args.celltype_filter}]"
ax.set_title(title)
cbar = fig.colorbar(im, ax=ax, shrink=0.6)
cbar.set_label("Mean log1p-normalized expression")

plt.tight_layout()
out_name = args.out_name or f"ig_genes_by_timepoint{ct_suffix}.png"
out_fig = os.path.join(args.output_dir, out_name)
plt.savefig(out_fig, bbox_inches="tight", dpi=150)
plt.close("all")
print(f"Saved: {out_fig}")
