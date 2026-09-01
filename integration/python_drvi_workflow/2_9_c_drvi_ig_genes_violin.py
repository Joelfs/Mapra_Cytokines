#!/usr/bin/env python3
"""
Violin plot: IGH*/IGL* genes (facets) x cell type (x-axis), colored by
timepoint (hue).

Timepoint derived from sample_id as in 2_9_a/2_9_b ("m6.4" -> TP4,
"k9" -> control).

Usage:
    conda run -n mapra_cytokines python 2_9_c_drvi_ig_genes_violin.py \
        --genes IGLC2,IGLC3,IGHM,IGHG1,IGHG2,IGHG3,IGHD,IGHA1 \
        --timepoints TP1,TP2,TP3,TP4
"""

import argparse
import os

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
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
parser.add_argument("--celltype-key", default="cell_type_Scanorama")
parser.add_argument("--timepoints", default="TP1,TP2,TP3,TP4",
                    help="Comma-separated list of timepoints to display")
parser.add_argument("--col-wrap", type=int, default=2)
parser.add_argument("--out-name", default="ig_genes_violin_by_celltype_timepoint.png")
args = parser.parse_args()

genes = [g.strip() for g in args.genes.split(",") if g.strip()]
timepoints = [t.strip() for t in args.timepoints.split(",") if t.strip()]

print("=== Load data ===")
adata = sc.read_h5ad(args.drvi_input)
adata.X = adata.layers["log1p_norm"]

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
adata.obs["timepoint"] = timepoint

keep = pd.Series(timepoint, index=adata.obs_names).isin(timepoints).values
print(f"Filtering to timepoints {timepoints}: {keep.sum()} / {adata.n_obs} cells")
adata = adata[keep].copy()

# ── Long-format DataFrame for seaborn ─────────────────────────────────────────

expr = adata[:, genes].X
expr = expr.toarray() if hasattr(expr, "toarray") else np.asarray(expr)
expr_df = pd.DataFrame(expr, columns=genes, index=adata.obs_names)
expr_df["cell_type"] = adata.obs[args.celltype_key].values
expr_df["timepoint"] = pd.Categorical(adata.obs["timepoint"].values, categories=timepoints, ordered=True)

long_df = expr_df.melt(id_vars=["cell_type", "timepoint"], value_vars=genes,
                        var_name="gene", value_name="expression")
long_df["gene"] = pd.Categorical(long_df["gene"], categories=genes, ordered=True)

# ── Plot ──────────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid")
g = sns.catplot(
    data=long_df, kind="violin",
    x="cell_type", y="expression", hue="timepoint",
    col="gene", col_wrap=args.col_wrap,
    height=4, aspect=1.6, sharey=False,
    cut=0, inner=None, linewidth=0.5, density_norm="width",
)
for ax in g.axes.flat:
    ax.tick_params(axis="x", rotation=90)
    ax.set_xlabel("")
g.set_titles("{col_name}")
g.set_ylabels("log1p-normalized expression")

out = os.path.join(args.output_dir, args.out_name)
g.savefig(out, bbox_inches="tight", dpi=130)
plt.close("all")
print(f"\nSaved: {out}")
