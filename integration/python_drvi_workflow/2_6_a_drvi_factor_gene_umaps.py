#!/usr/bin/env python3
"""
DRVI factor-gene UMAPs — gene expression of the top genes of selected factor
directions on the DRVI UMAP, e.g. to visually check clusters for doublet
signatures (simultaneous expression of markers from different lineages).

Uses the already existing top_genes_per_factor_*.csv (from 2_2_c/2_4_a) and
embed.h5ad (UMAP) — no model reload needed.

Usage:
    conda run -n mapra_cytokines python 2_5_b_drvi_factor_gene_umaps.py \
        --factor-dirs DR7+,DR22-,DR21+,DR35+,DR32+,DR56-,DR48+,DR39-,DR9-,DR4+,DR17+
"""

import argparse
import os
import re
import warnings

import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--drvi-input",
    default="/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/data/data_for_practicum_post_integration.h5ad",
)
parser.add_argument(
    "--output-dir",
    default="/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/visualization/drvi_interpretation",
)
parser.add_argument("--factor-dirs", default="",
                    help="Comma-separated list of factor+direction, e.g. DR7+,DR22-,DR21+")
parser.add_argument("--score-key", choices=["ood", "ind"], default="ind")
parser.add_argument("--n-genes", type=int, default=10, help="Top-N genes per factor direction")
parser.add_argument("--extra-genes", default="",
                    help="Comma-separated list of extra genes to plot independently of the factors")
parser.add_argument("--gene-prefixes", default="",
                    help="Comma-separated list of prefixes, e.g. IGH,IGL — all var_names starting "
                         "with these are plotted as well")
parser.add_argument("--out-name", default="doublet_check_selected_factors.png")
parser.add_argument("--color-map", default="YlOrBr",
                    help="Matplotlib colormap, e.g. YlOrBr (light yellow -> dark brown)")
args = parser.parse_args()

factor_dirs = [f.strip() for f in args.factor_dirs.split(",") if f.strip()]
parsed = []
for fd in factor_dirs:
    m = re.match(r"^(DR\d+)\s*([+-])$", fd)
    if not m:
        raise ValueError(f"Could not parse '{fd}' (expected e.g. 'DR7+').")
    parsed.append((m.group(1), m.group(2)))
print(f"Factor directions: {parsed}")

embed_path = os.path.join(args.output_dir, "embed.h5ad")
embed = sc.read_h5ad(embed_path)

if parsed:
    genes_path = os.path.join(args.output_dir, f"top_genes_per_factor_{args.score_key}.csv")
    genes_df = pd.read_csv(genes_path)

print("=== Load data (gene expression) ===")
adata = sc.read_h5ad(args.drvi_input)
adata.X = adata.layers["log1p_norm"]

umap_df = pd.DataFrame(embed.obsm["X_umap"], index=embed.obs_names)
adata.obsm["X_drvi_umap"] = umap_df.loc[adata.obs_names].values

sc.settings.set_figure_params(dpi=100, facecolor="white", frameon=False)

genes, labels = [], []
for factor, direction in parsed:
    sub = genes_df[(genes_df["factor"] == factor) & (genes_df["direction"] == direction)]
    if sub.empty:
        print(f"  {factor}{direction}: no top genes found — skipped.")
        continue
    top = sub.sort_values("rank").head(args.n_genes)
    found_any = False
    for gene in top["gene"]:
        if gene in adata.var_names and gene not in genes:
            genes.append(gene)
            labels.append(f"{gene} ({factor}{direction})")
            found_any = True
    if not found_any:
        print(f"  {factor}{direction}: no genes found in the dataset — skipped.")

for gene in [g.strip() for g in args.extra_genes.split(",") if g.strip()]:
    if gene not in adata.var_names:
        print(f"  {gene}: not found in the dataset — skipped.")
    elif gene in genes:
        print(f"  {gene}: already in the list — skipped.")
    else:
        genes.append(gene)
        labels.append(gene)

for prefix in [p.strip() for p in args.gene_prefixes.split(",") if p.strip()]:
    matches = sorted(g for g in adata.var_names if g.startswith(prefix))
    if not matches:
        print(f"  Prefix '{prefix}': no matching genes found.")
        continue
    n_added = 0
    for gene in matches:
        if gene not in genes:
            genes.append(gene)
            labels.append(gene)
            n_added += 1
    print(f"  Prefix '{prefix}': {len(matches)} genes found, {n_added} newly added.")

if not genes:
    raise RuntimeError("No genes found — nothing to plot.")

print(f"\n{len(genes)} genes total: {genes}")

fig = sc.pl.embedding(
    adata, "X_drvi_umap", color=genes, show=False, return_fig=True,
    ncols=4, title=labels, color_map=args.color_map, vmax="p99.5",
)
out = os.path.join(args.output_dir, args.out_name)
fig.savefig(out, bbox_inches="tight", dpi=120)
plt.close("all")
print(f"\nSaved: {out}")
