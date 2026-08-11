#!/usr/bin/env python3
"""
DRVI candidate UMAP — shows the candidate cells selected by threshold
(2_7_a) on the DRVI UMAP, to visually check whether they form a compact,
distinct cluster (or overlap with an existing cluster).

Uses embed.h5ad (UMAP) and the candidate_cells_<factor><dir>.csv from 2_7_a.

Usage:
    conda run -n mapra_cytokines python 2_7_b_drvi_candidate_umap.py \
        --factor-dirs DR9-,DR17+,DR4+
"""

import argparse
import os

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output-dir",
    default="/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/visualization/drvi_interpretation",
)
parser.add_argument("--factor-dirs", required=True,
                    help="Comma-separated list of factor+direction, e.g. DR9-,DR17+,DR4+")
parser.add_argument("--out-name", default="candidate_umap.png")
parser.add_argument("--celltype-key", default="cell_type_Scanorama")
parser.add_argument("--celltype-filter", default="",
                    help="Only cells of this cell type are plotted at all (background + candidates), "
                         "e.g. 'B-cell'. Should match --celltype-filter from 2_7_a. Empty = all cells.")
parser.add_argument("--exclude", default="",
                    help="Factor:cell-type pairs whose cells are excluded entirely from the plot "
                         "(background too), comma-separated, e.g. 'DR4+:Megakaryocytes'. "
                         "Should match --exclude from 2_7_a.")
args = parser.parse_args()

embed_path = os.path.join(args.output_dir, "embed.h5ad")
embed = sc.read_h5ad(embed_path)

if args.celltype_filter:
    keep = (embed.obs[args.celltype_key] == args.celltype_filter).values
    print(f"Filter '{args.celltype_key}' == '{args.celltype_filter}': {keep.sum()} / {embed.n_obs} cells")
    embed = embed[keep].copy()

umap = embed.obsm["X_umap"]
ct_suffix = f"_in_{args.celltype_filter.replace(' ', '_').replace('-', '')}" if args.celltype_filter else ""

exclude_map: dict[str, set[str]] = {}
for pair in [p.strip() for p in args.exclude.split(",") if p.strip()]:
    fd_key, celltype = pair.split(":")
    exclude_map.setdefault(fd_key.strip(), set()).add(celltype.strip())

factor_dirs = [fd.strip() for fd in args.factor_dirs.split(",") if fd.strip()]

sc.settings.set_figure_params(dpi=100, facecolor="white", frameon=False)
n = len(factor_dirs)
fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
if n == 1:
    axes = [axes]

for ax, fd in zip(axes, factor_dirs):
    direction = fd[-1]
    factor = fd[:-1]
    safe = factor + direction.replace("+", "pos").replace("-", "neg")
    candidate_csv = os.path.join(args.output_dir, f"candidate_cells_{safe}{ct_suffix}.csv")
    if not os.path.exists(candidate_csv):
        print(f"  {fd}: {candidate_csv} not found — run 2_7_a first. Skipped.")
        continue

    candidates = set(pd.read_csv(candidate_csv)["cell_barcode"])
    is_candidate = embed.obs_names.isin(candidates)
    n_candidates = int(is_candidate.sum())

    excluded_types = exclude_map.get(fd, set())
    if excluded_types:
        keep_mask = ~embed.obs[args.celltype_key].isin(excluded_types).values
        print(f"  {fd}: excluding {int((~keep_mask).sum())} cells ({', '.join(sorted(excluded_types))})")
    else:
        keep_mask = np.ones(embed.n_obs, dtype=bool)

    plot_umap = umap[keep_mask]
    plot_candidate = is_candidate[keep_mask]

    ax.scatter(plot_umap[~plot_candidate, 0], plot_umap[~plot_candidate, 1],
               s=2, c="lightgray", alpha=0.5, linewidths=0, rasterized=True)
    ax.scatter(plot_umap[plot_candidate, 0], plot_umap[plot_candidate, 1],
               s=4, c="crimson", alpha=0.9, linewidths=0, rasterized=True)
    title = f"{fd}  (n={n_candidates}, {100*n_candidates/embed.n_obs:.2f}%)"
    if excluded_types:
        title += f"\n[excluding {', '.join(sorted(excluded_types))}]"
    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
out_name = args.out_name
if args.celltype_filter and args.out_name == "candidate_umap.png":
    out_name = f"candidate_umap{ct_suffix}.png"
out = os.path.join(args.output_dir, out_name)
plt.savefig(out, bbox_inches="tight", dpi=150)
plt.close("all")
print(f"Saved: {out}")
