#!/usr/bin/env python3
"""
Gene candidate UMAP — like 2_7_b, but for candidates from 2_7_c
(gene-based threshold selection, optionally filtered to a cell type).

Panel 1 shows the candidate cells on the full DRVI UMAP (all 109504 cells as
a gray background). Panel 2 (--add-majority-panel) zooms into the donor with
the most candidate cells: background = only that donor's own cells, so the
location of the signal within that single donor becomes visible without
being obscured by all other cells.

Usage:
    conda run -n mapra_cytokines python 2_7_d_drvi_gene_candidate_umap.py \
        --candidate-csv candidate_cells_gene_IGHG3_in_Bcell.csv \
        --title "IGHG3 (B-cell, threshold=1.37)" \
        --add-majority-panel
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
parser.add_argument("--candidate-csv", required=True,
                    help="File name (relative to --output-dir) or full path to the candidate_cells_*.csv")
parser.add_argument("--title", default="")
parser.add_argument("--out-name", default="")
parser.add_argument("--add-majority-panel", action="store_true",
                    help="Second panel: only the donor with the most candidate cells "
                         "(background = only that donor's own cells)")
args = parser.parse_args()

embed_path = os.path.join(args.output_dir, "embed.h5ad")
embed = sc.read_h5ad(embed_path)
umap = embed.obsm["X_umap"]
patient_id = embed.obs["sample_id"].astype(str).str.split(".", n=1).str[0]

candidate_path = args.candidate_csv if os.path.isabs(args.candidate_csv) else os.path.join(args.output_dir, args.candidate_csv)
candidates = set(pd.read_csv(candidate_path)["cell_barcode"])
is_candidate = embed.obs_names.isin(candidates)
n_candidates = int(is_candidate.sum())

title = args.title or os.path.basename(candidate_path)

sc.settings.set_figure_params(dpi=100, facecolor="white", frameon=False)
n_panels = 2 if args.add_majority_panel else 1
fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.5))
axes = [axes] if n_panels == 1 else list(axes)

# Panel 1: all cells as background, all candidates in red.
ax = axes[0]
ax.scatter(umap[~is_candidate, 0], umap[~is_candidate, 1],
           s=2, c="lightgray", alpha=0.5, linewidths=0, rasterized=True)
ax.scatter(umap[is_candidate, 0], umap[is_candidate, 1],
           s=5, c="crimson", alpha=0.9, linewidths=0, rasterized=True)
ax.set_title(f"{title}\n(n={n_candidates}, {100*n_candidates/embed.n_obs:.2f}% of all cells)")
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.set_xticks([])
ax.set_yticks([])

if args.add_majority_panel:
    donor_counts = patient_id[is_candidate].value_counts()
    majority_donor = donor_counts.idxmax()
    n_majority = int(donor_counts.iloc[0])

    donor_mask = (patient_id == majority_donor).values
    donor_candidate = is_candidate & donor_mask
    n_donor_cells = int(donor_mask.sum())

    ax2 = axes[1]
    ax2.scatter(umap[donor_mask & ~is_candidate, 0], umap[donor_mask & ~is_candidate, 1],
                s=4, c="lightgray", alpha=0.6, linewidths=0, rasterized=True)
    ax2.scatter(umap[donor_candidate, 0], umap[donor_candidate, 1],
                s=8, c="crimson", alpha=0.95, linewidths=0, rasterized=True)
    ax2.set_title(f"Donor {majority_donor} only (majority sample)\n"
                  f"(n={n_majority} candidates out of {n_donor_cells} cells from this donor, "
                  f"{100*n_majority/n_donor_cells:.1f}%)")
    ax2.set_xlabel("UMAP1")
    ax2.set_ylabel("UMAP2")
    ax2.set_xticks([])
    ax2.set_yticks([])
    print(f"Majority sample: {majority_donor}  ({n_majority}/{n_candidates} candidates, "
          f"{n_donor_cells} cells from this donor in total)")

plt.tight_layout()
out_name = args.out_name or f"umap_{os.path.splitext(os.path.basename(candidate_path))[0]}.png"
out = os.path.join(args.output_dir, out_name)
plt.savefig(out, bbox_inches="tight", dpi=150)
plt.close("all")
print(f"Saved: {out}  (n={n_candidates})")
