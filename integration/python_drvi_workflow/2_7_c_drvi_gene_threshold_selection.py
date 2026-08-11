#!/usr/bin/env python3
"""
Gene threshold selection — like 2_7_a, but on gene expression instead of DRVI
factors, and optionally filtered to a cell type (e.g. only B cells).

For each specified gene, the distribution of log1p-normalized expression
within the filtered cells is plotted as a histogram + KDE. As in 2_7_a, the
threshold is set in a data-driven way in the valley between the bulk peak
(no/low expression) and the target peak (high expression), with a fallback
to a percentile or manual override.

Result per gene:
  - Histogram plot with threshold line
  - candidate_cells_gene_<gene>_in_<celltype>.csv (cell barcodes above threshold)

Usage:
    conda run -n mapra_cytokines python 2_7_c_drvi_gene_threshold_selection.py \
        --genes IGHG3 --celltype-filter "B-cell"
"""

import argparse
import os

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema

parser = argparse.ArgumentParser()
parser.add_argument(
    "--drvi-input",
    default="/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/data/data_for_practicum_post_integration.h5ad",
)
parser.add_argument(
    "--output-dir",
    default="/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/visualization/drvi_interpretation",
)
parser.add_argument("--genes", required=True, help="Comma-separated list of genes, e.g. IGHG3")
parser.add_argument("--celltype-key", default="cell_type_Scanorama")
parser.add_argument("--celltype-filter", default="",
                    help="Only cells of this cell type are considered, e.g. 'B-cell'. Empty = all cells.")
parser.add_argument("--grid-points", type=int, default=2000)
parser.add_argument("--fallback-percentile", type=float, default=95.0)
parser.add_argument("--manual-threshold", default="",
                    help="Gene:value pairs to override the automatic threshold, e.g. 'IGHG3:2.0'")
parser.add_argument("--yscale", choices=["linear", "sqrt", "log"], default="sqrt")
args = parser.parse_args()

manual_threshold_map = {}
for pair in [p.strip() for p in args.manual_threshold.split(",") if p.strip()]:
    gene, value = pair.split(":")
    manual_threshold_map[gene.strip()] = float(value)

print("=== Load data ===")
adata = sc.read_h5ad(args.drvi_input)
adata.X = adata.layers["log1p_norm"]

if args.celltype_filter:
    keep_mask = (adata.obs[args.celltype_key] == args.celltype_filter).values
    print(f"Filter '{args.celltype_key}' == '{args.celltype_filter}': {keep_mask.sum()} / {adata.n_obs} cells")
    adata = adata[keep_mask].copy()
else:
    print(f"No cell-type filter — all {adata.n_obs} cells.")

genes = [g.strip() for g in args.genes.split(",") if g.strip()]
missing = [g for g in genes if g not in adata.var_names]
if missing:
    print(f"WARNING: not found in the dataset, skipped: {missing}")
genes = [g for g in genes if g in adata.var_names]
if not genes:
    raise RuntimeError("None of the specified genes were found in the dataset.")

n = len(genes)
fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5))
if n == 1:
    axes = [axes]

summary_rows = []

for ax, gene in zip(axes, genes):
    values = np.asarray(adata[:, gene].X.todense()).flatten() if hasattr(adata[:, gene].X, "todense") \
        else np.asarray(adata[:, gene].X).flatten()
    grid = np.linspace(values.min(), values.max(), args.grid_points)
    kde = gaussian_kde(values)
    density = kde(grid)

    maxima_idx = argrelextrema(density, np.greater)[0]
    minima_idx = argrelextrema(density, np.less)[0]
    bulk_idx = maxima_idx[np.argmax(density[maxima_idx])]
    bulk_x = grid[bulk_idx]

    # Gene expression: the target peak always lies to the right of the bulk peak (high expression).
    candidate_peaks = maxima_idx[grid[maxima_idx] > bulk_x]

    threshold = None
    is_data_driven = False
    is_manual = gene in manual_threshold_map

    if is_manual:
        threshold = manual_threshold_map[gene]
        is_data_driven = True
        print(f"{gene}: manual threshold={threshold} (overrides auto-detection)")
    elif len(candidate_peaks) > 0:
        target_idx = candidate_peaks[np.argmax(density[candidate_peaks])]
        lo, hi = sorted([bulk_idx, target_idx])
        valley_candidates = minima_idx[(minima_idx > lo) & (minima_idx < hi)]
        if len(valley_candidates) > 0:
            valley_idx = valley_candidates[np.argmin(density[valley_candidates])]
            threshold = grid[valley_idx]
            is_data_driven = True

    if threshold is None:
        threshold = np.percentile(values, args.fallback_percentile)

    candidate_mask = values > threshold
    n_candidates = int(candidate_mask.sum())

    candidates = pd.Series(adata.obs_names[candidate_mask], name="cell_barcode")
    ct_safe = args.celltype_filter.replace(" ", "_").replace("-", "") if args.celltype_filter else "all"
    out_csv = os.path.join(args.output_dir, f"candidate_cells_gene_{gene}_in_{ct_safe}.csv")
    candidates.to_csv(out_csv, index=False)

    # Donor breakdown: sample_id encodes patient.timepoint (e.g. "m6.4"),
    # patient_id = the part before the dot (same convention as in the
    # pseudobulk aggregation, see 2_2_b_drvi_pseudobulk.py).
    patient_id_all = adata.obs["sample_id"].astype(str).str.split(".", n=1).str[0]
    donor_counts = (
        patient_id_all[candidate_mask]
        .value_counts()
        .rename_axis("patient_id")
        .reset_index(name="n_candidate_cells")
    )
    donor_out_csv = os.path.join(args.output_dir, f"candidate_cells_gene_{gene}_in_{ct_safe}_by_donor.csv")
    donor_counts.to_csv(donor_out_csv, index=False)
    print(f"  Donors with candidate cells: {len(donor_counts)}  (saved: {os.path.basename(donor_out_csv)})")
    print(donor_counts.head(10).to_string(index=False))

    if is_manual:
        method = "manually set"
    elif is_data_driven:
        method = "data-driven (valley between peaks)"
    else:
        method = f"fallback: {args.fallback_percentile:.0f}th percentile"

    pct = 100 * n_candidates / adata.n_obs
    print(f"{gene}: threshold={threshold:.3f}  n_candidates={n_candidates} "
          f"({pct:.2f}% of {adata.n_obs} cells{' [' + args.celltype_filter + ']' if args.celltype_filter else ''})  [{method}]")
    print(f"  Saved: {os.path.basename(out_csv)}")

    summary_rows.append({
        "gene": gene, "celltype_filter": args.celltype_filter or "all",
        "threshold": threshold, "n_candidates": n_candidates, "pct_candidates": pct,
        "data_driven": is_data_driven,
    })

    ax.hist(values, bins=100, density=True, alpha=0.4, color="seagreen")
    ax.plot(grid, density, color="black", lw=1.2)
    ax.axvline(threshold, color="crimson", ls="--", lw=1.5,
               label=f"Threshold={threshold:.2f}\n(n={n_candidates}, {pct:.1f}%)")
    ax.axvline(bulk_x, color="gray", ls=":", lw=1, alpha=0.7)
    title = gene
    if args.celltype_filter:
        title += f"  [{args.celltype_filter}]"
    if is_manual:
        title += "  [manual threshold]"
    elif not is_data_driven:
        title += "  [fallback percentile]"
    ax.set_title(title)
    ax.set_xlabel("log1p-normalized expression")
    ax.set_ylabel("Density" + (f" ({args.yscale})" if args.yscale != "linear" else ""))
    if args.yscale == "sqrt":
        ax.set_yscale("function", functions=(np.sqrt, np.square))
    else:
        ax.set_yscale(args.yscale)
    ax.legend(fontsize=8, loc="upper right")

plt.tight_layout()
out_fig = os.path.join(args.output_dir, f"gene_threshold_selection_{ct_safe}.png")
plt.savefig(out_fig, bbox_inches="tight", dpi=120)
plt.close("all")
print(f"\nSaved: {out_fig}")

pd.DataFrame(summary_rows).to_csv(
    os.path.join(args.output_dir, f"gene_threshold_selection_summary_{ct_safe}.csv"), index=False
)
print(f"Saved: gene_threshold_selection_summary_{ct_safe}.csv")
