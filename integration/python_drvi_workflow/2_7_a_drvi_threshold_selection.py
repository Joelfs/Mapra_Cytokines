#!/usr/bin/env python3
"""
DRVI threshold selection — step 1: factor activity as a candidate selector

For each specified factor (+ direction), the distribution of factor values
across all cells is shown as a histogram + KDE. For a clean cell-type factor
the distribution is often bimodal: a mass near zero (cells without the
program) and a separated peak (the target cells). The threshold is set in a
data-driven way in the valley between the two peaks (local density minimum),
not guessed.

If no clear bimodality is found, the script falls back to a percentile and
explicitly marks it as not data-driven.

Result per factor direction:
  - Histogram plot with threshold line
  - candidate_cells_<factor><direction>.csv (cell barcodes above threshold)

Usage:
    conda run -n mapra_cytokines python 2_7_a_drvi_threshold_selection.py \
        --factor-dirs DR9-,DR17+,DR4+
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
    "--output-dir",
    default="/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/visualization/drvi_interpretation",
)
parser.add_argument("--factor-dirs", required=True,
                    help="Comma-separated list of factor+direction, e.g. DR9-,DR17+,DR4+")
parser.add_argument("--grid-points", type=int, default=2000)
parser.add_argument("--fallback-percentile", type=float, default=95.0,
                    help="Percentile fallback (toward the tail), used if no bimodality is found")
parser.add_argument("--yscale", choices=["linear", "sqrt", "log"], default="sqrt",
                    help="y-axis scaling — sqrt is a middle ground between linear and log, "
                         "makes small peaks visible without stretching the axis extremely")
parser.add_argument("--celltype-key", default="cell_type_Scanorama")
parser.add_argument("--celltype-filter", default="",
                    help="Only cells of these cell types are considered at all (for ALL factors), "
                         "comma-separated list, e.g. 'Monocytes - CD14,Dendritic,Monocytes - CD16_FCGR3A'. "
                         "Empty = all cells.")
parser.add_argument("--exclude", default="",
                    help="Factor:cell-type pairs whose cells are excluded from the distribution/candidates, "
                         "comma-separated, e.g. 'DR4+:Megakaryocytes'. Cell type must match --celltype-key.")
parser.add_argument("--manual-threshold", default="",
                    help="Factor:value pairs to override the automatically detected threshold "
                         "(e.g. when more than two peaks are present and a later valley is desired), "
                         "comma-separated, e.g. 'DR4+:4.4'")
args = parser.parse_args()

embed_path = os.path.join(args.output_dir, "embed.h5ad")
embed = sc.read_h5ad(embed_path)


def _sanitize_celltype(name: str) -> str:
    return name.strip().replace(" ", "").replace("-", "").replace("_", "")


celltype_filter_list = [c.strip() for c in args.celltype_filter.split(",") if c.strip()]
if celltype_filter_list:
    keep = embed.obs[args.celltype_key].isin(celltype_filter_list).values
    print(f"Filter '{args.celltype_key}' in {celltype_filter_list}: {keep.sum()} / {embed.n_obs} cells")
    embed = embed[keep].copy()
    celltype_suffix = "_".join(_sanitize_celltype(c) for c in celltype_filter_list)
else:
    celltype_suffix = ""

exclude_map: dict[str, set[str]] = {}
for pair in [p.strip() for p in args.exclude.split(",") if p.strip()]:
    fd_key, celltype = pair.split(":")
    exclude_map.setdefault(fd_key.strip(), set()).add(celltype.strip())

manual_threshold_map: dict[str, float] = {}
for pair in [p.strip() for p in args.manual_threshold.split(",") if p.strip()]:
    fd_key, value = pair.split(":")
    manual_threshold_map[fd_key.strip()] = float(value)

factor_dirs = []
for fd in args.factor_dirs.split(","):
    fd = fd.strip()
    direction = fd[-1]
    factor = fd[:-1]
    assert direction in ("+", "-"), f"Could not read direction from '{fd}' (expected e.g. 'DR9-')."
    factor_dirs.append((factor, direction))

n = len(factor_dirs)
fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5))
if n == 1:
    axes = [axes]

summary_rows = []

for ax, (factor, direction) in zip(axes, factor_dirs):
    fd_key = f"{factor}{direction}"
    excluded_types = exclude_map.get(fd_key, set())
    if excluded_types:
        keep_mask = ~embed.obs[args.celltype_key].isin(excluded_types).values
        n_excluded = int((~keep_mask).sum())
        print(f"{fd_key}: excluding {n_excluded} cells ({', '.join(sorted(excluded_types))})")
    else:
        keep_mask = np.ones(embed.n_obs, dtype=bool)

    obs_names_subset = embed.obs_names[keep_mask]
    values = np.asarray(embed[keep_mask, factor].X).flatten()
    grid = np.linspace(values.min(), values.max(), args.grid_points)
    kde = gaussian_kde(values)
    density = kde(grid)

    maxima_idx = argrelextrema(density, np.greater)[0]
    minima_idx = argrelextrema(density, np.less)[0]

    # Main ("bulk") peak: the overall highest density.
    bulk_idx = maxima_idx[np.argmax(density[maxima_idx])]
    bulk_x = grid[bulk_idx]

    # Target peak: highest density among the peaks that lie BEYOND the bulk
    # peak in the requested direction.
    if direction == "+":
        candidate_peaks = maxima_idx[grid[maxima_idx] > bulk_x]
    else:
        candidate_peaks = maxima_idx[grid[maxima_idx] < bulk_x]

    threshold = None
    is_data_driven = False
    is_manual = fd_key in manual_threshold_map

    if is_manual:
        threshold = manual_threshold_map[fd_key]
        is_data_driven = True
        print(f"{fd_key}: manual threshold={threshold} (overrides auto-detection)")
    elif len(candidate_peaks) > 0:
        target_idx = candidate_peaks[np.argmax(density[candidate_peaks])]
        target_x = grid[target_idx]
        # Valley between bulk_x and target_x: local minimum with the smallest density
        lo, hi = sorted([bulk_idx, target_idx])
        valley_candidates = minima_idx[(minima_idx > lo) & (minima_idx < hi)]
        if len(valley_candidates) > 0:
            valley_idx = valley_candidates[np.argmin(density[valley_candidates])]
            threshold = grid[valley_idx]
            is_data_driven = True

    if threshold is None:
        # Fallback: percentile in the requested direction, clearly marked
        # as not data-driven.
        pct = args.fallback_percentile if direction == "+" else (100 - args.fallback_percentile)
        threshold = np.percentile(values, pct)

    if direction == "+":
        candidate_mask = values > threshold
    else:
        candidate_mask = values < threshold
    n_candidates = int(candidate_mask.sum())

    candidates = pd.Series(obs_names_subset[candidate_mask], name="cell_barcode")
    ct_suffix = f"_in_{celltype_suffix}" if celltype_suffix else ""
    out_csv = os.path.join(args.output_dir, f"candidate_cells_{factor}{direction.replace('+','pos').replace('-','neg')}{ct_suffix}.csv")
    candidates.to_csv(out_csv, index=False)

    if is_manual:
        method = "manually set"
    elif is_data_driven:
        method = "data-driven (valley between peaks)"
    else:
        method = f"fallback: {args.fallback_percentile:.0f}th percentile"
    pct_of_total = 100 * n_candidates / embed.n_obs
    print(f"{factor}{direction}: threshold={threshold:.3f}  n_candidates={n_candidates} "
          f"({pct_of_total:.2f}% of all {embed.n_obs} cells)  [{method}]")
    print(f"  Saved: {os.path.basename(out_csv)}")

    summary_rows.append({
        "factor": factor, "direction": direction, "threshold": threshold,
        "n_candidates": n_candidates, "pct_candidates": pct_of_total,
        "n_excluded": len(embed) - len(values), "data_driven": is_data_driven,
    })

    ax.hist(values, bins=100, density=True, alpha=0.4, color="steelblue")
    ax.plot(grid, density, color="black", lw=1.2)
    ax.axvline(threshold, color="crimson", ls="--", lw=1.5,
               label=f"Threshold={threshold:.2f}\n(n={n_candidates}, {pct_of_total:.1f}%)")
    ax.axvline(bulk_x, color="gray", ls=":", lw=1, alpha=0.7)
    title = f"{factor}{direction}"
    if excluded_types:
        title += f"  [excluding {', '.join(sorted(excluded_types))}]"
    if is_manual:
        title += "  [manual threshold]"
    elif not is_data_driven:
        title += "  [fallback percentile]"
    ax.set_title(title)
    ax.set_xlabel("Factor value")
    ax.set_ylabel("Density" + (f" ({args.yscale})" if args.yscale != "linear" else ""))
    if args.yscale == "sqrt":
        ax.set_yscale("function", functions=(np.sqrt, np.square))
    else:
        ax.set_yscale(args.yscale)
    ax.legend(fontsize=8, loc="upper right" if direction == "+" else "upper left")

plt.tight_layout()
fig_suffix = f"_in_{celltype_suffix}" if celltype_suffix else ""
out_fig = os.path.join(args.output_dir, f"threshold_selection_histograms{fig_suffix}.png")
plt.savefig(out_fig, bbox_inches="tight", dpi=120)
plt.close("all")
print(f"\nSaved: {out_fig}")

pd.DataFrame(summary_rows).to_csv(
    os.path.join(args.output_dir, f"threshold_selection_summary{fig_suffix}.csv"), index=False
)
print(f"Saved: threshold_selection_summary{fig_suffix}.csv")
