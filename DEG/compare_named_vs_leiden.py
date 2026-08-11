"""
Compare the coarse Scanorama-named DEG result against the finer raw Harmony
Leiden clusters that make it up, for Monocytes-CD14 (leiden 0, 9) and
T-cell-CD4 (leiden 1, 4, 5, 6). One combined heatmap per cell type per
comparison, "Named: {type}" and "Leiden {cluster}" side by side as columns.

Cluster -> majority Scanorama label, from the LIVE h5ad (not the stale
deg_metadata_shared.parquet, which was built 2026-07-28, a day before the
h5ad was last modified -- it undercounts clusters, 25/32 instead of the
real 14/19, and gives wrong majority labels for several of them):
    0  Monocytes - CD14   (purity 99.5%)
    1  T-cell-CD4         (purity 98.8%)
    4  T-cell-CD4         (purity 96.8%)
    5  T-cell-CD4         (purity 99.4%)
    6  T-cell-CD4         (purity 98.9%)
    9  Monocytes - CD14   (purity 77.4%)

Cluster 9 has zero rows in the ACS-vs-CCS comparisons (pooled or
per-timepoint) -- too few cells/samples there, not a mapping error. It does
have data in ACS_vs_nonCCS and ACS_TP1M_vs_TP4M, so those are included too.

Same top-5-up/top-5-down-per-column, clusters-with-hits-only style as
replot_timepoint_vs_ccs.py / sudenaz/fix_and_replot.py. Reads the
already-saved DESeq2 CSVs -- does not rerun DESeq2.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "/vol/disk/ubuntu/master_practicum_cytokines/lisa/deg_conditions"
TOP_N_PER_DIRECTION = 5
COMPARISONS = [
    "ACS_vs_CCS", "ACS_TP1M_vs_CCS", "ACS_TP2M_vs_CCS", "ACS_TP3M_vs_CCS", "ACS_TP4M_vs_CCS",
    "ACS_vs_nonCCS", "ACS_TP1M_vs_TP4M",
]

CELL_GROUPS = {
    "Monocytes": {
        "named": ["Monocytes - CD14"],
        "leiden": [0, 9],
    },
    "T-cell-CD4": {
        "named": ["T-cell-CD4"],
        "leiden": [1, 4, 5, 6],
    },
}


def load_combined(group_name, comparison):
    named_path = os.path.join(OUTPUT_DIR, f"Harmony_named_{comparison}.csv")
    leiden_path = os.path.join(OUTPUT_DIR, f"Harmony_leiden_{comparison}.csv")
    if not (os.path.exists(named_path) and os.path.exists(leiden_path)):
        return None

    named = pd.read_csv(named_path)
    leiden = pd.read_csv(leiden_path)

    spec = CELL_GROUPS[group_name]
    named = named[named["cell_type"].isin(spec["named"])].copy()
    named["cell_type"] = "Named: " + named["cell_type"]

    leiden["cell_type"] = leiden["cell_type"].astype(int)
    leiden = leiden[leiden["cell_type"].isin(spec["leiden"])].copy()
    present = sorted(leiden["cell_type"].unique())
    missing = sorted(set(spec["leiden"]) - set(present))
    if missing:
        print(f"  ({comparison}, {group_name}: cluster(s) {missing} not tested here -- omitted, not an error)")
    leiden["cell_type"] = "Leiden " + leiden["cell_type"].astype(str)

    return pd.concat([named, leiden], ignore_index=True)


def make_heatmap(df, title, out_path, top_n=TOP_N_PER_DIRECTION):
    sig = df[df["significant"]].copy()
    if sig.empty:
        print(f"  (no significant genes -- skipping heatmap for {title})")
        return
    top_up = sig.sort_values("log2FoldChange", ascending=False).groupby("cell_type").head(top_n)
    top_down = sig.sort_values("log2FoldChange", ascending=True).groupby("cell_type").head(top_n)
    top_genes = pd.concat([top_up, top_down])["gene"].unique()

    pivot = df[df["gene"].isin(top_genes)].pivot_table(
        index="gene", columns="cell_type", values="log2FoldChange", aggfunc="mean"
    )
    clusters_with_hits = sig["cell_type"].unique()
    pivot = pivot[[c for c in pivot.columns if c in clusters_with_hits]]
    if pivot.empty:
        print(f"  (empty pivot -- skipping heatmap for {title})")
        return

    fig_w = max(5, 0.7 * pivot.shape[1] + 2)
    fig_h = max(4, 0.3 * pivot.shape[0] + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(pivot, cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "log2FC"})
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Named cell type vs Leiden cluster")
    ax.set_ylabel("Gene")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


for group_name in CELL_GROUPS:
    for comparison in COMPARISONS:
        combined = load_combined(group_name, comparison)
        if combined is None:
            print(f"  (missing CSVs for {comparison} -- skipping)")
            continue
        title = f"Named vs Leiden, {group_name}: {comparison.replace('_', ' ')}"
        out_path = os.path.join(
            OUTPUT_DIR, f"heatmap_Harmony_NamedVsLeiden_{group_name.replace('-', '')}_{comparison}.png"
        )
        make_heatmap(combined, title, out_path)

print("\nDone.")
