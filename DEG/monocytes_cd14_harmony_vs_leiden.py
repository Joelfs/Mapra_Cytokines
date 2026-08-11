"""
Monocytes - CD14: Named (Harmony) vs Harmony leiden clusters 0 & 9, one
heatmap per comparison. Standalone from compare_named_vs_leiden.py so this
one comparison has its own results folder.

Cluster -> majority Scanorama label, from the LIVE h5ad (the
deg_metadata_shared.parquet-based mapping was stale -- see
compare_named_vs_leiden.py for the full explanation):
    Harmony leiden 0   Monocytes - CD14   purity 99.5%
    Harmony leiden 9   Monocytes - CD14   purity 77.4%

Cluster 9 has zero rows in ACS_vs_CCS at the STANDARD min_cells=10 threshold
(pooled or per-timepoint) -- only 1 of 14 CCS patients has >=10 cells in
that cluster, so the CCS side can never reach the min_samples=3 DESeq2
needs. monocyte_cluster9_relaxed.py reruns cluster 9 vs CCS per timepoint
with min_cells=5 (7 CCS patients qualify at that threshold), producing
Harmony_leiden_ACS_{tp}_vs_CCS_cluster9_relaxed.csv for TP1M-TP4M -- this
script pulls those in wherever they exist, labeled "Leiden 9 (relaxed
threshold)" so the lower confidence stays visible in the plot itself, not
just in this docstring. No relaxed version exists for the pooled ACS_vs_CCS
(different, patient-blocked design) -- that one stays Leiden-0-only.

Reads the already-saved DESeq2 CSVs from lisa/deg_conditions/ -- does not
rerun DESeq2 (except via the separate monocyte_cluster9_relaxed.py script).
Same top-5-up/top-5-down-per-column, clusters-with-hits-only style as
replot_timepoint_vs_ccs.py / sudenaz/fix_and_replot.py.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SOURCE_DIR = "/vol/disk/ubuntu/master_practicum_cytokines/lisa/deg_conditions"
OUTPUT_DIR = "/vol/disk/ubuntu/master_practicum_cytokines/lisa/monocytes_cd14_harmony_vs_leiden"
TOP_N_PER_DIRECTION = 5
COMPARISONS = [
    "ACS_vs_CCS", "ACS_TP1M_vs_CCS", "ACS_TP2M_vs_CCS", "ACS_TP3M_vs_CCS", "ACS_TP4M_vs_CCS",
]
NAMED_TYPES = ["Monocytes - CD14"]
LEIDEN_CLUSTERS = [0, 9]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_combined(comparison):
    named_path = os.path.join(SOURCE_DIR, f"Harmony_named_{comparison}.csv")
    leiden_path = os.path.join(SOURCE_DIR, f"Harmony_leiden_{comparison}.csv")
    if not (os.path.exists(named_path) and os.path.exists(leiden_path)):
        return None

    named = pd.read_csv(named_path)
    named = named[named["cell_type"].isin(NAMED_TYPES)].copy()
    named["cell_type"] = "Named: " + named["cell_type"]

    leiden = pd.read_csv(leiden_path)
    leiden["cell_type"] = leiden["cell_type"].astype(int)
    present = sorted(set(leiden["cell_type"].unique()) & set(LEIDEN_CLUSTERS))
    leiden = leiden[leiden["cell_type"].isin(LEIDEN_CLUSTERS)].copy()
    leiden["cell_type"] = "Leiden " + leiden["cell_type"].astype(str)

    parts = [named, leiden]

    # Cluster 9, relaxed-threshold rerun (min_cells=5) -- only for per-timepoint
    # comparisons, only where the standard-threshold fit above didn't already
    # cover cluster 9.
    if 9 not in present:
        relaxed_path = os.path.join(SOURCE_DIR, f"Harmony_leiden_{comparison}_cluster9_relaxed.csv")
        if os.path.exists(relaxed_path):
            relaxed = pd.read_csv(relaxed_path)
            relaxed["cell_type"] = "Leiden 9 (relaxed threshold)"
            parts.append(relaxed)
        else:
            print(f"  ({comparison}: cluster 9 not tested here, no relaxed rerun available either -- omitted)")

    return pd.concat(parts, ignore_index=True)


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


for comparison in COMPARISONS:
    combined = load_combined(comparison)
    if combined is None:
        print(f"  (missing CSVs for {comparison} -- skipping)")
        continue
    title = f"Named vs Leiden, Monocytes - CD14: {comparison.replace('_', ' ')}"
    out_path = os.path.join(OUTPUT_DIR, f"heatmap_Harmony_MonocytesCD14_{comparison}.png")
    make_heatmap(combined, title, out_path)

print(f"\nDone. Output in: {OUTPUT_DIR}")
