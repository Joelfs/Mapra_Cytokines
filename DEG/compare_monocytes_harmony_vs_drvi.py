"""
Monocytes - CD14: one combined heatmap per comparison with both integration
methods side by side -- Harmony leiden clusters 0 & 9, DRVI leiden clusters
2 & 7, and both methods' Named: Monocytes - CD14 columns.

Cluster -> majority Scanorama label, from the LIVE h5ad (see
compare_named_vs_leiden.py for why the deg_metadata_shared.parquet-based
mapping was wrong):
    Harmony leiden 0   Monocytes - CD14   purity 99.5%
    Harmony leiden 9   Monocytes - CD14   purity 77.4%
    DRVI leiden 2      Monocytes - CD14   purity 99.7%
    DRVI leiden 7      Monocytes - CD14   purity 99.4%

Harmony cluster 9 has zero rows in ACS_vs_CCS (pooled or per-timepoint) --
too few cells/samples there. It does have data in ACS_vs_nonCCS and
ACS_TP1M_vs_TP4M. DRVI clusters 2 and 7 have data in every comparison.

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
    "ACS_vs_CCS", "ACS_vs_nonCCS", "ACS_TP1M_vs_TP4M",
    "ACS_TP1M_vs_CCS", "ACS_TP2M_vs_CCS", "ACS_TP3M_vs_CCS", "ACS_TP4M_vs_CCS",
]

SOURCES = [
    ("Harmony_named", "cell_type", ["Monocytes - CD14"], "Named (Harmony): "),
    ("DRVI_named", "cell_type", ["Monocytes - CD14"], "Named (DRVI): "),
    ("Harmony_leiden", "cell_type_int", [0, 9], "Leiden (Harmony) "),
    ("DRVI_leiden", "cell_type_int", [2, 7], "Leiden (DRVI) "),
]


def load_combined(comparison):
    parts = []
    for grouping, _, values, prefix in SOURCES:
        path = os.path.join(OUTPUT_DIR, f"{grouping}_{comparison}.csv")
        if not os.path.exists(path):
            print(f"  (missing {path} -- skipping this source)")
            continue
        df = pd.read_csv(path)
        if grouping.endswith("_leiden"):
            df["cell_type"] = df["cell_type"].astype(int)
            present = sorted(set(df["cell_type"].unique()) & set(values))
            missing = sorted(set(values) - set(present))
            if missing:
                print(f"  ({comparison}, {grouping}: cluster(s) {missing} not tested here -- omitted)")
            df = df[df["cell_type"].isin(values)].copy()
            df["cell_type"] = prefix + df["cell_type"].astype(str)
        else:
            df = df[df["cell_type"].isin(values)].copy()
            df["cell_type"] = prefix + df["cell_type"]
        parts.append(df)
    if not parts:
        return None
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

    fig_w = max(6, 0.9 * pivot.shape[1] + 2)
    fig_h = max(4, 0.3 * pivot.shape[0] + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(pivot, cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "log2FC"})
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Named cell type vs Leiden cluster (Harmony / DRVI)")
    ax.set_ylabel("Gene")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


for comparison in COMPARISONS:
    combined = load_combined(comparison)
    if combined is None:
        print(f"  (no data at all for {comparison} -- skipping)")
        continue
    title = f"Monocytes - CD14, Harmony vs DRVI: {comparison.replace('_', ' ')}"
    out_path = os.path.join(OUTPUT_DIR, f"heatmap_MonocytesCD14_HarmonyVsDRVI_{comparison}.png")
    make_heatmap(combined, title, out_path)

print("\nDone.")
