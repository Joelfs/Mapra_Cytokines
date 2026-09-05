"""
Sterile vs infection ACS, compared separately at each available timepoint
(TP1M-TP4M), not just TP1M. Supervisor's hypothesis: since TP1M is the
peri-interventional sample (taken during the initial catheterization,
before/as the patient is admitted), a hospital-acquired infection may not
have developed or become immunologically detectable yet. Later timepoints
(TP2M ~14h, TP3M ~60h post-event, TP4M ~5-8 days pre-discharge) give the
immune system more time to react to an infectious trigger, so the
sterile-vs-infection signal may be stronger there.

Standalone script -- does not touch Lisa's deg_conditions.ipynb or the
earlier acs_subgroups_deg.py outputs. Same methodology (pseudobulk sum via
decoupler + PyDESeq2 via pertpy). Both bugs from the earlier run are fixed
from the start here: gene names are pulled from PyDESeq2's "variable"
column (not the row index), and cell types are real names (majority-vote of
the existing cell_type_Scanorama annotation per raw cluster), not raw
cluster numbers.

Direction convention (not printed on the plots -- same style as the
approved deck -- but true throughout): baseline = sterile, group_to_compare
= infection. log2FoldChange > 0 ("up") = higher in infection.
log2FoldChange < 0 ("down") = higher in sterile.
"""

import os
import warnings

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import decoupler as dc
import pertpy as pt

warnings.filterwarnings("ignore")

DATA_PATH = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/data/data_for_practicum_post_integration.h5ad"
OUTPUT_DIR = "/vol/disk/ubuntu/master_practicum_cytokines/sudenaz/sterile_infection_by_timepoint"

SAMPLE_COL = "display_name"
CONDITION_COL = "classification"
REFERENCE_CELLTYPE_COL = "cell_type_Scanorama"
CLUSTER_COLS = {
    "Harmony_leiden": "harmony_leiden",
    "DRVI_leiden": "drvi_leiden",
}

STERILE = "acs_w_o_infection"
INFECTION = "acs_w_infection"
TIMEPOINTS = ["TP1M", "TP2M", "TP3M", "TP4M"]

PADJ_THRESH = 0.05
LFC_THRESH = 1.0
MIN_CELLS = 10
MIN_COUNTS = 1000
MIN_COUNT = 10
MIN_TOTAL_COUNT = 15
TOP_N_PER_DIRECTION = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)
sc.settings.set_figure_params(dpi=150, facecolor="white", frameon=False)


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

    fig_w = max(5, 1.6 * pivot.shape[1] + 2.5)
    fig_h = max(4, 0.3 * pivot.shape[0] + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(pivot, cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "log2FC"})
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Cell type")
    ax.set_ylabel("Gene")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def faceted_volcano(df, title, out_path, max_labels=8):
    clusters = sorted(df["cell_type"].unique(), key=str)
    n = len(clusters)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

    for i, cluster in enumerate(clusters):
        ax = axes[i // ncols][i % ncols]
        sub = df[df["cell_type"] == cluster]
        not_sig = sub[~sub["significant"]]
        sig_up = sub[sub["significant"] & (sub["log2FoldChange"] > 0)]
        sig_down = sub[sub["significant"] & (sub["log2FoldChange"] < 0)]

        ax.scatter(not_sig["log2FoldChange"], -np.log10(not_sig["padj"].clip(lower=1e-300)),
                   s=6, color="lightgrey", alpha=0.5)
        ax.scatter(sig_up["log2FoldChange"], -np.log10(sig_up["padj"].clip(lower=1e-300)),
                   s=16, color="firebrick", label=f"up ({len(sig_up)})")
        ax.scatter(sig_down["log2FoldChange"], -np.log10(sig_down["padj"].clip(lower=1e-300)),
                   s=16, color="steelblue", label=f"down ({len(sig_down)})")

        labeled = pd.concat([sig_up, sig_down]).sort_values("padj").head(max_labels)
        for _, row in labeled.iterrows():
            ax.annotate(row["gene"], (row["log2FoldChange"], -np.log10(max(row["padj"], 1e-300))),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")

        ax.axvline(LFC_THRESH, color="black", linestyle="--", linewidth=0.6)
        ax.axvline(-LFC_THRESH, color="black", linestyle="--", linewidth=0.6)
        ax.axhline(-np.log10(PADJ_THRESH), color="black", linestyle="--", linewidth=0.6)
        ax.set_title(f"Cluster {cluster}", fontsize=10)
        ax.set_xlabel("log2FC")
        ax.set_ylabel("-log10(padj)")
        ax.legend(fontsize=7, loc="upper left")

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


print("Loading data...")
adata_full = sc.read_h5ad(DATA_PATH)


def parse_timepoint(display_name):
    s = str(display_name)
    if "." not in s:
        return "TP0M"
    return f"TP{s.split('.')[-1]}M"


adata_full.obs["timepoint"] = adata_full.obs["display_name"].apply(parse_timepoint)

# --- build cluster -> cell type mapping once, from the full dataset ---
print("Building cluster -> cell type mapping (majority vote of cell_type_Scanorama)...")
cluster_maps = {}
mapping_rows = []
for grouping_name, cluster_col in CLUSTER_COLS.items():
    cmap = {}
    for cluster_id, group in adata_full.obs.groupby(cluster_col, observed=True):
        counts = group[REFERENCE_CELLTYPE_COL].value_counts()
        top_label = counts.index[0]
        purity = counts.iloc[0] / counts.sum() * 100
        cmap[str(cluster_id)] = top_label
        mapping_rows.append({"grouping": grouping_name, "cluster_id": cluster_id,
                              "assigned_cell_type": top_label, "purity_pct": round(purity, 1),
                              "n_cells": counts.sum()})
    cluster_maps[grouping_name] = cmap
pd.DataFrame(mapping_rows).to_csv(os.path.join(OUTPUT_DIR, "cluster_to_celltype_mapping.csv"), index=False)

all_results = []

for tp in TIMEPOINTS:
    print(f"\n########## {tp} ##########")
    adata = adata_full[
        (adata_full.obs["timepoint"] == tp)
        & (adata_full.obs[CONDITION_COL].isin([STERILE, INFECTION]))
    ].copy()
    print(f"Cells: {adata.n_obs}")
    print(adata.obs.groupby(CONDITION_COL, observed=True)[SAMPLE_COL].nunique().to_string())

    if adata.obs[SAMPLE_COL].nunique() < 4:
        print(f"  Too few samples at {tp} -- skipping.")
        continue

    for grouping_name, cluster_col in CLUSTER_COLS.items():
        print(f"\n=== {tp} / {grouping_name} ===")
        cluster_to_name = cluster_maps[grouping_name]

        adata_pb = dc.pp.pseudobulk(
            adata, sample_col=SAMPLE_COL, groups_col=cluster_col, layer="counts", mode="sum"
        )
        dc.pp.filter_samples(adata_pb, min_cells=MIN_CELLS, min_counts=MIN_COUNTS)

        comparison_results = []
        for cluster_value in adata_pb.obs[cluster_col].unique():
            sub = adata_pb[adata_pb.obs[cluster_col] == cluster_value].copy()
            n_a = (sub.obs[CONDITION_COL] == STERILE).sum()
            n_b = (sub.obs[CONDITION_COL] == INFECTION).sum()
            if n_a < 2 or n_b < 2:
                continue
            try:
                sub.obs[CONDITION_COL] = sub.obs[CONDITION_COL].astype(str)
                dc.pp.filter_by_expr(sub, group=CONDITION_COL, min_count=MIN_COUNT, min_total_count=MIN_TOTAL_COUNT)
                if sub.n_vars < 10:
                    continue
                pds2 = pt.tl.PyDESeq2(adata=sub, design=f"~ {CONDITION_COL}")
                pds2.fit()
                res_df = pds2.test_contrasts(
                    pds2.contrast(column=CONDITION_COL, baseline=STERILE, group_to_compare=INFECTION)
                )
            except Exception as e:
                print(f"  {cluster_value}: FAILED ({e})")
                continue

            if "gene" not in res_df.columns:
                res_df["gene"] = res_df["variable"] if "variable" in res_df.columns else res_df.index

            res_df = res_df.rename(columns={"log_fc": "log2FoldChange", "adj_p_value": "padj", "p_value": "pvalue"})
            res_df["significant"] = (res_df["padj"] < PADJ_THRESH) & (res_df["log2FoldChange"].abs() > LFC_THRESH)
            res_df["cell_type"] = cluster_to_name.get(str(cluster_value), str(cluster_value))
            res_df["cluster_id"] = str(cluster_value)
            res_df["timepoint"] = tp
            res_df["grouping"] = grouping_name
            res_df["n_sterile"] = n_a
            res_df["n_infection"] = n_b
            comparison_results.append(res_df)
            all_results.append(res_df)

        if not comparison_results:
            print("  No clusters had enough samples to test.")
            continue

        combined = pd.concat(comparison_results, ignore_index=True)
        n_sig = combined["significant"].sum()
        print(f"  {len(combined)} gene-cluster rows tested, {n_sig} significant "
              f"(n={combined['n_sterile'].iloc[0]} sterile vs {combined['n_infection'].iloc[0]} infection)")

        title = f"{grouping_name}: sterile vs infection ({tp})"
        make_heatmap(combined, f"Top 5 up/down per cluster, {title}",
                     os.path.join(OUTPUT_DIR, f"heatmap_{grouping_name}_sterile_vs_infection_{tp}.png"))
        faceted_volcano(combined, title,
                         os.path.join(OUTPUT_DIR, f"volcano_{grouping_name}_sterile_vs_infection_{tp}.png"))

if all_results:
    final = pd.concat(all_results, ignore_index=True)
    for (grouping, tp), grp in final.groupby(["grouping", "timepoint"]):
        out_path = os.path.join(OUTPUT_DIR, f"sterile_vs_infection_{grouping}_{tp}.csv")
        grp.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

    summary = final.groupby(["grouping", "timepoint"])["significant"].sum().reset_index()
    summary.to_csv(os.path.join(OUTPUT_DIR, "summary_sig_counts_by_timepoint.csv"), index=False)
    print("\nSignificant gene-cluster rows per timepoint:")
    print(summary.to_string(index=False))
else:
    print("No results produced -- check sample sizes and errors above.")

print(f"\nAll outputs in: {OUTPUT_DIR}")