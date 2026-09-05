"""
ACS subgroup DEG (sterile vs infection vs subacute, TP1M only) grouped
DIRECTLY by cell_type_Scanorama -- no Leiden clustering involved at all.
This is the other half of the "decouple clustering from labeling" check:
what does the reference annotation say on its own, independent of how
Harmony/DRVI happen to cluster cells.

Same methodology (pseudobulk sum via decoupler + PyDESeq2), same direction
convention as the rest of the pipeline: for "X_vs_Y", baseline=X,
group_to_compare=Y, positive log2FC = higher in Y.
"""

import os
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import decoupler as dc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import issparse

warnings.filterwarnings("ignore")

DATA_PATH = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/data/data_for_practicum_post_integration.h5ad"
OUTPUT_DIR = "/vol/disk/ubuntu/master_practicum_cytokines/sudenaz/acs_subgroups_deg_scanorama_only"

SAMPLE_COL = "display_name"
CONDITION_COL = "classification"
CELL_TYPE_COL = "cell_type_Scanorama"

STERILE = "acs_w_o_infection"
INFECTION = "acs_w_infection"
SUBACUTE = "acs_subacute"

PADJ_THRESH = 0.05
LFC_THRESH = 1.0
TOP_N_PER_DIRECTION = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading full data (not backed -- needed for pseudobulk over raw counts)...")
adata_full = sc.read_h5ad(DATA_PATH)
print(f"Cells: {adata_full.n_obs:,}   Genes: {adata_full.n_vars:,}")


def parse_timepoint(display_name):
    s = str(display_name)
    if "." not in s:
        return "TP0M"
    return f"TP{s.split('.')[-1]}M"


adata_full.obs["timepoint"] = adata_full.obs["display_name"].apply(parse_timepoint)

adata = adata_full[
    (adata_full.obs["timepoint"] == "TP1M") &
    (adata_full.obs[CONDITION_COL].isin([STERILE, INFECTION, SUBACUTE]))
].copy()
print(f"TP1M ACS subgroup cells: {adata.n_obs:,}")
print(adata.obs.groupby(CONDITION_COL, observed=True)[SAMPLE_COL].nunique().to_string())

print(f"\n{CELL_TYPE_COL} categories: {adata.obs[CELL_TYPE_COL].nunique()}")

adata_pb = dc.pp.pseudobulk(adata, sample_col=SAMPLE_COL, groups_col=CELL_TYPE_COL, layer="counts", mode="sum")
dc.pp.filter_samples(adata_pb, min_cells=10, min_counts=1000)
print(f"Pseudobulk samples: {adata_pb.n_obs}")

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats


def run_deseq2_per_celltype(pdata, condition_col, baseline, group_to_compare, cell_type_col, min_samples=2, n_cpus=4):
    """
    baseline = first-named group in "X_vs_Y" (X), group_to_compare = second-named (Y).
    Positive log2FC = higher in group_to_compare (Y) -- matches the pertpy-based
    convention used everywhere else in this pipeline (deg_acs_subgroups.py,
    sterile_infection_by_timepoint.py).
    """
    results = {}
    for ct in sorted(pdata.obs[cell_type_col].unique()):
        mask = (pdata.obs[cell_type_col] == ct) & (pdata.obs[condition_col].isin([baseline, group_to_compare]))
        pdata_ct = pdata[mask].copy()
        n_a = (pdata_ct.obs[condition_col] == baseline).sum()
        n_b = (pdata_ct.obs[condition_col] == group_to_compare).sum()
        if n_a < min_samples or n_b < min_samples:
            print(f"  SKIP {ct}: {baseline}={n_a}, {group_to_compare}={n_b} (need >={min_samples})")
            continue
        print(f"  {ct}: {baseline}={n_a}, {group_to_compare}={n_b} samples")

        genes_to_keep = dc.pp.filter_by_expr(pdata_ct, group=condition_col, min_count=10,
                                              min_total_count=15, inplace=False)
        pdata_ct = pdata_ct[:, genes_to_keep].copy()
        if pdata_ct.n_vars < 10:
            print(f"    SKIP {ct}: too few genes after filtering")
            continue

        X = pdata_ct.X if not issparse(pdata_ct.X) else pdata_ct.X.toarray()
        counts_df = pd.DataFrame(X.astype(int), index=pdata_ct.obs_names, columns=pdata_ct.var_names)
        meta_df = pdata_ct.obs[[condition_col]].copy()
        # baseline = reference level; contrast test_level=group_to_compare, ref_level=baseline
        # -> positive log2FC = higher in group_to_compare (Y)
        meta_df[condition_col] = pd.Categorical(meta_df[condition_col], categories=[baseline, group_to_compare])

        try:
            dds = DeseqDataSet(counts=counts_df, metadata=meta_df, design_factors=condition_col, n_cpus=n_cpus)
            dds.deseq2()
            stat_res = DeseqStats(dds, contrast=[condition_col, group_to_compare, baseline], n_cpus=n_cpus)
            stat_res.summary()
            res = stat_res.results_df.copy()
            res["cell_type"] = ct
            results[ct] = res
        except Exception as e:
            print(f"    ERROR in {ct}: {e}")
    return results


def summarise(results_dict, label):
    if not results_dict:
        print(f"No results for {label}")
        return None
    all_res = pd.concat(results_dict.values())
    all_res["gene"] = all_res.index
    all_res["significant"] = (all_res["padj"] < PADJ_THRESH) & (all_res["log2FoldChange"].abs() > LFC_THRESH)
    sig = all_res[all_res["significant"]]
    all_res.to_csv(os.path.join(OUTPUT_DIR, f"{label}.csv"), index=False)
    print(f"{label}: {len(sig)} significant DEGs")
    return all_res


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
        return
    fig_w = max(4.5, 1.1 * pivot.shape[1] + 2.2)
    fig_h = max(4, 0.3 * pivot.shape[0] + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(pivot, cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "log2FC"})
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("cell_type_Scanorama", fontsize=9)
    ax.set_ylabel("Gene", fontsize=9)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
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
        ax.set_title(f"{cluster}", fontsize=10)
        ax.set_xlabel("log2FC")
        ax.set_ylabel("-log10(padj)")
        ax.legend(fontsize=7, loc="upper left")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


COMPARISONS = [
    ("sterile_vs_infection", STERILE, INFECTION),
    ("sterile_vs_subacute", STERILE, SUBACUTE),
    ("infection_vs_subacute", INFECTION, SUBACUTE),
]

for label, group_a, group_b in COMPARISONS:
    print(f"\n{'='*70}\nScanorama-only: {label} (baseline={group_a}, compare={group_b})\n{'='*70}")
    results = run_deseq2_per_celltype(adata_pb, CONDITION_COL, group_a, group_b, CELL_TYPE_COL)
    res_df = summarise(results, f"ACS_subgroups_ScanoramaOnly_{label}")
    if res_df is not None:
        title = f"Scanorama labels only: {label.replace('_', ' ')}"
        make_heatmap(res_df, f"Top 5 up/down per cell type, {title}",
                     os.path.join(OUTPUT_DIR, f"heatmap_ACS_subgroups_ScanoramaOnly_{label}.png"))
        faceted_volcano(res_df, title,
                         os.path.join(OUTPUT_DIR, f"volcano_ACS_subgroups_ScanoramaOnly_{label}.png"))

print(f"\nAll outputs in: {OUTPUT_DIR}")