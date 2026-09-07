"""
DEG comparison using the paper's OWN Leiden clusters (B2_Scanorama_Singlet_
rb_mt_cluster, equivalently the numeric prefix of cluster_cell_type_
Scanorama) -- confirmed by paper_cluster_check.log to structurally and
numerically match Pekayvaz et al. 2024's published clustering (19 clusters,
exact type composition, CD4 in 0/2/5 + Treg in 11, excluded low-count
clusters landing exactly on IDs 14-18). Supersedes both our own DRVI-based
CD4 subclustering AND the DRVI_named/Harmony_named pooled groupings for this
specific comparison -- no reclustering here, existing labels only.

STEP 1: cluster cell/sample counts, clusters 14-18 excluded (paper's own
exclusion criterion: <10 cells per patient-timepoint for most samples).

STEP 2: pseudobulk DESeq2 (sum aggregation, same setup as deg_conditions.
ipynb), ACS per-timepoint (TP1M-TP4M) vs CCS, run separately per timepoint,
across ALL retained clusters (0-13) in one pass. Unfiltered CSVs.
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import decoupler as dc
from scipy.sparse import issparse
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

DATA_PATH = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/data/data_for_practicum_post_integration.h5ad"
METADATA_PATH = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/DEG/deg_metadata_shared.parquet"
OUTPUT_DIR = "/vol/disk/ubuntu/master_practicum_cytokines/lisa/paper_leiden_clusters_deg"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLUSTER_COL = "paper_cluster"
SAMPLE_COL = "display_name"
EXCLUDED_CLUSTERS = [14, 15, 16, 17, 18]
TIMEPOINTS = ["TP1M", "TP2M", "TP3M", "TP4M"]
MIN_CELLS, MIN_COUNTS = 10, 1000


def get_pseudobulk(adata, sample_col, groups_col, layer="counts", min_cells=10, min_counts=1000):
    pdata = dc.pp.pseudobulk(adata, sample_col=sample_col, groups_col=groups_col, layer=layer, mode="sum")
    dc.pp.filter_samples(pdata, min_cells=min_cells, min_counts=min_counts)
    return pdata


def filter_by_expr(pdata, group, min_count=10, min_total_count=15):
    return dc.pp.filter_by_expr(pdata, group=group, min_count=min_count,
                                min_total_count=min_total_count, inplace=False)


def run_deseq2_per_cluster(pdata, condition_col, group_a, group_b, cluster_col, min_samples=3, n_cpus=4):
    results = {}
    for cl in sorted(pdata.obs[cluster_col].unique(), key=int):
        mask = (pdata.obs[cluster_col] == cl) & (pdata.obs[condition_col].isin([group_a, group_b]))
        pdata_cl = pdata[mask].copy()
        n_a = (pdata_cl.obs[condition_col] == group_a).sum()
        n_b = (pdata_cl.obs[condition_col] == group_b).sum()
        if n_a < min_samples or n_b < min_samples:
            print(f"  SKIP cluster {cl}: {group_a}={n_a}, {group_b}={n_b} (need >={min_samples})")
            continue
        print(f"  cluster {cl}: {group_a}={n_a}, {group_b}={n_b} samples")
        genes_to_keep = filter_by_expr(pdata_cl, group=condition_col, min_count=10, min_total_count=15)
        pdata_cl = pdata_cl[:, genes_to_keep].copy()
        X = pdata_cl.X if not issparse(pdata_cl.X) else pdata_cl.X.toarray()
        counts_df = pd.DataFrame(X.astype(int), index=pdata_cl.obs_names, columns=pdata_cl.var_names)
        meta_df = pdata_cl.obs[[condition_col]].copy()
        meta_df[condition_col] = pd.Categorical(meta_df[condition_col], categories=[group_b, group_a])
        try:
            dds = DeseqDataSet(counts=counts_df, metadata=meta_df, design_factors=condition_col, n_cpus=n_cpus)
            dds.deseq2()
            stat_res = DeseqStats(dds, contrast=[condition_col, group_a, group_b], n_cpus=n_cpus)
            stat_res.summary()
            res = stat_res.results_df.copy()
            res["cluster"] = cl
            results[cl] = res
        except Exception as e:
            print(f"    ERROR in cluster {cl}: {e}")
    return results


def save_unfiltered(results_dict, tp, label):
    if not results_dict:
        print(f"No results for {label}")
        return
    all_res = pd.concat(results_dict.values())
    all_res["gene"] = all_res.index
    all_res["timepoint"] = tp
    all_res["cluster"] = all_res["cluster"].astype(int)  # store as plain int, not decoupler's required str
    cols = ["gene", "cluster", "timepoint", "baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"]
    all_res = all_res[cols].reset_index(drop=True)
    out_path = os.path.join(OUTPUT_DIR, f"{label}.csv")
    all_res.to_csv(out_path, index=False)
    print(f"Saved (unfiltered, {len(all_res)} rows): {out_path}")


print("=== Loading full h5ad + metadata ===")
adata = sc.read_h5ad(DATA_PATH)
meta = pd.read_parquet(METADATA_PATH)
adata.obs["paper_cluster_int"] = adata.obs["B2_Scanorama_Singlet_rb_mt_cluster"].astype(int)
adata.obs[CLUSTER_COL] = adata.obs["paper_cluster_int"].astype(str)  # decoupler pseudobulk needs a string groups_col
for col in ["condition", "timepoint", "display_name"]:
    adata.obs[col] = meta[col].reindex(adata.obs_names)

print(f"{adata.n_obs:,} cells total")

# ── STEP 1: cluster counts, exclusion ────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 1: cells and samples per cluster (paper's own clusters)")
print("=" * 70)
type_map = adata.obs.drop_duplicates("paper_cluster_int").set_index("paper_cluster_int")["cell_type_Scanorama"]
cell_counts = adata.obs["paper_cluster_int"].value_counts().sort_index()
adata.obs["patient_timepoint"] = adata.obs["display_name"].astype(str)
sample_counts = adata.obs.groupby("paper_cluster_int", observed=True)["patient_timepoint"].nunique().sort_index()
summary = pd.DataFrame({
    "type": type_map.reindex(cell_counts.index),
    "n_cells": cell_counts,
    "n_patient_timepoint_samples": sample_counts,
    "excluded": [c in EXCLUDED_CLUSTERS for c in cell_counts.index],
})
print(summary.to_string())
summary.to_csv(os.path.join(OUTPUT_DIR, "cluster_counts_summary.csv"))

retained_mask = ~adata.obs["paper_cluster_int"].isin(EXCLUDED_CLUSTERS)
print(f"\nExcluding clusters {EXCLUDED_CLUSTERS}: {(~retained_mask).sum():,} cells dropped, "
      f"{retained_mask.sum():,} cells retained (clusters 0-13)")
adata = adata[retained_mask].copy()

# ── STEP 2: DESeq2 per cluster, per timepoint ────────────────────────────
print("\n" + "=" * 70)
print("STEP 2: pseudobulk DESeq2, ACS per-timepoint vs CCS, all retained clusters")
print("=" * 70)
for tp in TIMEPOINTS:
    print(f"\n{'='*70}\nACS ({tp}) vs CCS -- all clusters 0-13\n{'='*70}")
    mask_tp = ((adata.obs["condition"] == "ACS_sterile") & (adata.obs["timepoint"] == tp)) | \
              (adata.obs["condition"] == "CCS")
    adata_tp = adata[mask_tp].copy()
    pdata_tp = get_pseudobulk(adata_tp, SAMPLE_COL, CLUSTER_COL, layer="counts", min_cells=MIN_CELLS, min_counts=MIN_COUNTS)
    print(f"Pseudobulk samples: {pdata_tp.n_obs}")
    res = run_deseq2_per_cluster(pdata_tp, "condition", "ACS_sterile", "CCS", CLUSTER_COL)
    save_unfiltered(res, tp, f"PaperClusters_ACS_{tp}_vs_CCS")

print("\nDone.")
