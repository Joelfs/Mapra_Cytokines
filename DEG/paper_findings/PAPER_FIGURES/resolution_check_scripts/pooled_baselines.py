"""
Pooled baselines for Step 5's comparison: the full cell_type_Scanorama ==
"T-cell-CD4" / "Monocytes - CD14" populations (union of clusters 0+2+5+11
and 4+6+7 respectively), no cluster split, same PyDESeq2 setup, full
13-gene panel. Used to check what cluster-level resolution adds/loses
relative to pooling.
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

GENES = ["IL6ST", "JAK1", "STAT3", "SOCS3", "EIF3E", "HINT1", "HMGB1", "PIM1", "VCAN", "CD74", "UBC", "PSME2", "ODC1"]
TIMEPOINTS = ["TP1M", "TP2M", "TP3M", "TP4M"]
SAMPLE_COL = "display_name"
MIN_CELLS, MIN_COUNTS = 10, 1000

POPULATIONS = {
    "T-cell-CD4": "Pooled_CD4",
    "Monocytes - CD14": "Pooled_Monocytes",
}


def get_pseudobulk(adata, sample_col, groups_col, layer="counts", min_cells=10, min_counts=1000):
    pdata = dc.pp.pseudobulk(adata, sample_col=sample_col, groups_col=groups_col, layer=layer, mode="sum")
    dc.pp.filter_samples(pdata, min_cells=min_cells, min_counts=min_counts)
    return pdata


def filter_by_expr(pdata, group, min_count=10, min_total_count=15):
    return dc.pp.filter_by_expr(pdata, group=group, min_count=min_count,
                                min_total_count=min_total_count, inplace=False)


def run_deseq2(pdata, condition_col, group_a, group_b, label, n_cpus=4):
    n_a = (pdata.obs[condition_col] == group_a).sum()
    n_b = (pdata.obs[condition_col] == group_b).sum()
    print(f"  {label}: {group_a}={n_a}, {group_b}={n_b} samples")
    genes_to_keep = filter_by_expr(pdata, group=condition_col, min_count=10, min_total_count=15)
    pdata = pdata[:, genes_to_keep].copy()
    X = pdata.X if not issparse(pdata.X) else pdata.X.toarray()
    counts_df = pd.DataFrame(X.astype(int), index=pdata.obs_names, columns=pdata.var_names)
    meta_df = pdata.obs[[condition_col]].copy()
    meta_df[condition_col] = pd.Categorical(meta_df[condition_col], categories=[group_b, group_a])
    dds = DeseqDataSet(counts=counts_df, metadata=meta_df, design_factors=condition_col, n_cpus=n_cpus)
    dds.deseq2()
    stat_res = DeseqStats(dds, contrast=[condition_col, group_a, group_b], n_cpus=n_cpus)
    stat_res.summary()
    res = stat_res.results_df.copy()
    res["gene"] = res.index
    res["population"] = label
    return res[["gene", "population", "baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"]].reset_index(drop=True)


print("=== Loading full h5ad + metadata ===")
adata_full = sc.read_h5ad(DATA_PATH)
meta = pd.read_parquet(METADATA_PATH)
for col in ["condition", "timepoint", "display_name"]:
    adata_full.obs[col] = meta[col].reindex(adata_full.obs_names)
adata_full.obs["cell_type_Scanorama"] = meta["cell_type_Scanorama"].reindex(adata_full.obs_names)

for scanorama_type, label in POPULATIONS.items():
    print(f"\n{'#'*70}\n{label} ({scanorama_type})\n{'#'*70}")
    adata = adata_full[adata_full.obs["cell_type_Scanorama"] == scanorama_type].copy()
    adata.obs["group_col"] = label
    print(f"{adata.n_obs:,} cells")

    all_rows = []
    for tp in TIMEPOINTS:
        print(f"\n--- {label}: ACS ({tp}) vs CCS ---")
        mask_tp = ((adata.obs["condition"] == "ACS_sterile") & (adata.obs["timepoint"] == tp)) | \
                  (adata.obs["condition"] == "CCS")
        adata_tp = adata[mask_tp].copy()
        pdata_tp = get_pseudobulk(adata_tp, SAMPLE_COL, "group_col", layer="counts",
                                   min_cells=MIN_CELLS, min_counts=MIN_COUNTS)
        print(f"Pseudobulk samples: {pdata_tp.n_obs}")
        res = run_deseq2(pdata_tp, "condition", "ACS_sterile", "CCS", label)
        res["timepoint"] = tp
        all_rows.append(res)

    full_res = pd.concat(all_rows, ignore_index=True)
    out_path = os.path.join(OUTPUT_DIR, f"{label}_ACS_vs_CCS_per_timepoint.csv")
    full_res.to_csv(out_path, index=False)
    print(f"Saved (unfiltered, {len(full_res)} rows): {out_path}")

    print(f"\n=== {label}: log2FC / padj, 13-gene panel ===")
    for gene in GENES:
        parts = []
        for tp in TIMEPOINTS:
            row = full_res[(full_res["gene"] == gene) & (full_res["timepoint"] == tp)]
            parts.append(f"{tp}: NT" if row.empty else
                          f"{tp}: LFC={row['log2FoldChange'].iloc[0]:.2f}, padj={row['padj'].iloc[0]:.3g}")
        print(f"  {gene:8s} " + " | ".join(parts))

print("\nDone.")
