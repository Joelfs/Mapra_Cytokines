"""
Harmony leiden cluster 9 (Monocytes - CD14, purity 77.4%) vs CCS, per
timepoint -- run with a RELAXED min_cells threshold (5 instead of the
standard 10 used everywhere else in this project).

Why: cluster 9 has only 72 CCS cells total across 14 CCS patients, and only
1 of them clears the standard 10-cell threshold -- so at the standard
threshold this comparison can never reach the min_samples=3 DESeq2 needs on
the CCS side, in any timepoint (same fixed CCS pool reused every time,
since CCS patients have a single blood draw, no timepoints). At min_cells=5,
7 CCS patients qualify, and ACS-sterile has 9-19 qualifying patients per
timepoint -- so the comparison becomes viable.

This IS a lower-confidence result than the rest of the project (each
pseudobulk sample is built from as few as 5 cells, vs >=10 everywhere else)
-- flagged in the output filename and should be flagged in any plot title
that uses it, not presented as equivalent to the standard-threshold results.

Output: Harmony_leiden_ACS_{tp}_vs_CCS_cluster9_relaxed.csv, same schema as
the main pipeline's per-celltype CSVs (gene, baseMean, log2FoldChange, lfcSE,
stat, pvalue, padj, cell_type, significant), cluster 9 only, one row per gene.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import decoupler as dc
from scipy.sparse import issparse
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

DATA_PATH = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/data/data_for_practicum_post_integration.h5ad"
OUTPUT_DIR = "/vol/disk/ubuntu/master_practicum_cytokines/lisa/deg_conditions"
CLUSTER = 9
MIN_CELLS = 5  # relaxed from the standard 10
PADJ_THRESH = 0.05
LFC_THRESH = 1.0

print("Loading data...")
adata = sc.read_h5ad(DATA_PATH)


def parse_timepoint(display_name):
    s = str(display_name)
    return "TP0M" if "." not in s else f"TP{s.split('.')[-1]}M"


adata.obs["timepoint"] = adata.obs["display_name"].apply(parse_timepoint)
condition_map = {
    "acs_w_o_infection": "ACS_sterile", "acs_w_infection": "ACS_infection",
    "acs_subacute": "ACS_subacute", "ccs": "CCS",
    "vollstaendiger_ausschluss": "non-CCS", "koronarsklerose": "Sclerosis",
}
adata.obs["condition"] = adata.obs["classification"].map(condition_map)
adata.obs["harmony_leiden_int"] = adata.obs["harmony_leiden"].astype(int)
adata.obs["harmony_leiden"] = adata.obs["harmony_leiden"].astype(str)

adata_c9 = adata[adata.obs["harmony_leiden_int"] == CLUSTER].copy()
print(f"Cluster {CLUSTER}: {adata_c9.n_obs:,} cells total")

for tp in ["TP1M", "TP2M", "TP3M", "TP4M"]:
    print(f"\n{'='*60}\nCluster {CLUSTER}, {tp} vs CCS (min_cells={MIN_CELLS})\n{'='*60}")
    mask = ((adata_c9.obs["condition"] == "ACS_sterile") & (adata_c9.obs["timepoint"] == tp)) | \
           (adata_c9.obs["condition"] == "CCS")
    sub = adata_c9[mask].copy()
    print(f"{sub.n_obs:,} cells -- {sub.obs['condition'].value_counts().to_dict()}")

    pdata = dc.pp.pseudobulk(sub, sample_col="display_name", groups_col="harmony_leiden",
                              layer="counts", mode="sum")
    dc.pp.filter_samples(pdata, min_cells=MIN_CELLS, min_counts=1000)
    n_acs = (pdata.obs["condition"] == "ACS_sterile").sum()
    n_ccs = (pdata.obs["condition"] == "CCS").sum()
    print(f"Pseudobulk samples after filter_samples(min_cells={MIN_CELLS}): "
          f"ACS_sterile={n_acs}, CCS={n_ccs}")

    if n_acs < 3 or n_ccs < 3:
        print(f"  SKIP: still not enough samples (need >=3 each)")
        continue

    pdata.obs["condition"] = pdata.obs["condition"].astype(str)
    genes_to_keep = dc.pp.filter_by_expr(pdata, group="condition", min_count=10,
                                          min_total_count=15, inplace=False)
    pdata = pdata[:, genes_to_keep].copy()
    print(f"Genes after filter_by_expr: {pdata.n_vars}")

    X = pdata.X if not issparse(pdata.X) else pdata.X.toarray()
    counts_df = pd.DataFrame(X.astype(int), index=pdata.obs_names, columns=pdata.var_names)
    meta_df = pdata.obs[["condition"]].copy()
    meta_df["condition"] = pd.Categorical(meta_df["condition"], categories=["CCS", "ACS_sterile"])

    dds = DeseqDataSet(counts=counts_df, metadata=meta_df, design_factors=["condition"], n_cpus=4)
    dds.deseq2()
    stat_res = DeseqStats(dds, contrast=["condition", "ACS_sterile", "CCS"], n_cpus=4)
    stat_res.summary()
    res = stat_res.results_df.copy()
    res["gene"] = res.index
    res["cell_type"] = CLUSTER
    res["significant"] = (res["padj"] < PADJ_THRESH) & (res["log2FoldChange"].abs() > LFC_THRESH)

    out_path = os.path.join(OUTPUT_DIR, f"Harmony_leiden_ACS_{tp}_vs_CCS_cluster9_relaxed.csv")
    res.to_csv(out_path, index=False)
    n_sig = int(res["significant"].sum())
    print(f"Saved: {out_path} ({n_sig} significant genes)")

print("\nDone.")
