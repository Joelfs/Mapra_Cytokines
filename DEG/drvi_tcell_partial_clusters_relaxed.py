"""
DRVI T-cell-CD4 clusters 14, 16, 18 vs CCS, TP2M only -- run with a RELAXED
min_cells threshold (3 instead of the standard 10) since these are small,
partial-T-cell-CD4 clusters (887/680/314 total cells) that don't pass the
standard pipeline's sample-size filter.

CCS side sample counts at min_cells=3: cluster 14 = 6 patients, cluster 16 =
12 patients, cluster 18 = 5 patients (marginal, right at the min_samples=3
DESeq2 needs). Treat cluster 18 results as low-confidence.

Output: DRVI_leiden_ACS_TP2M_vs_CCS_cluster{14,16,18}_relaxed.csv, same
schema as the main pipeline's per-celltype CSVs.
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
CLUSTERS = [14, 16, 18]
MIN_CELLS = 3
TP = "TP2M"
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
adata.obs["drvi_leiden_int"] = adata.obs["drvi_leiden"].astype(int)
adata.obs["drvi_leiden"] = adata.obs["drvi_leiden"].astype(str)

for cluster in CLUSTERS:
    print(f"\n{'='*60}\nCluster {cluster}, {TP} vs CCS (min_cells={MIN_CELLS})\n{'='*60}")
    adata_c = adata[adata.obs["drvi_leiden_int"] == cluster].copy()
    mask = ((adata_c.obs["condition"] == "ACS_sterile") & (adata_c.obs["timepoint"] == TP)) | \
           (adata_c.obs["condition"] == "CCS")
    sub = adata_c[mask].copy()
    print(f"{sub.n_obs:,} cells -- {sub.obs['condition'].value_counts().to_dict()}")

    pdata = dc.pp.pseudobulk(sub, sample_col="display_name", groups_col="drvi_leiden",
                              layer="counts", mode="sum")
    dc.pp.filter_samples(pdata, min_cells=MIN_CELLS, min_counts=1000)
    n_acs = (pdata.obs["condition"] == "ACS_sterile").sum()
    n_ccs = (pdata.obs["condition"] == "CCS").sum()
    print(f"Pseudobulk samples: ACS_sterile={n_acs}, CCS={n_ccs}")

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
    res["cell_type"] = cluster
    res["significant"] = (res["padj"] < PADJ_THRESH) & (res["log2FoldChange"].abs() > LFC_THRESH)

    out_path = os.path.join(OUTPUT_DIR, f"DRVI_leiden_ACS_{TP}_vs_CCS_cluster{cluster}_relaxed.csv")
    res.to_csv(out_path, index=False)
    n_sig = int(res["significant"].sum())
    print(f"Saved: {out_path} ({n_sig} significant genes)")

print("\nDone.")
