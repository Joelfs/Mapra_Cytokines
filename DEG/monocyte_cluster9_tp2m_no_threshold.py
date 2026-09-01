"""
Harmony leiden cluster 9 vs CCS, TP2M only, with NO per-patient minimum-cell
threshold (every patient with >=1 cell in cluster 9 counts as a pseudobulk
sample) -- as opposed to monocyte_cluster9_relaxed.py's min_cells=5 or the
project standard's min_cells=10. Just "those labeled cluster 9", full stop.

Single example timepoint to test whether finer Leiden clustering (cluster 0
vs cluster 9, both majority-vote Monocytes - CD14) surfaces genes the other
misses entirely.
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
adata.obs["harmony_leiden_int"] = adata.obs["harmony_leiden"].astype(int)
adata.obs["harmony_leiden"] = adata.obs["harmony_leiden"].astype(str)

adata_c9 = adata[adata.obs["harmony_leiden_int"] == CLUSTER].copy()
mask = ((adata_c9.obs["condition"] == "ACS_sterile") & (adata_c9.obs["timepoint"] == TP)) | \
       (adata_c9.obs["condition"] == "CCS")
sub = adata_c9[mask].copy()
print(f"Cluster {CLUSTER}, {TP} vs CCS: {sub.n_obs:,} cells -- {sub.obs['condition'].value_counts().to_dict()}")

pdata = dc.pp.pseudobulk(sub, sample_col="display_name", groups_col="harmony_leiden",
                          layer="counts", mode="sum")
# No min_cells threshold -- every patient with >=1 cell in cluster 9 counts.
# Keep a nominal min_counts floor (decoupler default) just to drop truly-empty samples.
dc.pp.filter_samples(pdata, min_cells=1, min_counts=1)
n_acs = (pdata.obs["condition"] == "ACS_sterile").sum()
n_ccs = (pdata.obs["condition"] == "CCS").sum()
print(f"Pseudobulk samples, no cell-count threshold: ACS_sterile={n_acs}, CCS={n_ccs}")

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

out_path = os.path.join(OUTPUT_DIR, f"Harmony_leiden_ACS_{TP}_vs_CCS_cluster9_no_threshold.csv")
res.to_csv(out_path, index=False)
n_sig = int(res["significant"].sum())
print(f"\nSaved: {out_path} ({n_sig} significant genes)")
