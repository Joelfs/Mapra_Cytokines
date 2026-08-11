#!/usr/bin/env python3
"""
DRVI cluster differential analysis — which factors drive a Leiden cluster?

For a given Leiden cluster (default: cluster 13, res=1.0, from
data/data_for_practicum_post_integration.h5ad), the mean score within the
cluster is compared to the mean of all other cells for each DRVI factor. The
factors are sorted by absolute difference to see which dimension
distinguishes the cluster most strongly from all other cells.

Usage:
    conda run -n mapra_cytokines python 2_5_a_drvi_cluster_analysis.py --cluster 13
"""

import argparse
import os

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

# ── Argparse ──────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    default="/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/data/data_for_practicum_post_integration.h5ad",
)
parser.add_argument("--leiden-key", default="drvi_leiden")
parser.add_argument("--embedding-key", default="X_drvi")
parser.add_argument("--cluster", default="13")
parser.add_argument(
    "--output-dir",
    default="/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/visualization/drvi_interpretation",
)
parser.add_argument("--mapping-csv",
                    default="/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/visualization/drvi_interpretation/factor_title_mapping.csv",
                    help="Optional: factor_title_mapping.csv from 2_2_b, only used for the 'vanished' status")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ── Load data (only obs + obsm needed, no X) ─────────────────────────────────

print("=== Load data ===")
adata = ad.read_h5ad(args.input, backed="r")

if args.leiden_key not in adata.obs.columns:
    raise ValueError(f"'{args.leiden_key}' not present in obs. Available: {list(adata.obs.columns)}")

cluster_labels = adata.obs[args.leiden_key].astype(str)
if args.cluster not in cluster_labels.unique():
    raise ValueError(f"Cluster '{args.cluster}' not present in '{args.leiden_key}'. "
                     f"Available: {sorted(cluster_labels.unique(), key=lambda x: int(x) if x.isdigit() else x)}")

latent = np.asarray(adata.obsm[args.embedding_key])
n_dims = latent.shape[1]
dim_names = [f"DR{i+1}" for i in range(n_dims)]

mask_in  = (cluster_labels == args.cluster).values
mask_out = ~mask_in
print(f"Cluster {args.cluster}: {mask_in.sum()} cells  |  Rest: {mask_out.sum()} cells")

# ── Per dimension: mean in cluster vs. rest, difference + context ───────────
# A raw difference alone doesn't say whether the signal is large relative to
# the dimension's natural spread. So additionally:
#   - std per group (context for the difference)
#   - Cohen's d (difference standardized by the pooled spread)
#   - Welch's t-test (robust with very unequal group sizes:
#     1100 vs. 108404 cells -> a classic Student's t-test assuming equal
#     variance would not be appropriate here)

x_in  = latent[mask_in]
x_out = latent[mask_out]

mean_in  = x_in.mean(axis=0)
mean_out = x_out.mean(axis=0)
std_in   = x_in.std(axis=0, ddof=1)
std_out  = x_out.std(axis=0, ddof=1)
diff     = mean_in - mean_out

n_in, n_out = mask_in.sum(), mask_out.sum()
pooled_std = np.sqrt(
    ((n_in - 1) * std_in**2 + (n_out - 1) * std_out**2) / (n_in + n_out - 2)
)
cohens_d = diff / pooled_std

t_stat, p_val = stats.ttest_ind(x_in, x_out, axis=0, equal_var=False)

result = pd.DataFrame({
    "factor": dim_names,
    "mean_in_cluster": mean_in,
    "mean_rest": mean_out,
    "std_in_cluster": std_in,
    "std_rest": std_out,
    "diff": diff,
    "abs_diff": np.abs(diff),
    "cohens_d": cohens_d,
    "abs_cohens_d": np.abs(cohens_d),
    "welch_t": t_stat,
    "p_value": p_val,
}).sort_values("abs_cohens_d", ascending=False).reset_index(drop=True)

if os.path.exists(args.mapping_csv):
    dim_map = pd.read_csv(args.mapping_csv, index_col=0)
    dim_map["vanished"] = dim_map["vanished"].astype(bool)
    result = result.merge(dim_map[["vanished"]], left_on="factor", right_index=True, how="left")
else:
    result["vanished"] = pd.NA

out_path = os.path.join(args.output_dir, f"cluster{args.cluster}_factor_diff.csv")
result.to_csv(out_path, index=False)
print(f"\nSaved: {os.path.basename(out_path)}  ({result.shape[0]} factors)")

print("\n=== Top 10 factors by |Cohen's d| (difference standardized by spread) ===")
print(result.head(10)[["factor", "vanished", "diff", "std_in_cluster", "std_rest",
                       "cohens_d", "p_value"]].to_string(index=False))

top = result.iloc[0]
print(f"\nLargest effect: {top['factor']}  Cohen's d={top['cohens_d']:+.2f}  "
      f"(diff={top['diff']:+.3f}, std_cluster={top['std_in_cluster']:.3f}, "
      f"std_rest={top['std_rest']:.3f}, p={top['p_value']:.1e}, vanished={top['vanished']})")

print("\nFor comparison — top 5 by raw |difference| (without spread context):")
print(result.sort_values("abs_diff", ascending=False).head(5)[["factor", "diff", "cohens_d"]].to_string(index=False))

print(f"\nDone. Results in: {args.output_dir}")
