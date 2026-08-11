#!/usr/bin/env python3
"""
DRVI top-gene overview — top-N genes per (non-vanished) factor with score

Purely local evaluation of the results already produced by 2_2_a/2_2_b
(no model/data reload, no network requests):

  - factor_title_mapping.csv  (2_2_b): var_name -> title/order/vanished
  - ood_scores.csv or ind_scores.csv (2_2_a): genes x title-score matrix

Uses the correct var_name -> title mapping (see 2_2_b_drvi_factor_annotation.py
for the background on this) so that the genes really match the corresponding
factor in the pseudobulk matrix.

Result: top_genes_per_factor.csv in long format
    factor, direction, rank, gene, score

Usage:
    conda run -n mapra_cytokines python 2_2_c_drvi_top_genes.py --n-top-genes 10
"""

import argparse
import os

import pandas as pd

# ── Argparse ──────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output-dir",
    default="/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/visualization/drvi_interpretation",
)
parser.add_argument("--score-key", choices=["ood", "ind"], default="ood",
                    help="ood_scores.csv (OOD, default from 2_2_a) or ind_scores.csv (IND)")
parser.add_argument("--n-top-genes", type=int, default=10)
args = parser.parse_args()

mapping_path = os.path.join(args.output_dir, "factor_title_mapping.csv")
scores_path  = os.path.join(args.output_dir, f"{args.score_key}_scores.csv")

if not os.path.exists(mapping_path):
    raise FileNotFoundError(f"{mapping_path} not found — run 2_2_b_drvi_factor_annotation.py first.")
if not os.path.exists(scores_path):
    raise FileNotFoundError(f"{scores_path} not found — run 2_2_a_drvi_interpretation.py first.")

dim_map = pd.read_csv(mapping_path, index_col=0)
for col in ["vanished", "vanished_positive_direction", "vanished_negative_direction"]:
    dim_map[col] = dim_map[col].astype(bool)

scores = pd.read_csv(scores_path, index_col=0)

active_factors = dim_map.index[~dim_map["vanished"]].tolist()
print(f"Active factors: {len(active_factors)}")

# ── Top-N genes per factor direction ─────────────────────────────────────────

rows = []
for factor in active_factors:
    title = dim_map.loc[factor, "title"]
    for direction, vanished_col in [("+", "vanished_positive_direction"),
                                     ("-", "vanished_negative_direction")]:
        if dim_map.loc[factor, vanished_col]:
            continue
        col = f"{title}{direction}"
        if col not in scores.columns:
            print(f"  WARNING: column '{col}' for factor {factor} not found in {os.path.basename(scores_path)} — skipped.")
            continue
        top = scores[col].nlargest(args.n_top_genes)
        for rank, (gene, score) in enumerate(top.items(), start=1):
            rows.append({
                "factor": factor,
                "direction": direction,
                "rank": rank,
                "gene": gene,
                "score": score,
            })

top_genes_df = pd.DataFrame(rows)

out_path = os.path.join(args.output_dir, f"top_genes_per_factor_{args.score_key}.csv")
top_genes_df.to_csv(out_path, index=False)
print(f"\nSaved: {os.path.basename(out_path)}  ({top_genes_df.shape[0]} rows, "
      f"{top_genes_df.groupby(['factor', 'direction']).ngroups} factor directions)")

# Short console preview
preview_factors = active_factors[:3]
for factor in preview_factors:
    sub = top_genes_df[top_genes_df["factor"] == factor]
    for direction in ["+", "-"]:
        d = sub[sub["direction"] == direction]
        if d.empty:
            continue
        genes_str = ", ".join(f"{g} ({s:.2f})" for g, s in zip(d["gene"], d["score"]))
        print(f"  {factor}{direction}: {genes_str}")
