#!/usr/bin/env python3
"""
DRVI Pseudobulk — aggregating factor activity from cell level to patient level

DRVI factors are available per cell, but the outcome (e.g. `classification`)
is per patient. This script summarizes the per-cell values into a single
metric per factor (mean and median), producing a unit x factors matrix that
can be used directly for patient-based downstream analyses (e.g.
classification by outcome).

Some patients have multiple samples at different timepoints (e.g. `m6.1` ..
`m6.4` for ACS patient 6, TP1-TP4; controls like `k9` have only one sample).
`sample_id`/`display_name` therefore encode "patient.timepoint", not the
patient alone (same convention as in `DEG/deg_conditions.ipynb`:
`patient_id = display_name.split(".")[0]`). This script therefore supports
two aggregation levels (`--level`):

  - "sample":  one row per sample (patient x timepoint). Timepoint signal is
               preserved; for repeated-measures analyses `patient_id`
               (included in the metadata) must be considered as a blocking
               factor (see deg_conditions.ipynb).
  - "patient": one row per actual patient, averaged across all
               timepoints/samples. Loses temporal resolution, but delivers
               the pure patient x factors matrix requested by the user (one
               patient = one row).

Usage:
    conda run -n mapra_cytokines python 2_2_b_drvi_pseudobulk.py \
        --embed-input .../visualization/drvi_interpretation/embed.h5ad \
        --output-dir  .../visualization/drvi_interpretation \
        --level patient
"""

import argparse
import os

import anndata as ad
import pandas as pd

# ── Argparse ──────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument(
    "--embed-input",
    default="/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/visualization/drvi_interpretation/embed.h5ad",
    help="embed.h5ad from 2_2_a_drvi_interpretation.py (cells x factors)",
)
parser.add_argument(
    "--summary-csv",
    default="/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/visualization/drvi_interpretation/interpretation_summary.csv",
    help="interpretation_summary.csv from 2_2_a — provides the list of active factors",
)
parser.add_argument(
    "--output-dir",
    default="/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/visualization/drvi_interpretation",
)
parser.add_argument("--sample-key", default="sample_id",
                    help="obs column identifying the sample (patient.timepoint)")
parser.add_argument("--outcome-key", default="classification",
                    help="obs column with the patient outcome (metadata export only)")
parser.add_argument("--factor-scope", choices=["active", "all"], default="active",
                    help="aggregate only active factors (from summary-csv) or all factors")
parser.add_argument("--level", choices=["sample", "patient"], default="patient",
                    help="aggregation level: 'sample' = per sample (patient.timepoint), "
                         "'patient' = per actual patient (averaged across timepoints)")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────

print("=== Load embedding ===")
embed = ad.read_h5ad(args.embed_input)
print(f"Embedding: {embed.n_obs} cells x {embed.n_vars} factors")

if args.factor_scope == "active" and os.path.exists(args.summary_csv):
    summary = pd.read_csv(args.summary_csv, header=None, index_col=0).squeeze("columns")
    active_factors = [f.strip() for f in str(summary["active_factors"]).split(",")]
    factors = [f for f in active_factors if f in embed.var_names]
    print(f"Active factors (from {os.path.basename(args.summary_csv)}): {len(factors)}")
else:
    factors = embed.var_names.tolist()
    print(f"All factors: {len(factors)}")

if args.sample_key not in embed.obs.columns:
    raise ValueError(f"Sample key '{args.sample_key}' not present in embed.obs.")

# ── Derive patient/timepoint from the sample ID ──────────────────────────────
# Same convention as in DEG/deg_conditions.ipynb: "m6.4" -> patient_id="m6", timepoint="4"
# Samples without a dot (e.g. "k9", single draw) -> timepoint=None.

sample_ids = embed.obs[args.sample_key].astype(str)
split       = sample_ids.str.split(".", n=1, expand=True)
patient_id  = split[0]
timepoint   = split[1] if split.shape[1] > 1 else pd.Series(pd.NA, index=sample_ids.index)

samples_per_patient = pd.DataFrame({"patient_id": patient_id.values, "sample": sample_ids.values}) \
    .drop_duplicates().groupby("patient_id", observed=True).size()
n_patients_multi = (samples_per_patient > 1).sum()
print(f"\n{sample_ids.nunique()} samples -> {patient_id.nunique()} patients "
      f"({n_patients_multi} of which have multiple timepoint samples)")

group_key = args.sample_key if args.level == "sample" else "patient_id"

# ── Pseudobulk: cell -> unit (sample or patient) ─────────────────────────────

print(f"\n=== Aggregation at '{args.level}' level (grouping: {group_key}) ===")

latent = embed[:, factors].X
if hasattr(latent, "toarray"):
    latent = latent.toarray()

factor_df = pd.DataFrame(latent, index=embed.obs_names, columns=factors)
factor_df["patient_id"] = patient_id.values
factor_df[args.sample_key] = sample_ids.values

grouped = factor_df.groupby(group_key, observed=True)

mean_matrix   = grouped[factors].mean()
median_matrix = grouped[factors].median()
n_cells       = grouped.size().rename("n_cells")

print(f"Matrix shape ({args.level.capitalize()} x factors): {mean_matrix.shape}")

# ── Save ────────────────────────────────────────────────────────────────────

mean_path   = os.path.join(args.output_dir, f"pseudobulk_{args.level}_factor_mean.csv")
median_path = os.path.join(args.output_dir, f"pseudobulk_{args.level}_factor_median.csv")
meta_path   = os.path.join(args.output_dir, f"pseudobulk_{args.level}_metadata.csv")

mean_matrix.to_csv(mean_path)
median_matrix.to_csv(median_path)
print(f"Saved: {os.path.basename(mean_path)}")
print(f"Saved: {os.path.basename(median_path)}")

# Metadata is kept separate (outcome + cell count + timepoint info) so that
# the factor matrices themselves stay purely numeric.
obs_extra = embed.obs.copy()
obs_extra["patient_id"] = patient_id.values
obs_extra[args.sample_key] = sample_ids.values

meta_cols = [c for c in [args.outcome_key, "library"] if c in obs_extra.columns]
metadata = (
    obs_extra[[group_key] + meta_cols]
    .drop_duplicates(subset=group_key)
    .set_index(group_key)
)
metadata = metadata.join(n_cells)

if args.level == "patient":
    n_timepoints = (
        obs_extra[["patient_id", args.sample_key]]
        .drop_duplicates()
        .groupby("patient_id", observed=True)
        .size()
        .rename("n_samples")
    )
    metadata = metadata.join(n_timepoints)
else:
    metadata["patient_id"] = obs_extra.drop_duplicates(subset=group_key).set_index(group_key)["patient_id"]

metadata.to_csv(meta_path)
print(f"Saved: {os.path.basename(meta_path)}")

print(f"\nDone. {mean_matrix.shape[0]} {args.level}s x {mean_matrix.shape[1]} factors.")
print(f"Results in: {args.output_dir}")
