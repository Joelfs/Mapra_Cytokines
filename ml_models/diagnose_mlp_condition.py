#!/usr/bin/env python3
"""
Diagnose the suspiciously perfect patient-level CV/test accuracy of
mlp_condition_classifier.py: is the model actually learning ACS/CCS/non-CCS
biology, or just recognizing per-patient technical signatures (which happen
to perfectly determine condition, since every one of a patient's cells
always carries the same condition label)?

Loads the already-saved model (mlp_condition_classifier_stratified.joblib)
and re-predicts on the test-holdout cells (never used for training), then
reports, per patient, what fraction of their cells were predicted as each
class. If a patient's cells are almost all one class (near 100% one way),
that's consistent with the model keying on something patient-specific
rather than a graded biological signal. If the split is more mixed/noisy
(e.g. 40/35/25), that's more consistent with a real, if modest, per-cell
signal that majority vote is amplifying.

Usage:
    conda run -n mapra_cytokines python diagnose_mlp_condition.py --split stratified
"""

import argparse
import os

import numpy as np
import pandas as pd
import scanpy as sc
import joblib

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-h5ad",
    default="/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/data/data_for_practicum_post_integration_curated_activedims_split.h5ad",
)
parser.add_argument(
    "--model-dir",
    default="/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/ml_models",
)
parser.add_argument("--split", default="stratified")
args = parser.parse_args()

CONDITION_MAP = {
    "acs_w_o_infection": "ACS",
    "ccs": "CCS",
    "vollstaendiger_ausschluss": "non-CCS",
}

model_path = os.path.join(args.model_dir, f"mlp_condition_classifier_{args.split}.joblib")
print(f"Loading model: {model_path}")
bundle = joblib.load(model_path)
model = bundle["model"]
scaler = bundle["scaler"]
le = bundle["label_encoder"]
factors = bundle["factors"]

print("Loading data...")
adata = sc.read_h5ad(args.data_h5ad)
holdout_col = f"holdout_{args.split}"
patient_id = adata.obs["sample_id"].astype(str).str.split(".", n=1).str[0]

cell_df = pd.DataFrame(adata.obsm["X_drvi"], columns=factors, index=adata.obs_names)
cell_df["patient_id"] = patient_id.values
cell_df["condition"] = adata.obs["classification"].astype(str).map(CONDITION_MAP).values
cell_df["holdout"] = adata.obs[holdout_col].astype(str).values

test_df = cell_df[cell_df["holdout"] == "test"].copy()
print(f"Test-holdout: {test_df.shape[0]} cells from {test_df['patient_id'].nunique()} patients\n")

X_test_scaled = scaler.transform(test_df[factors])
pred_enc = model.predict(X_test_scaled)
pred = le.inverse_transform(pred_enc)
test_df["pred"] = pred

print("Per-patient cell-level prediction breakdown (true condition | predicted-class fractions):")
print("=" * 80)
for patient, group in test_df.groupby("patient_id", observed=True):
    true_cond = group["condition"].iloc[0]
    frac = group["pred"].value_counts(normalize=True).round(3).to_dict()
    majority = group["pred"].value_counts().idxmax()
    correct = "CORRECT" if majority == true_cond else "WRONG"
    print(f"Patient {patient}: true={true_cond:8s} n_cells={len(group):5d}  "
          f"pred_fractions={frac}  majority_vote={majority:8s} [{correct}]")

print("\n" + "=" * 80)
print("Interpretation: if pred_fractions are close to 1.0 for one class, that patient's")
print("cells are being predicted almost uniformly -- consistent with a per-patient")
print("signature. If fractions are more spread out (e.g. 0.45/0.30/0.25), majority vote")
print("is amplifying a weaker, more graded signal instead.")