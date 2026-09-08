#!/usr/bin/env python3
"""
Post-hoc patient/sample-level evaluation for the already-trained MLP models.

mlp_condition_classifier.py / mlp_timepoint_classifier.py train and report at
cell level only (no majority-vote aggregation, unlike rf_*/xgb_*_classifier.py
which report both cell-level AND a majority-vote-per-patient/sample number).
This script does NOT retrain anything -- it loads the saved .joblib models,
re-derives the same test-holdout cells from the source h5ad (same holdout
columns the training scripts used), predicts, and aggregates cell predictions
to one prediction per patient (condition) / sample (timepoint) via majority
vote -- identical logic to rf_condition_classifier.py /
xgb_condition_classifier.py, so all four classifier types (LR, RF, XGB, MLP)
become comparable on the same patient/sample-level accuracy + f1_macro
metric.

Usage:
    conda run -n mapra_cytokines python mlp_patient_level_eval.py
"""
import os
import joblib
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

DATA_PATH = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/data/data_for_practicum_post_integration_curated_activedims_split.h5ad"
ML_DIR = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/ml_models"

CONDITION_MAP = {
    "acs_w_o_infection": "ACS",
    "ccs": "CCS",
    "vollstaendiger_ausschluss": "non-CCS",
}


def majority_vote(df, group_col, pred_col):
    return df.groupby(group_col, observed=True)[pred_col].agg(lambda s: s.value_counts().idxmax())


def eval_condition(adata):
    d = joblib.load(os.path.join(ML_DIR, "mlp_condition_classifier_stratified.joblib"))
    model, scaler, le, factors = d["model"], d["scaler"], d["label_encoder"], d["factors"]

    patient_id = adata.obs["sample_id"].astype(str).str.split(".", n=1).str[0]
    cell_df = pd.DataFrame(adata.obsm["X_drvi"], columns=factors, index=adata.obs_names)
    cell_df["patient_id"] = patient_id.values
    cell_df["condition"] = adata.obs["classification"].astype(str).map(CONDITION_MAP).values
    cell_df["holdout"] = adata.obs["holdout_stratified"].astype(str).values
    cell_df = cell_df[cell_df["holdout"] == "test"].copy()

    X_scaled = scaler.transform(cell_df[factors])
    pred_enc = model.predict(X_scaled)
    cell_df["pred"] = le.inverse_transform(pred_enc)

    sample_pred = majority_vote(cell_df, "patient_id", "pred")
    sample_true = cell_df.drop_duplicates("patient_id").set_index("patient_id")["condition"].reindex(sample_pred.index)

    acc = accuracy_score(sample_true, sample_pred)
    f1_macro = f1_score(sample_true, sample_pred, average="macro", zero_division=0)
    report = classification_report(sample_true, sample_pred, zero_division=0)
    labels = sorted(cell_df["condition"].unique())
    cm = confusion_matrix(sample_true, sample_pred, labels=labels)

    print("=== MLP condition, patient-level (majority vote) ===")
    print(f"n_patients={len(sample_pred)}, accuracy={acc:.3f}, f1_macro={f1_macro:.3f}")
    print(report)

    out_path = os.path.join(ML_DIR, "mlp_test_holdout_report_stratified_patient_level.txt")
    with open(out_path, "w") as f:
        f.write("Split-Schema: stratified\n")
        f.write(f"Test-Holdout: {len(sample_pred)} Patienten (Mehrheitsentscheid ueber "
                f"{cell_df.shape[0]} Zellen, post-hoc aus gespeichertem Zell-Ebene-Modell)\n\n")
        f.write(report)
        f.write("\nConfusion matrix (Zeilen=wahr, Spalten=vorhergesagt):\n")
        f.write(pd.DataFrame(cm, index=labels, columns=labels).to_string())
    print(f"Saved: {out_path}\n")
    return {"model": "MLP", "task": "condition", "accuracy": acc, "f1_macro": f1_macro, "n": len(sample_pred)}


def eval_timepoint(adata):
    d = joblib.load(os.path.join(ML_DIR, "mlp_timepoint_classifier.joblib"))
    model, scaler, le, factors = d["model"], d["scaler"], d["label_encoder"], d["factors"]

    sample_id = adata.obs["sample_id"].astype(str)
    timepoint = sample_id.str.split(".", n=1).str[1]
    cell_df = pd.DataFrame(adata.obsm["X_drvi"], columns=factors, index=adata.obs_names)
    cell_df["sample_id"] = sample_id.values
    cell_df["timepoint"] = timepoint.values
    cell_df["holdout"] = adata.obs["holdout_timepoint"].astype(str).values
    cell_df = cell_df[cell_df["holdout"] == "test"].copy()

    X_scaled = scaler.transform(cell_df[factors])
    pred_enc = model.predict(X_scaled)
    cell_df["pred"] = le.inverse_transform(pred_enc)

    sample_pred = majority_vote(cell_df, "sample_id", "pred")
    sample_true = cell_df.drop_duplicates("sample_id").set_index("sample_id")["timepoint"].reindex(sample_pred.index)

    acc = accuracy_score(sample_true, sample_pred)
    f1_macro = f1_score(sample_true, sample_pred, average="macro", zero_division=0)
    report = classification_report(sample_true, sample_pred, zero_division=0)
    labels = sorted(cell_df["timepoint"].unique())
    cm = confusion_matrix(sample_true, sample_pred, labels=labels)

    print("=== MLP timepoint, sample-level (majority vote) ===")
    print(f"n_samples={len(sample_pred)}, accuracy={acc:.3f}, f1_macro={f1_macro:.3f}")
    print(report)

    out_path = os.path.join(ML_DIR, "mlp_timepoint_test_holdout_report_sample_level.txt")
    with open(out_path, "w") as f:
        f.write("Zielgroesse: Zeitpunkt (TP1-TP4), nur ACS_sterile\n")
        f.write(f"Test-Holdout: {len(sample_pred)} Proben (Mehrheitsentscheid ueber "
                f"{cell_df.shape[0]} Zellen, post-hoc aus gespeichertem Zell-Ebene-Modell)\n\n")
        f.write(report)
        f.write("\nConfusion matrix (Zeilen=wahr, Spalten=vorhergesagt):\n")
        f.write(pd.DataFrame(cm, index=labels, columns=labels).to_string())
    print(f"Saved: {out_path}\n")
    return {"model": "MLP", "task": "timepoint", "accuracy": acc, "f1_macro": f1_macro, "n": len(sample_pred)}


def main():
    print("Loading h5ad...")
    adata = sc.read_h5ad(DATA_PATH)
    rows = [eval_condition(adata), eval_timepoint(adata)]
    pd.DataFrame(rows).to_csv(os.path.join(ML_DIR, "mlp_patient_level_summary.csv"), index=False)
    print("Saved: mlp_patient_level_summary.csv")


if __name__ == "__main__":
    main()
