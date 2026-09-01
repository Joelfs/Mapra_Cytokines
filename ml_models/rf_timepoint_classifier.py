#!/usr/bin/env python3
"""
Random-Forest-Klassifikator: sagt den Zeitpunkt (TP1-TP4) einer Probe aus den
gemittelten aktiven DRVI-Faktoren voraus — nur fuer ACS_sterile
(classification == "acs_w_o_infection", 19 Patienten, 62 Proben).

Nutzt data_for_practicum_post_integration_curated_activedims_split.h5ad,
Spalten `holdout_timepoint`/`cv_fold_timepoint` (siehe
integration/2_D_drvi_curation_ig_genes.ipynb, Sektion 7). Anders als
rf_condition_classifier.py wird hier auf **Proben-Ebene** (Patient.Zeitpunkt,
z.B. "m6.1") pseudobulked, NICHT auf Patienten-Ebene — der Zeitpunkt ist ja
gerade die Zielgroesse. Die Split-/Fold-Zuordnung ist trotzdem
Patienten-gruppiert (StratifiedGroupKFold): alle Proben eines Patienten
bleiben im selben Split, um Data Leakage zu vermeiden.

Ablauf:
  1. Pseudobulk auf Proben-Ebene: Mittelwert der aktiven DRVI-Faktoren ueber
     alle Zellen EINER Probe (nicht ueber alle Zeitpunkte eines Patienten).
  2. 5-fach Cross-Validation innerhalb des CV-Pools (`holdout_timepoint == "cv"`),
     Fold-Zuordnung aus `cv_fold_timepoint` uebernommen.
  3. Finales Modell auf dem gesamten CV-Pool trainiert, einmalig auf dem nie
     angeruehrten Test-Holdout (`holdout_timepoint == "test"`) bewertet.

Ergebnis:
  - rf_timepoint_classifier.joblib
  - rf_timepoint_cv_fold_metrics.csv
  - rf_timepoint_test_holdout_report.txt
  - rf_timepoint_feature_importances.csv

Verwendung:
    conda run -n mapra_cytokines python rf_timepoint_classifier.py
"""

import argparse
import os

import numpy as np
import pandas as pd
import scanpy as sc
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-h5ad",
    default="/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/data/data_for_practicum_post_integration_curated_activedims_split.h5ad",
)
parser.add_argument(
    "--output-dir",
    default="/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/ml_models",
)
parser.add_argument("--n-estimators", type=int, default=500)
parser.add_argument("--max-depth", type=int, default=None)
parser.add_argument("--random-state", type=int, default=0)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ── 1. Daten laden + Pseudobulk auf PROBEN-Ebene ─────────────────────────────

print("=== Daten laden ===")
adata = sc.read_h5ad(args.data_h5ad)
print(f"{adata.n_obs} Zellen, X_drvi shape: {adata.obsm['X_drvi'].shape}")

factors = adata.uns["X_drvi_active_dims"]
sample_id = adata.obs["sample_id"].astype(str)
timepoint = sample_id.str.split(".", n=1).str[1]  # "m6.1" -> "1"

feat_df = pd.DataFrame(adata.obsm["X_drvi"], columns=factors, index=adata.obs_names)
feat_df["sample_id"] = sample_id.values
feat_df["timepoint"] = timepoint.values
feat_df["holdout"] = adata.obs["holdout_timepoint"].astype(str).values
feat_df["cv_fold"] = adata.obs["cv_fold_timepoint"].values

n_before_cells = feat_df.shape[0]
feat_df = feat_df[feat_df["holdout"] != "excluded"]
print(f"{n_before_cells - feat_df.shape[0]} Zellen ausgeschlossen (nicht ACS_sterile)")

sample_df = feat_df.groupby("sample_id", observed=True).agg(
    {**{f: "mean" for f in factors}, "timepoint": "first", "holdout": "first", "cv_fold": "first"}
)
print(f"\n{sample_df.shape[0]} Proben x {len(factors)} Faktoren (Pseudobulk-Mittelwert pro Probe)")
print(sample_df["timepoint"].value_counts().sort_index())

# ── 2. 5-fach CV (Fold-Zuordnung aus cv_fold_timepoint uebernommen) ──────────

print("\n=== 5-fach Cross-Validation (CV-Pool) ===")
cv_mask = sample_df["holdout"] == "cv"
cv_df = sample_df[cv_mask]
n_folds = int(cv_df["cv_fold"].max()) + 1
print(f"CV-Pool: {int(cv_mask.sum())} Proben, {n_folds} Folds")
print(cv_df.groupby("cv_fold")["timepoint"].value_counts().unstack())

fold_rows = []
for fold in range(n_folds):
    train_mask = cv_mask & (sample_df["cv_fold"] != fold)
    val_mask = cv_mask & (sample_df["cv_fold"] == fold)

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth, class_weight="balanced",
        random_state=args.random_state, n_jobs=-1,
    )
    clf.fit(sample_df.loc[train_mask, factors], sample_df.loc[train_mask, "timepoint"])
    y_pred = clf.predict(sample_df.loc[val_mask, factors])
    y_true = sample_df.loc[val_mask, "timepoint"]

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    fold_rows.append({"fold": fold, "n_train": int(train_mask.sum()), "n_val": int(val_mask.sum()),
                      "accuracy": acc, "f1_macro": f1_macro})
    print(f"  Fold {fold}: n_train={int(train_mask.sum())}, n_val={int(val_mask.sum())}, "
          f"accuracy={acc:.3f}, f1_macro={f1_macro:.3f}")

fold_metrics = pd.DataFrame(fold_rows)
fold_metrics.to_csv(os.path.join(args.output_dir, "rf_timepoint_cv_fold_metrics.csv"), index=False)
print(f"\nCV-Mittelwert: accuracy={fold_metrics['accuracy'].mean():.3f} "
      f"(+/-{fold_metrics['accuracy'].std():.3f}), "
      f"f1_macro={fold_metrics['f1_macro'].mean():.3f} (+/-{fold_metrics['f1_macro'].std():.3f})")
print("Gespeichert: rf_timepoint_cv_fold_metrics.csv")

# ── 3. Finales Modell (ganzer CV-Pool) + Test-Holdout ────────────────────────

print("\n=== Finales Modell + Test-Holdout ===")
test_mask = sample_df["holdout"] == "test"
print(f"CV-Pool (Training): {int(cv_mask.sum())} Proben, Test-Holdout: {int(test_mask.sum())} Proben")

final_clf = RandomForestClassifier(
    n_estimators=args.n_estimators, max_depth=args.max_depth, class_weight="balanced",
    random_state=args.random_state, n_jobs=-1,
)
final_clf.fit(sample_df.loc[cv_mask, factors], sample_df.loc[cv_mask, "timepoint"])

y_pred_test = final_clf.predict(sample_df.loc[test_mask, factors])
y_true_test = sample_df.loc[test_mask, "timepoint"]

report = classification_report(y_true_test, y_pred_test, zero_division=0)
print(report)
print("Confusion matrix (Zeilen=wahr, Spalten=vorhergesagt):")
labels = sorted(sample_df["timepoint"].unique())
cm = confusion_matrix(y_true_test, y_pred_test, labels=labels)
print(pd.DataFrame(cm, index=labels, columns=labels))

with open(os.path.join(args.output_dir, "rf_timepoint_test_holdout_report.txt"), "w") as f:
    f.write("Zielgroesse: Zeitpunkt (TP1-TP4), nur ACS_sterile\n")
    f.write(f"Test-Holdout: {int(test_mask.sum())} Proben\n\n")
    f.write(report)
    f.write("\nConfusion matrix (Zeilen=wahr, Spalten=vorhergesagt):\n")
    f.write(pd.DataFrame(cm, index=labels, columns=labels).to_string())
print("\nGespeichert: rf_timepoint_test_holdout_report.txt")

# ── Feature Importances + Modell speichern ───────────────────────────────────

importances = pd.Series(final_clf.feature_importances_, index=factors).sort_values(ascending=False)
importances.to_csv(os.path.join(args.output_dir, "rf_timepoint_feature_importances.csv"), header=["importance"])
print("\nTop-10 wichtigste Faktoren:")
print(importances.head(10))
print("Gespeichert: rf_timepoint_feature_importances.csv")

model_path = os.path.join(args.output_dir, "rf_timepoint_classifier.joblib")
joblib.dump({"model": final_clf, "factors": factors, "classes": final_clf.classes_.tolist()}, model_path)
print(f"\nGespeichert: {model_path}")
