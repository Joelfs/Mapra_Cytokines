#!/usr/bin/env python3
"""
Random-Forest-Klassifikator: sagt die Bedingung (ACS / CCS / non-CCS) eines
Patienten aus den gemittelten aktiven DRVI-Faktoren voraus.

Nutzt data_for_practicum_post_integration_curated_activedims_split.h5ad
(kuratiert, auf aktive Dimensionen reduziert, siehe
integration/2_D_drvi_curation_ig_genes.ipynb, Sektion 5+6). Die Datei enthaelt
DREI Split-Schemata als Spalten-Paare `holdout_<schema>`/`cv_fold_<schema>`;
mit --split waehlt man, welches verwendet wird:

  - stratified     15% Test-Holdout, Rest per StratifiedKFold in 5 Folds
                    (Fold-Groessen orientieren sich an den natuerlichen
                    Gruppengroessen, kein fixer Wert pro Fold)
  - balanced_322    jeder Fold hat exakt 3 ACS / 2 CCS / 2 non-CCS, Rest im
                    Holdout
  - balanced_222    jeder Fold hat exakt 2 ACS / 2 CCS / 2 non-CCS, Rest im
                    Holdout (kleinster, aber am staerksten balancierter Pool)
  - balanced_432    4-fach CV, 4 ACS / 3 CCS / 2 non-CCS pro Fold. Nutzt alle
                    45 Patienten exakt aus (19=4x4+3, 16=3x4+4, 10=2x4+2) —
                    einziges balanciertes Schema mit non-CCS im Test-Holdout.

Bedingungs-Mapping eingeschraenkt: ACS = nur "acs_w_o_infection" (19
Patienten), CCS = "ccs" (16), non-CCS = nur "vollstaendiger_ausschluss" (10).
"acs_w_infection", "acs_subacute" und "koronarsklerose" sind komplett
ausgeschlossen (Patienten dieser Kategorien haben in den Split-Spalten den
Wert "excluded"/-2 und werden hier automatisch aus dem Training entfernt).

Ablauf (gleich fuer alle Schemata):
  1. Pseudobulk auf Patienten-Ebene: Mittelwert der aktiven DRVI-Faktoren
     ueber alle Zellen eines Patienten (alle Zeitpunkte zusammen).
  2. 5-fach Cross-Validation innerhalb des CV-Pools (`holdout_<schema> == "cv"`),
     Fold-Zuordnung aus `cv_fold_<schema>` uebernommen.
  3. Finales Modell auf dem gesamten CV-Pool trainiert, einmalig auf dem nie
     angeruehrten Test-Holdout (`holdout_<schema> == "test"`) bewertet.

Ergebnis (Dateinamen mit Schema-Suffix):
  - rf_condition_classifier_<schema>.joblib
  - rf_cv_fold_metrics_<schema>.csv
  - rf_test_holdout_report_<schema>.txt
  - rf_feature_importances_<schema>.csv

Verwendung:
    conda run -n mapra_cytokines python rf_condition_classifier.py --split stratified
    conda run -n mapra_cytokines python rf_condition_classifier.py --split balanced_322
    conda run -n mapra_cytokines python rf_condition_classifier.py --split balanced_222
    conda run -n mapra_cytokines python rf_condition_classifier.py --split balanced_432
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
parser.add_argument("--split", choices=["stratified", "balanced_322", "balanced_222", "balanced_432"], default="stratified",
                    help="Welches Split-Schema (Spalten holdout_<schema>/cv_fold_<schema>) verwendet wird.")
parser.add_argument("--n-estimators", type=int, default=500)
parser.add_argument("--max-depth", type=int, default=None)
parser.add_argument("--random-state", type=int, default=0)
args = parser.parse_args()

# Gleiches (eingeschraenktes) Mapping wie beim Split (2_D, Notebook-Sektion 6):
# nur diese drei classification-Werte zaehlen ueberhaupt als ACS/CCS/non-CCS.
# Alle anderen Patienten (acs_w_infection, acs_subacute, koronarsklerose)
# sind in den Split-Spalten als "excluded"/-2 markiert und werden unten
# automatisch aus dem Training entfernt.
CONDITION_MAP = {
    "acs_w_o_infection": "ACS",
    "ccs": "CCS",
    "vollstaendiger_ausschluss": "non-CCS",
}

os.makedirs(args.output_dir, exist_ok=True)
holdout_col = f"holdout_{args.split}"
fold_col = f"cv_fold_{args.split}"
suffix = f"_{args.split}"

# ── 1. Daten laden + Pseudobulk auf Patienten-Ebene ──────────────────────────

print(f"=== Daten laden (Split-Schema: {args.split}) ===")
adata = sc.read_h5ad(args.data_h5ad)
print(f"{adata.n_obs} Zellen, X_drvi shape: {adata.obsm['X_drvi'].shape}")

factors = adata.uns["X_drvi_active_dims"]
patient_id = adata.obs["sample_id"].astype(str).str.split(".", n=1).str[0]

feat_df = pd.DataFrame(adata.obsm["X_drvi"], columns=factors, index=adata.obs_names)
feat_df["patient_id"] = patient_id.values
feat_df["condition"] = adata.obs["classification"].astype(str).map(CONDITION_MAP).values
feat_df["holdout"] = adata.obs[holdout_col].astype(str).values
feat_df["cv_fold"] = adata.obs[fold_col].values

patient_df = feat_df.groupby("patient_id", observed=True).agg(
    {**{f: "mean" for f in factors}, "condition": "first", "holdout": "first", "cv_fold": "first"}
)

n_before = patient_df.shape[0]
patient_df = patient_df[patient_df["holdout"] != "excluded"].copy()
print(f"\n{n_before} Patienten insgesamt, {n_before - patient_df.shape[0]} ausgeschlossen "
      f"(classification ausserhalb ACS/CCS/non-CCS-Mapping)")
print(f"{patient_df.shape[0]} Patienten x {len(factors)} Faktoren (Pseudobulk-Mittelwert)")
print(patient_df["condition"].value_counts())

# ── 2. 5-fach CV (Fold-Zuordnung aus cv_fold_<schema> uebernommen) ───────────

print(f"\n=== 5-fach Cross-Validation (CV-Pool, Schema={args.split}) ===")
cv_mask = patient_df["holdout"] == "cv"
cv_df = patient_df[cv_mask]
n_folds = int(cv_df["cv_fold"].max()) + 1
print(f"CV-Pool: {int(cv_mask.sum())} Patienten, {n_folds} Folds")
print(cv_df.groupby("cv_fold")["condition"].value_counts().unstack())

fold_rows = []
for fold in range(n_folds):
    train_mask = cv_mask & (patient_df["cv_fold"] != fold)
    val_mask = cv_mask & (patient_df["cv_fold"] == fold)

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth, class_weight="balanced",
        random_state=args.random_state, n_jobs=-1,
    )
    clf.fit(patient_df.loc[train_mask, factors], patient_df.loc[train_mask, "condition"])
    y_pred = clf.predict(patient_df.loc[val_mask, factors])
    y_true = patient_df.loc[val_mask, "condition"]

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    fold_rows.append({"fold": fold, "n_train": int(train_mask.sum()), "n_val": int(val_mask.sum()),
                      "accuracy": acc, "f1_macro": f1_macro})
    print(f"  Fold {fold}: n_train={int(train_mask.sum())}, n_val={int(val_mask.sum())}, "
          f"accuracy={acc:.3f}, f1_macro={f1_macro:.3f}")

fold_metrics = pd.DataFrame(fold_rows)
fold_metrics.to_csv(os.path.join(args.output_dir, f"rf_cv_fold_metrics{suffix}.csv"), index=False)
print(f"\nCV-Mittelwert: accuracy={fold_metrics['accuracy'].mean():.3f} "
      f"(+/-{fold_metrics['accuracy'].std():.3f}), "
      f"f1_macro={fold_metrics['f1_macro'].mean():.3f} (+/-{fold_metrics['f1_macro'].std():.3f})")
print(f"Gespeichert: rf_cv_fold_metrics{suffix}.csv")

# ── 3. Finales Modell (ganzer CV-Pool) + Test-Holdout ────────────────────────

print("\n=== Finales Modell + Test-Holdout ===")
test_mask = patient_df["holdout"] == "test"
print(f"CV-Pool (Training): {int(cv_mask.sum())} Patienten, Test-Holdout: {int(test_mask.sum())} Patienten")

final_clf = RandomForestClassifier(
    n_estimators=args.n_estimators, max_depth=args.max_depth, class_weight="balanced",
    random_state=args.random_state, n_jobs=-1,
)
final_clf.fit(patient_df.loc[cv_mask, factors], patient_df.loc[cv_mask, "condition"])

y_pred_test = final_clf.predict(patient_df.loc[test_mask, factors])
y_true_test = patient_df.loc[test_mask, "condition"]

report = classification_report(y_true_test, y_pred_test, zero_division=0)
print(report)
print("Confusion matrix (Zeilen=wahr, Spalten=vorhergesagt):")
labels = sorted(patient_df["condition"].unique())
cm = confusion_matrix(y_true_test, y_pred_test, labels=labels)
print(pd.DataFrame(cm, index=labels, columns=labels))

with open(os.path.join(args.output_dir, f"rf_test_holdout_report{suffix}.txt"), "w") as f:
    f.write(f"Split-Schema: {args.split}\n")
    f.write(f"Test-Holdout: {int(test_mask.sum())} Patienten\n\n")
    f.write(report)
    f.write("\nConfusion matrix (Zeilen=wahr, Spalten=vorhergesagt):\n")
    f.write(pd.DataFrame(cm, index=labels, columns=labels).to_string())
print(f"\nGespeichert: rf_test_holdout_report{suffix}.txt")

# ── Feature Importances + Modell speichern ───────────────────────────────────

importances = pd.Series(final_clf.feature_importances_, index=factors).sort_values(ascending=False)
importances.to_csv(os.path.join(args.output_dir, f"rf_feature_importances{suffix}.csv"), header=["importance"])
print("\nTop-10 wichtigste Faktoren:")
print(importances.head(10))
print(f"Gespeichert: rf_feature_importances{suffix}.csv")

model_path = os.path.join(args.output_dir, f"rf_condition_classifier{suffix}.joblib")
joblib.dump({"model": final_clf, "factors": factors, "classes": final_clf.classes_.tolist(),
            "split_scheme": args.split}, model_path)
print(f"\nGespeichert: {model_path}")
