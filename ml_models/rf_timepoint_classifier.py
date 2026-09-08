#!/usr/bin/env python3
"""
Random-Forest-Klassifikator: sagt den Zeitpunkt (TP1-TP4) einer Probe aus den
aktiven DRVI-Faktoren voraus — nur fuer ACS_sterile
(classification == "acs_w_o_infection", 19 Patienten, 62 Proben).

Nutzt data_for_practicum_post_integration_curated_activedims_split.h5ad,
Spalten `holdout_timepoint`/`cv_fold_timepoint` (siehe
integration/2_D_drvi_curation_ig_genes.ipynb, Sektion 7). Die Split-/
Fold-Zuordnung ist Patienten-gruppiert (StratifiedGroupKFold): alle Proben
eines Patienten bleiben im selben Split, um Data Leakage zu vermeiden.

KEIN Pseudobulking mehr: statt vorher eine Zeile pro Probe (Mittelwert der
DRVI-Faktoren ueber alle Zellen dieser Probe) zu bilden, wird jetzt JEDE
EINZELNE ZELLE als eigene Trainings-/Test-Zeile verwendet (Label = der
Zeitpunkt der Probe, zu der die Zelle gehoert).

Weil die Bewertung pro Zelle (viele stark korrelierte Zeilen pro Probe) ein
anderes Bild liefert als die Frage "wird die Probe richtig klassifiziert",
werden Vorhersagen zusaetzlich per Mehrheitsentscheid ("majority vote") ueber
alle Zellen einer Probe zu einer Proben-Vorhersage aggregiert. Report/
Confusion-Matrix/`accuracy`+`f1_macro` in den Ergebnisdateien beziehen sich
auf diese Proben-Ebene (fuer Vergleichbarkeit mit den bisherigen
Pseudobulk-Ergebnissen); die reinen Zell-Ebene-Metriken stehen zusaetzlich
als `accuracy_cell`/`f1_macro_cell` in rf_timepoint_cv_fold_metrics.csv und
werden auf der Konsole ausgegeben.

Ablauf:
  1. Alle Zellen der Proben im CV-Pool/Test-Holdout, keine Aggregation.
  2. 5-fach Cross-Validation innerhalb des CV-Pools (`holdout_timepoint == "cv"`),
     Fold-Zuordnung aus `cv_fold_timepoint` uebernommen (Patienten-gruppiert).
  3. Finales Modell auf allen Zellen des CV-Pools trainiert, einmalig auf dem
     nie angeruehrten Test-Holdout bewertet (Zell- und Proben-Ebene).

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
parser.add_argument("--max-depth", type=int, default=10,
                    help="Ohne Pseudobulking (Zell-Ebene, >32000 Trainingszeilen) wird das RF bei "
                         "max_depth=None sehr tief und ueberanpasst stark an Zell-/Proben-Rauschen "
                         "(kollabiert auf Mehrheitsklasse) -- Default hier bewusst begrenzt.")
parser.add_argument("--min-samples-leaf", type=int, default=5,
                    help="Zweiter Regularisierungs-Hebel gegen Overfitting auf Zell-Ebene.")
parser.add_argument("--random-state", type=int, default=0)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)


def majority_vote(df, group_col, pred_col):
    """Aggregiert Zell-Vorhersagen zu einer Vorhersage pro Gruppe (Probe) per Mehrheitsentscheid."""
    return df.groupby(group_col, observed=True)[pred_col].agg(lambda s: s.value_counts().idxmax())


# ── 1. Daten laden (kein Pseudobulking, Zell-Ebene) ──────────────────────────

print("=== Daten laden ===")
adata = sc.read_h5ad(args.data_h5ad)
print(f"{adata.n_obs} Zellen, X_drvi shape: {adata.obsm['X_drvi'].shape}")

factors = adata.uns["X_drvi_active_dims"]
sample_id = adata.obs["sample_id"].astype(str)
timepoint = sample_id.str.split(".", n=1).str[1]  # "m6.1" -> "1"

cell_df = pd.DataFrame(adata.obsm["X_drvi"], columns=factors, index=adata.obs_names)
cell_df["sample_id"] = sample_id.values
cell_df["timepoint"] = timepoint.values
cell_df["holdout"] = adata.obs["holdout_timepoint"].astype(str).values
cell_df["cv_fold"] = adata.obs["cv_fold_timepoint"].values

n_before_cells = cell_df.shape[0]
cell_df = cell_df[cell_df["holdout"] != "excluded"].copy()
print(f"{n_before_cells - cell_df.shape[0]} Zellen ausgeschlossen (nicht ACS_sterile)")
print(f"\n{cell_df.shape[0]} Zellen x {len(factors)} Faktoren "
      f"({cell_df['sample_id'].nunique()} Proben, kein Pseudobulking)")
print(cell_df.drop_duplicates("sample_id")["timepoint"].value_counts().sort_index())

# ── 2. 5-fach CV (Fold-Zuordnung aus cv_fold_timepoint uebernommen) ──────────

print("\n=== 5-fach Cross-Validation (CV-Pool) ===")
cv_mask = cell_df["holdout"] == "cv"
cv_df = cell_df[cv_mask]
n_folds = int(cv_df["cv_fold"].max()) + 1
n_samples_cv = cv_df.drop_duplicates("sample_id")
print(f"CV-Pool: {cv_df.shape[0]} Zellen / {n_samples_cv.shape[0]} Proben, {n_folds} Folds")
print(n_samples_cv.groupby("cv_fold")["timepoint"].value_counts().unstack())

fold_rows = []
for fold in range(n_folds):
    train_mask = cv_mask & (cell_df["cv_fold"] != fold)
    val_mask = cv_mask & (cell_df["cv_fold"] == fold)

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf, class_weight="balanced",
        random_state=args.random_state, n_jobs=-1,
    )
    clf.fit(cell_df.loc[train_mask, factors], cell_df.loc[train_mask, "timepoint"])
    y_pred_cell = clf.predict(cell_df.loc[val_mask, factors])
    y_true_cell = cell_df.loc[val_mask, "timepoint"]

    acc_cell = accuracy_score(y_true_cell, y_pred_cell)
    f1_cell = f1_score(y_true_cell, y_pred_cell, average="macro", zero_division=0)

    # Mehrheitsentscheid pro Probe (Zell-Vorhersagen -> Proben-Vorhersage)
    val_df = cell_df.loc[val_mask, ["sample_id", "timepoint"]].copy()
    val_df["pred"] = y_pred_cell
    sample_pred = majority_vote(val_df, "sample_id", "pred")
    sample_true = val_df.drop_duplicates("sample_id").set_index("sample_id")["timepoint"].reindex(sample_pred.index)

    acc = accuracy_score(sample_true, sample_pred)
    f1_macro = f1_score(sample_true, sample_pred, average="macro", zero_division=0)

    n_val_samples = sample_pred.shape[0]
    fold_rows.append({
        "fold": fold, "n_train_cells": int(train_mask.sum()), "n_val_cells": int(val_mask.sum()),
        "n_val_samples": n_val_samples,
        "accuracy": acc, "f1_macro": f1_macro,
        "accuracy_cell": acc_cell, "f1_macro_cell": f1_cell,
    })
    print(f"  Fold {fold}: n_train_cells={int(train_mask.sum())}, n_val_cells={int(val_mask.sum())}, "
          f"n_val_samples={n_val_samples}, "
          f"accuracy(sample)={acc:.3f}, f1_macro(sample)={f1_macro:.3f}, "
          f"accuracy(cell)={acc_cell:.3f}, f1_macro(cell)={f1_cell:.3f}")

fold_metrics = pd.DataFrame(fold_rows)
fold_metrics.to_csv(os.path.join(args.output_dir, "rf_timepoint_cv_fold_metrics.csv"), index=False)
print(f"\nCV-Mittelwert (Proben-Ebene, Mehrheitsentscheid): accuracy={fold_metrics['accuracy'].mean():.3f} "
      f"(+/-{fold_metrics['accuracy'].std():.3f}), "
      f"f1_macro={fold_metrics['f1_macro'].mean():.3f} (+/-{fold_metrics['f1_macro'].std():.3f})")
print(f"CV-Mittelwert (Zell-Ebene): accuracy={fold_metrics['accuracy_cell'].mean():.3f} "
      f"(+/-{fold_metrics['accuracy_cell'].std():.3f}), "
      f"f1_macro={fold_metrics['f1_macro_cell'].mean():.3f} (+/-{fold_metrics['f1_macro_cell'].std():.3f})")
print("Gespeichert: rf_timepoint_cv_fold_metrics.csv")

# ── 3. Finales Modell (ganzer CV-Pool) + Test-Holdout ────────────────────────

print("\n=== Finales Modell + Test-Holdout ===")
test_mask = cell_df["holdout"] == "test"
test_df_samples = cell_df.loc[test_mask].drop_duplicates("sample_id")
print(f"CV-Pool (Training): {int(cv_mask.sum())} Zellen / {n_samples_cv.shape[0]} Proben, "
      f"Test-Holdout: {int(test_mask.sum())} Zellen / {test_df_samples.shape[0]} Proben")

final_clf = RandomForestClassifier(
    n_estimators=args.n_estimators, max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf, class_weight="balanced",
    random_state=args.random_state, n_jobs=-1,
)
final_clf.fit(cell_df.loc[cv_mask, factors], cell_df.loc[cv_mask, "timepoint"])

y_pred_test_cell = final_clf.predict(cell_df.loc[test_mask, factors])
y_true_test_cell = cell_df.loc[test_mask, "timepoint"]
acc_test_cell = accuracy_score(y_true_test_cell, y_pred_test_cell)
f1_test_cell = f1_score(y_true_test_cell, y_pred_test_cell, average="macro", zero_division=0)
print(f"Test-Holdout Zell-Ebene: accuracy={acc_test_cell:.3f}, f1_macro={f1_test_cell:.3f} "
      f"(n={int(test_mask.sum())} Zellen)")

# Mehrheitsentscheid pro Probe fuer den eigentlichen Report (Proben-Ebene,
# vergleichbar mit den bisherigen Pseudobulk-Ergebnissen)
test_df = cell_df.loc[test_mask, ["sample_id", "timepoint"]].copy()
test_df["pred"] = y_pred_test_cell
y_pred_test = majority_vote(test_df, "sample_id", "pred")
y_true_test = test_df.drop_duplicates("sample_id").set_index("sample_id")["timepoint"].reindex(y_pred_test.index)

report = classification_report(y_true_test, y_pred_test, zero_division=0)
print(report)
print("Confusion matrix (Zeilen=wahr, Spalten=vorhergesagt):")
labels = sorted(cell_df["timepoint"].unique())
cm = confusion_matrix(y_true_test, y_pred_test, labels=labels)
print(pd.DataFrame(cm, index=labels, columns=labels))

with open(os.path.join(args.output_dir, "rf_timepoint_test_holdout_report.txt"), "w") as f:
    f.write("Zielgroesse: Zeitpunkt (TP1-TP4), nur ACS_sterile\n")
    f.write(f"Test-Holdout: {y_pred_test.shape[0]} Proben "
            f"(Mehrheitsentscheid ueber {int(test_mask.sum())} Zellen, kein Pseudobulking)\n\n")
    f.write(report)
    f.write(f"\nZell-Ebene (zum Vergleich, ohne Mehrheitsentscheid): accuracy={acc_test_cell:.3f}, "
            f"f1_macro={f1_test_cell:.3f} (n={int(test_mask.sum())} Zellen)\n")
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
joblib.dump({"model": final_clf, "factors": factors, "classes": final_clf.classes_.tolist(),
            "pseudobulk": False}, model_path)
print(f"\nGespeichert: {model_path}")
