#!/usr/bin/env python3
"""
XGBoost-Klassifikator: sagt den Zeitpunkt (TP1-TP4) einer Probe aus den
aktiven DRVI-Faktoren voraus -- nur fuer ACS_sterile
(classification == "acs_w_o_infection", 19 Patienten, 62 Proben).

XGBoost-Gegenstueck zu rf_timepoint_classifier.py -- gleiche Daten, gleiches
Split-Schema, gleiche Zell-Ebene-/CV-/Holdout-Logik (kein Pseudobulking mehr,
siehe unten), nur der Classifier ist ausgetauscht (und Feature-Importance ist
SHAP-basiert statt impurity-based, wie bei xgb_condition_classifier.py).

Nutzt data_for_practicum_post_integration_curated_activedims_split.h5ad,
Spalten `holdout_timepoint`/`cv_fold_timepoint`. Das ist ein SEPARATES
Split-Schema von `holdout_stratified`/`cv_fold_stratified` (dem
Condition-Klassifikator) -- eigene Trainingsdaten, wie angefragt:
`cv_fold_timepoint` markiert CCS und non-CCS komplett als "excluded"
(nicht Teil dieser Aufgabe), waehrend `cv_fold_stratified` alle drei
Bedingungsklassen abdeckt, aber keine Zeitpunkt-Information nutzt.

KEIN Pseudobulking mehr (gleiche Begruendung wie rf_timepoint_classifier.py):
statt vorher eine Zeile pro Probe (Mittelwert der DRVI-Faktoren ueber alle
Zellen dieser Probe) zu bilden, wird jetzt JEDE EINZELNE ZELLE als eigene
Trainings-/Test-Zeile verwendet (Label = der Zeitpunkt der Probe, zu der die
Zelle gehoert). Die Split-/Fold-Zuordnung ist weiterhin Patienten-gruppiert
(StratifiedGroupKFold, beim Erstellen der Split-Spalten): alle Zellen eines
Patienten bleiben im selben Split, um Data Leakage zu vermeiden.

Weil die Bewertung pro Zelle (viele stark korrelierte Zeilen pro Probe) ein
anderes Bild liefert als die Frage "wird die Probe richtig klassifiziert",
werden Vorhersagen zusaetzlich per Mehrheitsentscheid ("majority vote") ueber
alle Zellen einer Probe zu einer Proben-Vorhersage aggregiert. Report/
Confusion-Matrix/`accuracy`+`f1_macro` in den Ergebnisdateien beziehen sich
auf diese Proben-Ebene (fuer Vergleichbarkeit mit den bisherigen
Pseudobulk-Ergebnissen); die reinen Zell-Ebene-Metriken stehen zusaetzlich
als `accuracy_cell`/`f1_macro_cell` in xgb_timepoint_cv_fold_metrics.csv und
werden auf der Konsole ausgegeben.

Ablauf:
  1. Alle Zellen der Proben im CV-Pool/Test-Holdout, keine Aggregation.
  2. 5-fach Cross-Validation innerhalb des CV-Pools (`holdout_timepoint == "cv"`),
     Fold-Zuordnung aus `cv_fold_timepoint` uebernommen. Sample-Weights
     (inverse Klassenhaeufigkeit, jetzt auf Zell-Ebene) statt sklearn's
     class_weight="balanced".
  3. Finales Modell auf allen Zellen des CV-Pools trainiert, einmalig auf dem
     nie angeruehrten Test-Holdout bewertet (Zell- und Proben-Ebene).

Ergebnis:
  - xgb_timepoint_classifier.joblib
  - xgb_timepoint_cv_fold_metrics.csv
  - xgb_timepoint_test_holdout_report.txt
  - xgb_timepoint_feature_importances.csv  (= mean |SHAP value|)
  - xgb_timepoint_shap_values.csv          (rohe mean |SHAP value| pro Klasse)

Verwendung:
    conda run -n mapra_cytokines python xgb_timepoint_classifier.py
"""

import argparse
import os

import numpy as np
import pandas as pd
import scanpy as sc
import joblib
import shap
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

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
parser.add_argument("--max-depth", type=int, default=4)
parser.add_argument("--learning-rate", type=float, default=0.05)
parser.add_argument("--random-state", type=int, default=0)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)


def majority_vote(df, group_col, pred_col):
    """Aggregiert Zell-Vorhersagen zu einer Vorhersage pro Gruppe (Probe) per Mehrheitsentscheid."""
    return df.groupby(group_col, observed=True)[pred_col].agg(lambda s: s.value_counts().idxmax())


def sample_weights(y_enc):
    """Inverse Klassenhaeufigkeit (jetzt auf Zell-Ebene) -- XGBoost's Gegenstueck zu
    class_weight='balanced'."""
    counts = np.bincount(y_enc)
    return len(y_enc) / (len(counts) * counts[y_enc])


def make_model():
    return XGBClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        learning_rate=args.learning_rate, objective="multi:softprob",
        eval_metric="mlogloss", random_state=args.random_state, n_jobs=-1,
    )


# ── 1. Daten laden (kein Pseudobulking, Zell-Ebene) ──────────────────────────

print("=== Daten laden ===")
adata = sc.read_h5ad(args.data_h5ad)
print(f"{adata.n_obs} Zellen, X_drvi shape: {adata.obsm['X_drvi'].shape}")

factors = adata.uns["X_drvi_active_dims"]
sample_id = adata.obs["sample_id"].astype(str)
timepoint = sample_id.str.split(".", n=1).str[1]  # "m6.1" -> "1"

cell_df = pd.DataFrame(adata.obsm["X_drvi"], columns=factors, index=adata.obs_names)
cell_df["sample_id"] = sample_id.values
cell_df["patient_id"] = sample_id.str.split(".", n=1).str[0].values
cell_df["timepoint"] = timepoint.values
cell_df["holdout"] = adata.obs["holdout_timepoint"].astype(str).values
cell_df["cv_fold"] = adata.obs["cv_fold_timepoint"].values

n_before_cells = cell_df.shape[0]
cell_df = cell_df[cell_df["holdout"] != "excluded"].copy()
print(f"{n_before_cells - cell_df.shape[0]} Zellen ausgeschlossen (nicht ACS_sterile)")
print(f"\n{cell_df.shape[0]} Zellen x {len(factors)} Faktoren "
      f"({cell_df['sample_id'].nunique()} Proben, kein Pseudobulking)")
print(cell_df.drop_duplicates("sample_id")["timepoint"].value_counts().sort_index())

le = LabelEncoder()
cell_df["timepoint_enc"] = le.fit_transform(cell_df["timepoint"])
print(f"Klassen-Encoding: {dict(zip(le.classes_, range(len(le.classes_))))}")

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

    y_train = cell_df.loc[train_mask, "timepoint_enc"].values
    clf = make_model()
    clf.fit(cell_df.loc[train_mask, factors], y_train, sample_weight=sample_weights(y_train))
    y_pred_cell_enc = clf.predict(cell_df.loc[val_mask, factors])
    y_pred_cell = le.inverse_transform(y_pred_cell_enc)
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
fold_metrics.to_csv(os.path.join(args.output_dir, "xgb_timepoint_cv_fold_metrics.csv"), index=False)
print(f"\nCV-Mittelwert (Proben-Ebene, Mehrheitsentscheid): accuracy={fold_metrics['accuracy'].mean():.3f} "
      f"(+/-{fold_metrics['accuracy'].std():.3f}), "
      f"f1_macro={fold_metrics['f1_macro'].mean():.3f} (+/-{fold_metrics['f1_macro'].std():.3f})")
print(f"CV-Mittelwert (Zell-Ebene): accuracy={fold_metrics['accuracy_cell'].mean():.3f} "
      f"(+/-{fold_metrics['accuracy_cell'].std():.3f}), "
      f"f1_macro={fold_metrics['f1_macro_cell'].mean():.3f} (+/-{fold_metrics['f1_macro_cell'].std():.3f})")
print("Gespeichert: xgb_timepoint_cv_fold_metrics.csv")

# ── 3. Finales Modell (ganzer CV-Pool) + Test-Holdout ────────────────────────

print("\n=== Finales Modell + Test-Holdout ===")
test_mask = cell_df["holdout"] == "test"
test_df_samples = cell_df.loc[test_mask].drop_duplicates("sample_id")
print(f"CV-Pool (Training): {int(cv_mask.sum())} Zellen / {n_samples_cv.shape[0]} Proben, "
      f"Test-Holdout: {int(test_mask.sum())} Zellen / {test_df_samples.shape[0]} Proben")

y_cv = cell_df.loc[cv_mask, "timepoint_enc"].values
final_clf = make_model()
final_clf.fit(cell_df.loc[cv_mask, factors], y_cv, sample_weight=sample_weights(y_cv))

y_pred_test_cell_enc = final_clf.predict(cell_df.loc[test_mask, factors])
y_pred_test_cell = le.inverse_transform(y_pred_test_cell_enc)
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

with open(os.path.join(args.output_dir, "xgb_timepoint_test_holdout_report.txt"), "w") as f:
    f.write("Zielgroesse: Zeitpunkt (TP1-TP4), nur ACS_sterile\n")
    f.write(f"Test-Holdout: {y_pred_test.shape[0]} Proben "
            f"(Mehrheitsentscheid ueber {int(test_mask.sum())} Zellen, kein Pseudobulking)\n\n")
    f.write(report)
    f.write(f"\nZell-Ebene (zum Vergleich, ohne Mehrheitsentscheid): accuracy={acc_test_cell:.3f}, "
            f"f1_macro={f1_test_cell:.3f} (n={int(test_mask.sum())} Zellen)\n")
    f.write("\nConfusion matrix (Zeilen=wahr, Spalten=vorhergesagt):\n")
    f.write(pd.DataFrame(cm, index=labels, columns=labels).to_string())
print("\nGespeichert: xgb_timepoint_test_holdout_report.txt")

# ── Feature Importances via SHAP (TreeExplainer, Zell-Ebene, CV-Pool) ────────

print("\n=== SHAP-Werte (TreeExplainer, CV-Pool, Zell-Ebene) ===")
explainer = shap.TreeExplainer(final_clf)
shap_values = explainer.shap_values(cell_df.loc[cv_mask, factors])
shap_values = np.asarray(shap_values)
if shap_values.ndim == 3:
    shap_values = np.moveaxis(shap_values, -1, 0)
mean_abs_per_class = np.stack([np.abs(sv).mean(axis=0) for sv in shap_values])

shap_df = pd.DataFrame(mean_abs_per_class.T, index=factors, columns=le.classes_)
shap_df["mean_abs_shap"] = shap_df.mean(axis=1)
shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)
shap_df.to_csv(os.path.join(args.output_dir, "xgb_timepoint_shap_values.csv"))
print("Gespeichert: xgb_timepoint_shap_values.csv")

importances = shap_df["mean_abs_shap"]
importances.to_csv(os.path.join(args.output_dir, "xgb_timepoint_feature_importances.csv"), header=["importance"])
print("\nTop-10 wichtigste Faktoren (mean |SHAP value|):")
print(importances.head(10))
print("Gespeichert: xgb_timepoint_feature_importances.csv")

model_path = os.path.join(args.output_dir, "xgb_timepoint_classifier.joblib")
joblib.dump({"model": final_clf, "factors": factors, "label_encoder": le,
            "classes": le.classes_.tolist(), "pseudobulk": False}, model_path)
print(f"\nGespeichert: {model_path}")
