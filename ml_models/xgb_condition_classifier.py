#!/usr/bin/env python3
"""
XGBoost-Klassifikator: sagt die Bedingung (ACS / CCS / non-CCS) eines
Patienten aus den aktiven DRVI-Faktoren voraus.

XGBoost-Gegenstueck zu rf_condition_classifier.py -- gleiche Daten, gleiches
Split-Schema, gleiche Zell-Ebene-/CV-/Holdout-Logik (kein Pseudobulking mehr,
siehe unten), nur der Classifier ist ausgetauscht.

WICHTIG zum Split-Schema: `cv_fold_timepoint`/`holdout_timepoint` (wie urspruenglich
angefragt) funktioniert hier NICHT -- das ist das Schema von
rf_timepoint_classifier.py und deckt AUSSCHLIESSLICH ACS_sterile ab (CCS und
non-CCS sind dort zu 100% "excluded", nicht nur "alle anderen Klassen" wie
acs_w_infection/acs_subacute/koronarsklerose). Verifiziert per Crosstab
gegen `classification`:
    cv_fold_timepoint:   acs_w_o_infection -> cv/test, ALLE anderen -> excluded
                          (auch ccs und vollstaendiger_ausschluss!)
    cv_fold_stratified:  acs_w_o_infection, ccs, vollstaendiger_ausschluss -> cv/test,
                          acs_w_infection/acs_subacute/koronarsklerose -> excluded
Nur `stratified` deckt alle drei Zielklassen ab -- das wird hier verwendet.

Nutzt data_for_practicum_post_integration_curated_activedims_split.h5ad.
Bedingungs-Mapping eingeschraenkt (gleich wie rf_condition_classifier.py):
ACS = nur "acs_w_o_infection" (19 Patienten), CCS = "ccs" (16), non-CCS =
nur "vollstaendiger_ausschluss" (10). "acs_w_infection", "acs_subacute" und
"koronarsklerose" sind komplett ausgeschlossen (holdout_stratified=="excluded").

KEIN Pseudobulking mehr (gleiche Begruendung wie rf_condition_classifier.py):
statt vorher eine Zeile pro Patient (Mittelwert der DRVI-Faktoren ueber alle
seine Zellen) zu bilden, wird jetzt JEDE EINZELNE ZELLE als eigene Trainings-/
Test-Zeile verwendet (Label = die Bedingung des Patienten, dem die Zelle
gehoert). Die Split-/Fold-Zuordnung bleibt weiterhin auf Patienten-Ebene (aus
`holdout_stratified`/`cv_fold_stratified`, per Zelle identisch fuer alle
Zellen desselben Patienten) -- kein Patient landet je in Training UND
Validation/Test, also kein Data Leakage.

Weil die Bewertung pro Zelle (viele stark korrelierte Zeilen pro Patient) ein
anderes Bild liefert als die klinisch relevante Frage "wird der Patient
richtig klassifiziert", werden Vorhersagen zusaetzlich per Mehrheitsentscheid
("majority vote") ueber alle Zellen eines Patienten zu einer
Patienten-Vorhersage aggregiert. Report/Confusion-Matrix/`accuracy`+`f1_macro`
in den Ergebnisdateien beziehen sich auf diese Patienten-Ebene (fuer
Vergleichbarkeit mit den bisherigen Pseudobulk-Ergebnissen); die reinen
Zell-Ebene-Metriken stehen zusaetzlich als `accuracy_cell`/`f1_macro_cell`
in xgb_cv_fold_metrics.csv und werden auf der Konsole ausgegeben.

Ablauf:
  1. Alle Zellen der Patienten im CV-Pool/Test-Holdout, keine Aggregation.
  2. 5-fach Cross-Validation innerhalb des CV-Pools (`holdout_stratified == "cv"`),
     Fold-Zuordnung aus `cv_fold_stratified` uebernommen. Sample-Weights
     (inverse Klassenhaeufigkeit, jetzt auf Zell-Ebene) statt sklearn's
     class_weight="balanced", das XGBoost nicht kennt.
  3. Finales Modell auf allen Zellen des CV-Pools trainiert, einmalig auf dem
     nie angeruehrten Test-Holdout bewertet (Zell- und Patienten-Ebene).

Feature-Importance wird per SHAP (TreeExplainer) berechnet, nicht per
XGBoost's eingebautem `feature_importances_` (gain-based) -- wie angefragt.
Jetzt auf Zell-Ebene (CV-Pool-Zellen statt Pseudobulk-Patientenzeilen), mean
|SHAP value|, gemittelt zusaetzlich ueber die drei Klassen (SHAP liefert bei
multi:softprob einen Werte-Satz pro Klasse).

Ergebnis:
  - xgb_condition_classifier.joblib
  - xgb_cv_fold_metrics.csv
  - xgb_test_holdout_report.txt
  - xgb_feature_importances.csv   (= mean |SHAP value|, Spaltenname bleibt
    "importance" fuer Kompatibilitaet mit plot_xgb_results.py)
  - xgb_shap_values.csv           (rohe mean |SHAP value| pro Klasse + Mittel)

Verwendung:
    conda run -n mapra_cytokines python xgb_condition_classifier.py
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
parser.add_argument("--split", default="stratified",
                    help="Split-Schema (Spalten holdout_<schema>/cv_fold_<schema>). "
                         "'timepoint' funktioniert NICHT fuer diese Aufgabe -- siehe Docstring.")
parser.add_argument("--n-estimators", type=int, default=500)
parser.add_argument("--max-depth", type=int, default=4)
parser.add_argument("--learning-rate", type=float, default=0.05)
parser.add_argument("--random-state", type=int, default=0)
args = parser.parse_args()

if args.split == "timepoint":
    raise SystemExit(
        "cv_fold_timepoint/holdout_timepoint excludes ALL CCS and non-CCS cells "
        "(that schema is for predicting timepoint within ACS_sterile only, see "
        "rf_timepoint_classifier.py) -- it cannot be used for a 3-class ACS/CCS/non-CCS "
        "condition model. Use --split stratified instead."
    )

# Gleiches (eingeschraenktes) Mapping wie rf_condition_classifier.py.
CONDITION_MAP = {
    "acs_w_o_infection": "ACS",
    "ccs": "CCS",
    "vollstaendiger_ausschluss": "non-CCS",
}

os.makedirs(args.output_dir, exist_ok=True)
holdout_col = f"holdout_{args.split}"
fold_col = f"cv_fold_{args.split}"
suffix = f"_{args.split}"


def majority_vote(df, group_col, pred_col):
    """Aggregiert Zell-Vorhersagen zu einer Vorhersage pro Gruppe (Patient) per Mehrheitsentscheid."""
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

print(f"=== Daten laden (Split-Schema: {args.split}) ===")
adata = sc.read_h5ad(args.data_h5ad)
print(f"{adata.n_obs} Zellen, X_drvi shape: {adata.obsm['X_drvi'].shape}")

factors = adata.uns["X_drvi_active_dims"]
patient_id = adata.obs["sample_id"].astype(str).str.split(".", n=1).str[0]

cell_df = pd.DataFrame(adata.obsm["X_drvi"], columns=factors, index=adata.obs_names)
cell_df["patient_id"] = patient_id.values
cell_df["condition"] = adata.obs["classification"].astype(str).map(CONDITION_MAP).values
cell_df["holdout"] = adata.obs[holdout_col].astype(str).values
cell_df["cv_fold"] = adata.obs[fold_col].values

n_before = cell_df.shape[0]
cell_df = cell_df[cell_df["holdout"] != "excluded"].copy()
print(f"\n{n_before} Zellen insgesamt, {n_before - cell_df.shape[0]} ausgeschlossen "
      f"(classification ausserhalb ACS/CCS/non-CCS-Mapping)")
print(f"{cell_df.shape[0]} Zellen x {len(factors)} Faktoren "
      f"({cell_df['patient_id'].nunique()} Patienten, kein Pseudobulking)")
print(cell_df["condition"].value_counts())

le = LabelEncoder()
cell_df["condition_enc"] = le.fit_transform(cell_df["condition"])
print(f"Klassen-Encoding: {dict(zip(le.classes_, range(len(le.classes_))))}")

# ── 2. 5-fach CV (Fold-Zuordnung aus cv_fold_<schema> uebernommen) ───────────

print(f"\n=== 5-fach Cross-Validation (CV-Pool, Schema={args.split}) ===")
cv_mask = cell_df["holdout"] == "cv"
cv_df = cell_df[cv_mask]
n_folds = int(cv_df["cv_fold"].max()) + 1
n_patients_cv = cv_df.drop_duplicates("patient_id")
print(f"CV-Pool: {cv_df.shape[0]} Zellen / {n_patients_cv.shape[0]} Patienten, {n_folds} Folds")
print(n_patients_cv.groupby("cv_fold")["condition"].value_counts().unstack())

fold_rows = []
for fold in range(n_folds):
    train_mask = cv_mask & (cell_df["cv_fold"] != fold)
    val_mask = cv_mask & (cell_df["cv_fold"] == fold)

    y_train = cell_df.loc[train_mask, "condition_enc"].values
    clf = make_model()
    clf.fit(cell_df.loc[train_mask, factors], y_train, sample_weight=sample_weights(y_train))
    y_pred_cell_enc = clf.predict(cell_df.loc[val_mask, factors])
    y_pred_cell = le.inverse_transform(y_pred_cell_enc)
    y_true_cell = cell_df.loc[val_mask, "condition"]

    acc_cell = accuracy_score(y_true_cell, y_pred_cell)
    f1_cell = f1_score(y_true_cell, y_pred_cell, average="macro", zero_division=0)

    # Mehrheitsentscheid pro Patient (Zell-Vorhersagen -> Patienten-Vorhersage)
    val_df = cell_df.loc[val_mask, ["patient_id", "condition"]].copy()
    val_df["pred"] = y_pred_cell
    patient_pred = majority_vote(val_df, "patient_id", "pred")
    patient_true = val_df.drop_duplicates("patient_id").set_index("patient_id")["condition"].reindex(patient_pred.index)

    acc = accuracy_score(patient_true, patient_pred)
    f1_macro = f1_score(patient_true, patient_pred, average="macro", zero_division=0)

    n_val_patients = patient_pred.shape[0]
    fold_rows.append({
        "fold": fold, "n_train_cells": int(train_mask.sum()), "n_val_cells": int(val_mask.sum()),
        "n_val_patients": n_val_patients,
        "accuracy": acc, "f1_macro": f1_macro,
        "accuracy_cell": acc_cell, "f1_macro_cell": f1_cell,
    })
    print(f"  Fold {fold}: n_train_cells={int(train_mask.sum())}, n_val_cells={int(val_mask.sum())}, "
          f"n_val_patients={n_val_patients}, "
          f"accuracy(patient)={acc:.3f}, f1_macro(patient)={f1_macro:.3f}, "
          f"accuracy(cell)={acc_cell:.3f}, f1_macro(cell)={f1_cell:.3f}")

fold_metrics = pd.DataFrame(fold_rows)
fold_metrics.to_csv(os.path.join(args.output_dir, f"xgb_cv_fold_metrics{suffix}.csv"), index=False)
print(f"\nCV-Mittelwert (Patienten-Ebene, Mehrheitsentscheid): accuracy={fold_metrics['accuracy'].mean():.3f} "
      f"(+/-{fold_metrics['accuracy'].std():.3f}), "
      f"f1_macro={fold_metrics['f1_macro'].mean():.3f} (+/-{fold_metrics['f1_macro'].std():.3f})")
print(f"CV-Mittelwert (Zell-Ebene): accuracy={fold_metrics['accuracy_cell'].mean():.3f} "
      f"(+/-{fold_metrics['accuracy_cell'].std():.3f}), "
      f"f1_macro={fold_metrics['f1_macro_cell'].mean():.3f} (+/-{fold_metrics['f1_macro_cell'].std():.3f})")
print(f"Gespeichert: xgb_cv_fold_metrics{suffix}.csv")

# ── 3. Finales Modell (ganzer CV-Pool) + Test-Holdout ────────────────────────

print("\n=== Finales Modell + Test-Holdout ===")
test_mask = cell_df["holdout"] == "test"
test_df_patients = cell_df.loc[test_mask].drop_duplicates("patient_id")
print(f"CV-Pool (Training): {int(cv_mask.sum())} Zellen / {n_patients_cv.shape[0]} Patienten, "
      f"Test-Holdout: {int(test_mask.sum())} Zellen / {test_df_patients.shape[0]} Patienten")

y_cv = cell_df.loc[cv_mask, "condition_enc"].values
final_clf = make_model()
final_clf.fit(cell_df.loc[cv_mask, factors], y_cv, sample_weight=sample_weights(y_cv))

y_pred_test_cell_enc = final_clf.predict(cell_df.loc[test_mask, factors])
y_pred_test_cell = le.inverse_transform(y_pred_test_cell_enc)
y_true_test_cell = cell_df.loc[test_mask, "condition"]
acc_test_cell = accuracy_score(y_true_test_cell, y_pred_test_cell)
f1_test_cell = f1_score(y_true_test_cell, y_pred_test_cell, average="macro", zero_division=0)
print(f"Test-Holdout Zell-Ebene: accuracy={acc_test_cell:.3f}, f1_macro={f1_test_cell:.3f} "
      f"(n={int(test_mask.sum())} Zellen)")

# Mehrheitsentscheid pro Patient fuer den eigentlichen Report (Patienten-Ebene,
# vergleichbar mit den bisherigen Pseudobulk-Ergebnissen)
test_df = cell_df.loc[test_mask, ["patient_id", "condition"]].copy()
test_df["pred"] = y_pred_test_cell
y_pred_test = majority_vote(test_df, "patient_id", "pred")
y_true_test = test_df.drop_duplicates("patient_id").set_index("patient_id")["condition"].reindex(y_pred_test.index)

report = classification_report(y_true_test, y_pred_test, zero_division=0)
print(report)
print("Confusion matrix (Zeilen=wahr, Spalten=vorhergesagt):")
labels = sorted(cell_df["condition"].unique())
cm = confusion_matrix(y_true_test, y_pred_test, labels=labels)
print(pd.DataFrame(cm, index=labels, columns=labels))

with open(os.path.join(args.output_dir, f"xgb_test_holdout_report{suffix}.txt"), "w") as f:
    f.write(f"Split-Schema: {args.split}\n")
    f.write(f"Test-Holdout: {y_pred_test.shape[0]} Patienten "
            f"(Mehrheitsentscheid ueber {int(test_mask.sum())} Zellen, kein Pseudobulking)\n\n")
    f.write(report)
    f.write(f"\nZell-Ebene (zum Vergleich, ohne Mehrheitsentscheid): accuracy={acc_test_cell:.3f}, "
            f"f1_macro={f1_test_cell:.3f} (n={int(test_mask.sum())} Zellen)\n")
    f.write("\nConfusion matrix (Zeilen=wahr, Spalten=vorhergesagt):\n")
    f.write(pd.DataFrame(cm, index=labels, columns=labels).to_string())
print(f"\nGespeichert: xgb_test_holdout_report{suffix}.txt")

# ── Feature Importances via SHAP (TreeExplainer, Zell-Ebene, CV-Pool) ────────

print("\n=== SHAP-Werte (TreeExplainer, CV-Pool, Zell-Ebene) ===")
explainer = shap.TreeExplainer(final_clf)
shap_values = explainer.shap_values(cell_df.loc[cv_mask, factors])
# multi:softprob -> eine (n_samples, n_features) Matrix pro Klasse, als Liste
# ODER ein (n_samples, n_features, n_classes) Array, je nach shap-Version.
shap_values = np.asarray(shap_values)
if shap_values.ndim == 3:
    # (n_samples, n_features, n_classes) -> (n_classes, n_samples, n_features)
    shap_values = np.moveaxis(shap_values, -1, 0)
mean_abs_per_class = np.stack([np.abs(sv).mean(axis=0) for sv in shap_values])  # (n_classes, n_features)

shap_df = pd.DataFrame(mean_abs_per_class.T, index=factors, columns=le.classes_)
shap_df["mean_abs_shap"] = shap_df.mean(axis=1)
shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)
shap_df.to_csv(os.path.join(args.output_dir, f"xgb_shap_values{suffix}.csv"))
print(f"Gespeichert: xgb_shap_values{suffix}.csv")

# gleicher Dateiname/Spaltenname wie zuvor (importance), jetzt SHAP-basiert,
# damit plot_xgb_results.py unveraendert bleibt.
importances = shap_df["mean_abs_shap"]
importances.to_csv(os.path.join(args.output_dir, f"xgb_feature_importances{suffix}.csv"), header=["importance"])
print("\nTop-10 wichtigste Faktoren (mean |SHAP value|):")
print(importances.head(10))
print(f"Gespeichert: xgb_feature_importances{suffix}.csv")

model_path = os.path.join(args.output_dir, f"xgb_condition_classifier{suffix}.joblib")
joblib.dump({"model": final_clf, "factors": factors, "label_encoder": le,
            "classes": le.classes_.tolist(), "split_scheme": args.split, "pseudobulk": False}, model_path)
print(f"\nGespeichert: {model_path}")
