#!/usr/bin/env python3
"""
XGBoost-Klassifikator: sagt den Zeitpunkt (TP1-TP4) einer Probe aus den
gemittelten aktiven DRVI-Faktoren voraus -- nur fuer ACS_sterile
(classification == "acs_w_o_infection", 19 Patienten, 62 Proben).

XGBoost-Gegenstueck zu rf_timepoint_classifier.py -- gleiche Daten, gleiches
Split-Schema, gleiche Pseudobulk-/CV-/Holdout-Logik, nur der Classifier ist
ausgetauscht (und Feature-Importance ist SHAP-basiert statt gain-based, wie
bei xgb_condition_classifier.py).

Nutzt data_for_practicum_post_integration_curated_activedims_split.h5ad,
Spalten `holdout_timepoint`/`cv_fold_timepoint`. Das ist ein SEPARATES
Split-Schema von `holdout_stratified`/`cv_fold_stratified` (dem
Condition-Klassifikator) -- eigene Trainingsdaten, wie angefragt:
`cv_fold_timepoint` markiert CCS und non-CCS komplett als "excluded"
(nicht Teil dieser Aufgabe), waehrend `cv_fold_stratified` alle drei
Bedingungsklassen abdeckt, aber keine Zeitpunkt-Information nutzt.

Anders als xgb_condition_classifier.py wird hier auf **Proben-Ebene**
(Patient.Zeitpunkt, z.B. "m6.1") pseudobulked, NICHT auf Patienten-Ebene --
der Zeitpunkt ist ja gerade die Zielgroesse. Die Split-/Fold-Zuordnung ist
trotzdem Patienten-gruppiert (StratifiedGroupKFold, beim Erstellen der
Split-Spalten): alle Proben eines Patienten bleiben im selben Split, um
Data Leakage zu vermeiden.

Ablauf:
  1. Pseudobulk auf Proben-Ebene: Mittelwert der aktiven DRVI-Faktoren ueber
     alle Zellen EINER Probe (nicht ueber alle Zeitpunkte eines Patienten).
  2. 5-fach Cross-Validation innerhalb des CV-Pools (`holdout_timepoint == "cv"`),
     Fold-Zuordnung aus `cv_fold_timepoint` uebernommen. Sample-Weights
     (inverse Klassenhaeufigkeit) statt sklearn's class_weight="balanced".
  3. Finales Modell auf dem gesamten CV-Pool trainiert, einmalig auf dem nie
     angeruehrten Test-Holdout (`holdout_timepoint == "test"`) bewertet.

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

le = LabelEncoder()
sample_df["timepoint_enc"] = le.fit_transform(sample_df["timepoint"])
print(f"Klassen-Encoding: {dict(zip(le.classes_, range(len(le.classes_))))}")


def sample_weights(y_enc):
    """Inverse Klassenhaeufigkeit -- XGBoost's Gegenstueck zu class_weight='balanced'."""
    counts = np.bincount(y_enc)
    return len(y_enc) / (len(counts) * counts[y_enc])


def make_model():
    return XGBClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        learning_rate=args.learning_rate, objective="multi:softprob",
        eval_metric="mlogloss", random_state=args.random_state, n_jobs=-1,
    )


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

    y_train = sample_df.loc[train_mask, "timepoint_enc"].values
    clf = make_model()
    clf.fit(sample_df.loc[train_mask, factors], y_train, sample_weight=sample_weights(y_train))
    y_pred = clf.predict(sample_df.loc[val_mask, factors])
    y_true = sample_df.loc[val_mask, "timepoint_enc"].values

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    fold_rows.append({"fold": fold, "n_train": int(train_mask.sum()), "n_val": int(val_mask.sum()),
                      "accuracy": acc, "f1_macro": f1_macro})
    print(f"  Fold {fold}: n_train={int(train_mask.sum())}, n_val={int(val_mask.sum())}, "
          f"accuracy={acc:.3f}, f1_macro={f1_macro:.3f}")

fold_metrics = pd.DataFrame(fold_rows)
fold_metrics.to_csv(os.path.join(args.output_dir, "xgb_timepoint_cv_fold_metrics.csv"), index=False)
print(f"\nCV-Mittelwert: accuracy={fold_metrics['accuracy'].mean():.3f} "
      f"(+/-{fold_metrics['accuracy'].std():.3f}), "
      f"f1_macro={fold_metrics['f1_macro'].mean():.3f} (+/-{fold_metrics['f1_macro'].std():.3f})")
print("Gespeichert: xgb_timepoint_cv_fold_metrics.csv")

# ── 3. Finales Modell (ganzer CV-Pool) + Test-Holdout ────────────────────────

print("\n=== Finales Modell + Test-Holdout ===")
test_mask = sample_df["holdout"] == "test"
print(f"CV-Pool (Training): {int(cv_mask.sum())} Proben, Test-Holdout: {int(test_mask.sum())} Proben")

y_cv = sample_df.loc[cv_mask, "timepoint_enc"].values
final_clf = make_model()
final_clf.fit(sample_df.loc[cv_mask, factors], y_cv, sample_weight=sample_weights(y_cv))

y_pred_test_enc = final_clf.predict(sample_df.loc[test_mask, factors])
y_pred_test = le.inverse_transform(y_pred_test_enc)
y_true_test = sample_df.loc[test_mask, "timepoint"]

report = classification_report(y_true_test, y_pred_test, zero_division=0)
print(report)
print("Confusion matrix (Zeilen=wahr, Spalten=vorhergesagt):")
labels = sorted(sample_df["timepoint"].unique())
cm = confusion_matrix(y_true_test, y_pred_test, labels=labels)
print(pd.DataFrame(cm, index=labels, columns=labels))

with open(os.path.join(args.output_dir, "xgb_timepoint_test_holdout_report.txt"), "w") as f:
    f.write("Zielgroesse: Zeitpunkt (TP1-TP4), nur ACS_sterile\n")
    f.write(f"Test-Holdout: {int(test_mask.sum())} Proben\n\n")
    f.write(report)
    f.write("\nConfusion matrix (Zeilen=wahr, Spalten=vorhergesagt):\n")
    f.write(pd.DataFrame(cm, index=labels, columns=labels).to_string())
print("\nGespeichert: xgb_timepoint_test_holdout_report.txt")

# ── Feature Importances via SHAP (TreeExplainer, nicht gain-based) ──────────

print("\n=== SHAP-Werte (TreeExplainer, CV-Pool) ===")
explainer = shap.TreeExplainer(final_clf)
shap_values = explainer.shap_values(sample_df.loc[cv_mask, factors])
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
            "classes": le.classes_.tolist()}, model_path)
print(f"\nGespeichert: {model_path}")
