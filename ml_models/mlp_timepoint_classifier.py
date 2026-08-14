#!/usr/bin/env python3
"""
MLP-Klassifikator: sagt den Zeitpunkt (TP1-TP4) einer Probe aus den
gemittelten aktiven DRVI-Faktoren voraus -- nur fuer ACS_sterile
(classification == "acs_w_o_infection", 19 Patienten, 62 Proben).

MLP-Gegenstueck zu rf_timepoint_classifier.py und xgb_timepoint_classifier.py
-- gleiche Daten, gleiches Split-Schema, gleiche Pseudobulk-/CV-/
Holdout-Logik, nur der Classifier ist ausgetauscht.

Nutzt data_for_practicum_post_integration_curated_activedims_split.h5ad,
Spalten `holdout_timepoint`/`cv_fold_timepoint`.

Zwei Unterschiede zu RF/XGBoost, technisch bedingt durch den MLP:
  1. Feature-Skalierung: MLPs sind (anders als Baum-Modelle) empfindlich
     gegenueber der Skala der Eingabe-Features -- StandardScaler wird pro
     Trainings-Fold NEU gefittet (kein Leakage aus Val/Test).
  2. Klassen-Balance: sklearn's MLPClassifier unterstuetzt kein
     sample_weight/class_weight (anders als RandomForest/XGBoost). Statt
     inverser Klassenhaeufigkeit wird daher im Training per Oversampling
     (Ziehen mit Zuruecklegen bis alle Klassen gleich gross sind)
     balanciert.
  3. Feature Importance: kein eingebautes Aequivalent zu
     feature_importances_/SHAP-TreeExplainer -- stattdessen
     Permutation-Importance (model-agnostisch, angemessen fuer neuronale
     Netze) auf dem CV-Pool.
  4. Label-Encoding: MLPClassifier(early_stopping=True) ruft intern
     predict() auf und prueft das Ergebnis mit np.isnan() -- das crasht mit
     String-Klassenlabels ("1".."4"). Deshalb LabelEncoder wie im
     XGBoost-Skript; Reports/Confusion-Matrix werden wieder in
     String-Labels zurueckdecodiert.

Ablauf:
  1. Pseudobulk auf Proben-Ebene: Mittelwert der aktiven DRVI-Faktoren ueber
     alle Zellen EINER Probe.
  2. 5-fach Cross-Validation innerhalb des CV-Pools (`holdout_timepoint == "cv"`),
     Fold-Zuordnung aus `cv_fold_timepoint` uebernommen.
  3. Finales Modell auf dem gesamten CV-Pool trainiert, einmalig auf dem nie
     angeruehrten Test-Holdout (`holdout_timepoint == "test"`) bewertet.

Ergebnis:
  - mlp_timepoint_classifier.joblib
  - mlp_timepoint_cv_fold_metrics.csv
  - mlp_timepoint_test_holdout_report.txt
  - mlp_timepoint_feature_importances.csv  (Permutation-Importance)

Verwendung:
    conda run -n mapra_cytokines python mlp_timepoint_classifier.py
"""

import argparse
import os

import numpy as np
import pandas as pd
import scanpy as sc
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils import resample

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-h5ad",
    default="/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/data/data_for_practicum_post_integration_curated_activedims_split.h5ad",
)
parser.add_argument(
    "--output-dir",
    default="/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/ml_models",
)
parser.add_argument("--hidden-layer-sizes", default="32,16",
                     help="Kommagetrennt, z.B. '32,16' fuer zwei Hidden Layer")
parser.add_argument("--alpha", type=float, default=1e-2, help="L2-Regularisierung")
parser.add_argument("--learning-rate-init", type=float, default=1e-3)
parser.add_argument("--max-iter", type=int, default=2000)
parser.add_argument("--random-state", type=int, default=0)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
hidden_layer_sizes = tuple(int(x) for x in args.hidden_layer_sizes.split(","))

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

# MLPClassifier(early_stopping=True) ruft intern predict() waehrend des
# Trainings auf und prueft das Ergebnis mit np.isnan() -- das crasht mit
# String-Klassenlabels ("1","2","3","4"). Deshalb hier (wie im
# XGBoost-Skript) auf Integer-codierte Labels umsteigen; fuer die
# Reports/Confusion-Matrix am Ende wieder zurueck in Strings decodieren.
le = LabelEncoder()
sample_df["timepoint_enc"] = le.fit_transform(sample_df["timepoint"])
print(f"Klassen-Encoding: {dict(zip(le.classes_, range(len(le.classes_))))}")


def oversample_balanced(X, y, random_state):
    """Ziehen mit Zuruecklegen pro Klasse bis alle Klassen so gross sind wie
    die groesste -- Ersatz fuer class_weight, das MLPClassifier nicht
    unterstuetzt."""
    max_n = y.value_counts().max()
    parts_X, parts_y = [], []
    for cls in y.unique():
        idx = y[y == cls].index
        X_cls, y_cls = resample(X.loc[idx], y.loc[idx], replace=True, n_samples=max_n,
                                 random_state=random_state)
        parts_X.append(X_cls)
        parts_y.append(y_cls)
    return pd.concat(parts_X), pd.concat(parts_y)


def make_model():
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes, alpha=args.alpha,
        learning_rate_init=args.learning_rate_init, max_iter=args.max_iter,
        early_stopping=True, n_iter_no_change=20, validation_fraction=0.15,
        random_state=args.random_state,
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

    X_train_raw = sample_df.loc[train_mask, factors]
    y_train_raw = sample_df.loc[train_mask, "timepoint_enc"]
    X_train_bal, y_train_bal = oversample_balanced(X_train_raw, y_train_raw, args.random_state)

    scaler = StandardScaler().fit(X_train_bal)
    X_train_scaled = scaler.transform(X_train_bal)
    X_val_scaled = scaler.transform(sample_df.loc[val_mask, factors])

    clf = make_model()
    clf.fit(X_train_scaled, y_train_bal)
    y_pred = clf.predict(X_val_scaled)
    y_true = sample_df.loc[val_mask, "timepoint_enc"]

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    fold_rows.append({"fold": fold, "n_train": int(train_mask.sum()), "n_val": int(val_mask.sum()),
                      "accuracy": acc, "f1_macro": f1_macro})
    print(f"  Fold {fold}: n_train={int(train_mask.sum())}, n_val={int(val_mask.sum())}, "
          f"accuracy={acc:.3f}, f1_macro={f1_macro:.3f}")

fold_metrics = pd.DataFrame(fold_rows)
fold_metrics.to_csv(os.path.join(args.output_dir, "mlp_timepoint_cv_fold_metrics.csv"), index=False)
print(f"\nCV-Mittelwert: accuracy={fold_metrics['accuracy'].mean():.3f} "
      f"(+/-{fold_metrics['accuracy'].std():.3f}), "
      f"f1_macro={fold_metrics['f1_macro'].mean():.3f} (+/-{fold_metrics['f1_macro'].std():.3f})")
print("Gespeichert: mlp_timepoint_cv_fold_metrics.csv")

# ── 3. Finales Modell (ganzer CV-Pool) + Test-Holdout ────────────────────────

print("\n=== Finales Modell + Test-Holdout ===")
test_mask = sample_df["holdout"] == "test"
print(f"CV-Pool (Training): {int(cv_mask.sum())} Proben, Test-Holdout: {int(test_mask.sum())} Proben")

X_cv_raw = sample_df.loc[cv_mask, factors]
y_cv_raw = sample_df.loc[cv_mask, "timepoint_enc"]
X_cv_bal, y_cv_bal = oversample_balanced(X_cv_raw, y_cv_raw, args.random_state)

final_scaler = StandardScaler().fit(X_cv_bal)
X_cv_scaled = final_scaler.transform(X_cv_bal)

final_clf = make_model()
final_clf.fit(X_cv_scaled, y_cv_bal)

X_test_scaled = final_scaler.transform(sample_df.loc[test_mask, factors])
y_pred_test_enc = final_clf.predict(X_test_scaled)
y_pred_test = le.inverse_transform(y_pred_test_enc)
y_true_test = sample_df.loc[test_mask, "timepoint"]

report = classification_report(y_true_test, y_pred_test, zero_division=0)
print(report)
print("Confusion matrix (Zeilen=wahr, Spalten=vorhergesagt):")
labels = sorted(sample_df["timepoint"].unique())
cm = confusion_matrix(y_true_test, y_pred_test, labels=labels)
print(pd.DataFrame(cm, index=labels, columns=labels))

with open(os.path.join(args.output_dir, "mlp_timepoint_test_holdout_report.txt"), "w") as f:
    f.write("Zielgroesse: Zeitpunkt (TP1-TP4), nur ACS_sterile\n")
    f.write(f"Test-Holdout: {int(test_mask.sum())} Proben\n\n")
    f.write(report)
    f.write("\nConfusion matrix (Zeilen=wahr, Spalten=vorhergesagt):\n")
    f.write(pd.DataFrame(cm, index=labels, columns=labels).to_string())
print("\nGespeichert: mlp_timepoint_test_holdout_report.txt")

# ── Feature Importances via Permutation-Importance (CV-Pool) ────────────────

print("\n=== Permutation-Importance (CV-Pool) ===")
perm = permutation_importance(final_clf, X_cv_scaled, y_cv_bal, n_repeats=30,
                               random_state=args.random_state, scoring="f1_macro", n_jobs=-1)
importances = pd.Series(perm.importances_mean, index=factors).sort_values(ascending=False)
importances.to_csv(os.path.join(args.output_dir, "mlp_timepoint_feature_importances.csv"), header=["importance"])
print("\nTop-10 wichtigste Faktoren (Permutation-Importance, f1_macro-Abfall):")
print(importances.head(10))
print("Gespeichert: mlp_timepoint_feature_importances.csv")

model_path = os.path.join(args.output_dir, "mlp_timepoint_classifier.joblib")
joblib.dump({"model": final_clf, "scaler": final_scaler, "label_encoder": le, "factors": factors,
            "classes": le.inverse_transform(final_clf.classes_).tolist()}, model_path)
print(f"\nGespeichert: {model_path}")