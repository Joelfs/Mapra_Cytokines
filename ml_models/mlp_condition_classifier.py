#!/usr/bin/env python3
"""
MLP-Klassifikator: sagt die Bedingung (ACS / CCS / non-CCS) eines Patienten
aus den gemittelten aktiven DRVI-Faktoren voraus.

MLP-Gegenstueck zu rf_condition_classifier.py und xgb_condition_classifier.py
-- gleiche Daten, gleiches Split-Schema, gleiche Pseudobulk-/CV-/
Holdout-Logik, nur der Classifier ist ausgetauscht. Vier technisch bedingte
Unterschiede (dieselben wie in mlp_timepoint_classifier.py):
  1. Feature-Skalierung: StandardScaler, pro Trainings-Fold neu gefittet.
  2. Klassen-Balance: Oversampling statt class_weight/sample_weight (von
     MLPClassifier nicht unterstuetzt).
  3. Feature Importance: Permutation-Importance statt
     feature_importances_/SHAP-TreeExplainer.
  4. Label-Encoding: LabelEncoder noetig, weil MLPClassifier(early_stopping=
     True) intern predict() aufruft und mit np.isnan() prueft -- das crasht
     mit String-Klassenlabels ("ACS"/"CCS"/"non-CCS").

Nutzt data_for_practicum_post_integration_curated_activedims_split.h5ad, mit
denselben vier moeglichen Split-Schemata wie rf_condition_classifier.py
(--split stratified/balanced_322/balanced_222/balanced_432). Wie beim
XGBoost-Skript festgestellt: NUR 'stratified' (und die balanced_* Schemata)
decken alle drei Zielklassen ab -- 'timepoint' waere hier falsch (siehe
xgb_condition_classifier.py-Docstring).

Bedingungs-Mapping eingeschraenkt (identisch zu RF/XGBoost): ACS = nur
"acs_w_o_infection" (19 Patienten), CCS = "ccs" (16), non-CCS = nur
"vollstaendiger_ausschluss" (10). Alle anderen Klassifikationswerte sind in
den Split-Spalten "excluded" und werden automatisch entfernt.

Ablauf:
  1. Pseudobulk auf Patienten-Ebene: Mittelwert der aktiven DRVI-Faktoren
     ueber alle Zellen eines Patienten (alle Zeitpunkte zusammen).
  2. 5-fach Cross-Validation innerhalb des CV-Pools (`holdout_<schema> == "cv"`),
     Fold-Zuordnung aus `cv_fold_<schema>` uebernommen.
  3. Finales Modell auf dem gesamten CV-Pool trainiert, einmalig auf dem nie
     angeruehrten Test-Holdout (`holdout_<schema> == "test"`) bewertet.

Ergebnis (Dateinamen mit Schema-Suffix, gleiche Konvention wie RF/XGBoost):
  - mlp_condition_classifier_<schema>.joblib
  - mlp_cv_fold_metrics_<schema>.csv
  - mlp_test_holdout_report_<schema>.txt
  - mlp_feature_importances_<schema>.csv   (Permutation-Importance)

Verwendung:
    conda run -n mapra_cytokines python mlp_condition_classifier.py --split stratified
    conda run -n mapra_cytokines python mlp_condition_classifier.py --split balanced_322
    conda run -n mapra_cytokines python mlp_condition_classifier.py --split balanced_222
    conda run -n mapra_cytokines python mlp_condition_classifier.py --split balanced_432
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
parser.add_argument("--split", choices=["stratified", "balanced_322", "balanced_222", "balanced_432"], default="stratified",
                    help="Welches Split-Schema (Spalten holdout_<schema>/cv_fold_<schema>) verwendet wird.")
parser.add_argument("--hidden-layer-sizes", default="32,16",
                     help="Kommagetrennt, z.B. '32,16' fuer zwei Hidden Layer")
parser.add_argument("--alpha", type=float, default=1e-2, help="L2-Regularisierung")
parser.add_argument("--learning-rate-init", type=float, default=1e-3)
parser.add_argument("--max-iter", type=int, default=2000)
parser.add_argument("--random-state", type=int, default=0)
args = parser.parse_args()

# Gleiches (eingeschraenktes) Mapping wie rf_condition_classifier.py /
# xgb_condition_classifier.py.
CONDITION_MAP = {
    "acs_w_o_infection": "ACS",
    "ccs": "CCS",
    "vollstaendiger_ausschluss": "non-CCS",
}

os.makedirs(args.output_dir, exist_ok=True)
holdout_col = f"holdout_{args.split}"
fold_col = f"cv_fold_{args.split}"
suffix = f"_{args.split}"
hidden_layer_sizes = tuple(int(x) for x in args.hidden_layer_sizes.split(","))

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

# Siehe Docstring Punkt 4: LabelEncoder noetig wegen MLPClassifier(early_stopping)
# + np.isnan() auf String-Labels.
le = LabelEncoder()
patient_df["condition_enc"] = le.fit_transform(patient_df["condition"])
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

    X_train_raw = patient_df.loc[train_mask, factors]
    y_train_raw = patient_df.loc[train_mask, "condition_enc"]
    X_train_bal, y_train_bal = oversample_balanced(X_train_raw, y_train_raw, args.random_state)

    scaler = StandardScaler().fit(X_train_bal)
    X_train_scaled = scaler.transform(X_train_bal)
    X_val_scaled = scaler.transform(patient_df.loc[val_mask, factors])

    clf = make_model()
    clf.fit(X_train_scaled, y_train_bal)
    y_pred = clf.predict(X_val_scaled)
    y_true = patient_df.loc[val_mask, "condition_enc"]

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    fold_rows.append({"fold": fold, "n_train": int(train_mask.sum()), "n_val": int(val_mask.sum()),
                      "accuracy": acc, "f1_macro": f1_macro})
    print(f"  Fold {fold}: n_train={int(train_mask.sum())}, n_val={int(val_mask.sum())}, "
          f"accuracy={acc:.3f}, f1_macro={f1_macro:.3f}")

fold_metrics = pd.DataFrame(fold_rows)
fold_metrics.to_csv(os.path.join(args.output_dir, f"mlp_cv_fold_metrics{suffix}.csv"), index=False)
print(f"\nCV-Mittelwert: accuracy={fold_metrics['accuracy'].mean():.3f} "
      f"(+/-{fold_metrics['accuracy'].std():.3f}), "
      f"f1_macro={fold_metrics['f1_macro'].mean():.3f} (+/-{fold_metrics['f1_macro'].std():.3f})")
print(f"Gespeichert: mlp_cv_fold_metrics{suffix}.csv")

# ── 3. Finales Modell (ganzer CV-Pool) + Test-Holdout ────────────────────────

print("\n=== Finales Modell + Test-Holdout ===")
test_mask = patient_df["holdout"] == "test"
print(f"CV-Pool (Training): {int(cv_mask.sum())} Patienten, Test-Holdout: {int(test_mask.sum())} Patienten")

X_cv_raw = patient_df.loc[cv_mask, factors]
y_cv_raw = patient_df.loc[cv_mask, "condition_enc"]
X_cv_bal, y_cv_bal = oversample_balanced(X_cv_raw, y_cv_raw, args.random_state)

final_scaler = StandardScaler().fit(X_cv_bal)
X_cv_scaled = final_scaler.transform(X_cv_bal)

final_clf = make_model()
final_clf.fit(X_cv_scaled, y_cv_bal)

X_test_scaled = final_scaler.transform(patient_df.loc[test_mask, factors])
y_pred_test_enc = final_clf.predict(X_test_scaled)
y_pred_test = le.inverse_transform(y_pred_test_enc)
y_true_test = patient_df.loc[test_mask, "condition"]

report = classification_report(y_true_test, y_pred_test, zero_division=0)
print(report)
print("Confusion matrix (Zeilen=wahr, Spalten=vorhergesagt):")
labels = sorted(patient_df["condition"].unique())
cm = confusion_matrix(y_true_test, y_pred_test, labels=labels)
print(pd.DataFrame(cm, index=labels, columns=labels))

with open(os.path.join(args.output_dir, f"mlp_test_holdout_report{suffix}.txt"), "w") as f:
    f.write(f"Split-Schema: {args.split}\n")
    f.write(f"Test-Holdout: {int(test_mask.sum())} Patienten\n\n")
    f.write(report)
    f.write("\nConfusion matrix (Zeilen=wahr, Spalten=vorhergesagt):\n")
    f.write(pd.DataFrame(cm, index=labels, columns=labels).to_string())
print(f"\nGespeichert: mlp_test_holdout_report{suffix}.txt")

# ── Feature Importances via Permutation-Importance (CV-Pool) ────────────────

print("\n=== Permutation-Importance (CV-Pool) ===")
perm = permutation_importance(final_clf, X_cv_scaled, y_cv_bal, n_repeats=30,
                               random_state=args.random_state, scoring="f1_macro", n_jobs=-1)
importances = pd.Series(perm.importances_mean, index=factors).sort_values(ascending=False)
importances.to_csv(os.path.join(args.output_dir, f"mlp_feature_importances{suffix}.csv"), header=["importance"])
print("\nTop-10 wichtigste Faktoren (Permutation-Importance, f1_macro-Abfall):")
print(importances.head(10))
print(f"Gespeichert: mlp_feature_importances{suffix}.csv")

model_path = os.path.join(args.output_dir, f"mlp_condition_classifier{suffix}.joblib")
joblib.dump({"model": final_clf, "scaler": final_scaler, "label_encoder": le, "factors": factors,
            "classes": le.inverse_transform(final_clf.classes_).tolist(), "split_scheme": args.split},
            model_path)
print(f"\nGespeichert: {model_path}")