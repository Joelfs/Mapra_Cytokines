#!/usr/bin/env python3
"""
MLP-Klassifikator: sagt die Bedingung (ACS / CCS / non-CCS) eines Patienten
aus den aktiven DRVI-Faktoren einzelner Zellen voraus.

MLP-Gegenstueck zu rf_condition_classifier.py und xgb_condition_classifier.py
-- gleiche Daten, gleiches Split-Schema, gleiche Zell-Ebene-/CV-/Holdout-
Logik (kein Pseudobulking mehr, siehe unten), nur der Classifier ist
ausgetauscht.

Nutzt data_for_practicum_post_integration_curated_activedims_split.h5ad, mit
denselben vier moeglichen Split-Schemata wie rf_condition_classifier.py
(--split stratified/balanced_322/balanced_222/balanced_432). NUR 'stratified'
(und die balanced_* Schemata) decken alle drei Zielklassen ab -- 'timepoint'
ist deshalb bewusst NICHT in den --split-choices enthalten (siehe
xgb_condition_classifier.py-Docstring: cv_fold_timepoint schliesst CCS und
non-CCS komplett aus).

Bedingungs-Mapping eingeschraenkt (identisch zu RF/XGBoost): ACS = nur
"acs_w_o_infection" (19 Patienten), CCS = "ccs" (16), non-CCS = nur
"vollstaendiger_ausschluss" (10). Alle anderen Klassifikationswerte sind in
den Split-Spalten "excluded" und werden automatisch entfernt.

KEIN Pseudobulking mehr (gleiche Begruendung wie rf_condition_classifier.py/
xgb_condition_classifier.py): statt einer Zeile pro Patient (Mittelwert der
DRVI-Faktoren ueber alle seine Zellen) wird JEDE EINZELNE ZELLE als eigene
Trainings-/Test-Zeile verwendet (Label = die Bedingung des Patienten, dem
die Zelle gehoert). Die Split-/Fold-Zuordnung bleibt weiterhin auf
Patienten-Ebene (aus `holdout_<schema>`/`cv_fold_<schema>`, pro Zelle
identisch fuer alle Zellen desselben Patienten) -- kein Patient landet je in
Training UND Validation/Test, also kein Data Leakage.

Weil die Bewertung pro Zelle (viele stark korrelierte Zeilen pro Patient)
ein anderes Bild liefert als die klinisch relevante Frage "wird der Patient
richtig klassifiziert", werden Vorhersagen zusaetzlich per Mehrheitsentscheid
("majority vote") ueber alle Zellen eines Patienten zu einer
Patienten-Vorhersage aggregiert (identisch zu xgb_condition_classifier.py).
Report/Confusion-Matrix/`accuracy`+`f1_macro` in den Ergebnisdateien beziehen
sich auf diese Patienten-Ebene (fuer Vergleichbarkeit mit den bisherigen
Pseudobulk-Ergebnissen); die reinen Zell-Ebene-Metriken stehen zusaetzlich
als `accuracy_cell`/`f1_macro_cell` in mlp_cv_fold_metrics_<schema>.csv und
werden auf der Konsole ausgegeben.

Vier technisch bedingte Unterschiede zu RF/XGBoost (dieselben wie in
mlp_timepoint_classifier.py):
  1. Feature-Skalierung: StandardScaler, pro Trainings-Fold neu gefittet.
  2. Klassen-Balance: Oversampling statt class_weight/sample_weight (von
     MLPClassifier nicht unterstuetzt).
  3. Feature Importance: SHAP ueber KernelExplainer statt TreeExplainer --
     MLPClassifier hat kein Baum-Aequivalent, KernelExplainer ist
     modell-agnostisch, aber teuer. Deshalb wird (anders als bei
     xgb_condition_classifier.py, das TreeExplainer auf dem GESAMTEN
     CV-Pool laufen lassen kann) nur eine Zufallsstichprobe von Zellen
     erklaert, gegen einen kmeans-summierten Background (siehe unten).
  4. Label-Encoding: LabelEncoder noetig, weil MLPClassifier(early_stopping=
     True) intern predict() aufruft und mit np.isnan() prueft -- das crasht
     mit String-Klassenlabels ("ACS"/"CCS"/"non-CCS").

Ablauf:
  1. Zell-Ebene: jede Zelle eines eingeschlossenen Patienten ist ein
     Trainingsbeispiel (Feature = aktive DRVI-Faktoren dieser Zelle, Label =
     Bedingung des Patienten). Kein Aggregieren/Mitteln pro Patient.
  2. 5-fach Cross-Validation innerhalb des CV-Pools (`holdout_<schema> == "cv"`),
     Fold-Zuordnung aus `cv_fold_<schema>` uebernommen (patienten-rein).
     Metriken auf Zell- UND (per Mehrheitsentscheid) Patienten-Ebene.
  3. Finales Modell auf dem gesamten CV-Pool (alle Zellen) trainiert, einmalig
     auf dem nie angeruehrten Test-Holdout (`holdout_<schema> == "test"`)
     bewertet -- ebenfalls beide Ebenen.

Ergebnis (Dateinamen mit Schema-Suffix, gleiche Konvention wie RF/XGBoost --
ueberschreibt die alten pseudobulk-basierten Outputs):
  - mlp_condition_classifier_<schema>.joblib
  - mlp_cv_fold_metrics_<schema>.csv        (Patienten- UND Zell-Ebene)
  - mlp_test_holdout_report_<schema>.txt    (Patienten- UND Zell-Ebene)
  - mlp_feature_importances_<schema>.csv    (= mean |SHAP value|, Spaltenname
    bleibt "importance" fuer Kompatibilitaet mit plot_mlp_results.py)
  - mlp_shap_values_<schema>.csv            (rohe mean |SHAP value| pro
    Klasse + Mittel, gleiche Struktur wie xgb_shap_values_<schema>.csv)

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
import shap
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
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


def majority_vote(df, group_col, pred_col):
    """Aggregiert Zell-Vorhersagen zu einer Vorhersage pro Gruppe (Patient) per
    Mehrheitsentscheid (identisch zu xgb_condition_classifier.py)."""
    return df.groupby(group_col, observed=True)[pred_col].agg(lambda s: s.value_counts().idxmax())


# ── 1. Daten laden (Zell-Ebene, KEIN Pseudobulk) ─────────────────────────────

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

n_cells_before = cell_df.shape[0]
cell_df = cell_df[cell_df["holdout"] != "excluded"].copy()
n_patients = cell_df["patient_id"].nunique()
print(f"\n{n_cells_before} Zellen insgesamt, {n_cells_before - cell_df.shape[0]} ausgeschlossen "
      f"(classification ausserhalb ACS/CCS/non-CCS-Mapping)")
print(f"{cell_df.shape[0]} Zellen von {n_patients} Patienten x {len(factors)} Faktoren "
      f"(Zell-Ebene, kein Pseudobulk)")
print(cell_df["condition"].value_counts())
print(cell_df.groupby("condition", observed=True)["patient_id"].nunique().rename("n_patients"))

# Siehe Docstring Punkt 4: LabelEncoder noetig wegen MLPClassifier(early_stopping)
# + np.isnan() auf String-Labels.
le = LabelEncoder()
cell_df["condition_enc"] = le.fit_transform(cell_df["condition"])
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


# ── 2. 5-fach CV (Fold-Zuordnung aus cv_fold_<schema> uebernommen, patienten-rein) ──

print(f"\n=== 5-fach Cross-Validation (CV-Pool, Schema={args.split}) ===")
cv_mask = cell_df["holdout"] == "cv"
cv_df = cell_df[cv_mask]
n_folds = int(cv_df["cv_fold"].max()) + 1
n_patients_cv = cv_df.drop_duplicates("patient_id")
print(f"CV-Pool: {int(cv_mask.sum())} Zellen von {n_patients_cv.shape[0]} Patienten, {n_folds} Folds")
print(n_patients_cv.groupby("cv_fold")["condition"].value_counts().unstack())

fold_rows = []
for fold in range(n_folds):
    train_mask = cv_mask & (cell_df["cv_fold"] != fold)
    val_mask = cv_mask & (cell_df["cv_fold"] == fold)

    X_train_raw = cell_df.loc[train_mask, factors]
    y_train_raw = cell_df.loc[train_mask, "condition_enc"]
    X_train_bal, y_train_bal = oversample_balanced(X_train_raw, y_train_raw, args.random_state)

    scaler = StandardScaler().fit(X_train_bal)
    X_train_scaled = scaler.transform(X_train_bal)
    X_val_scaled = scaler.transform(cell_df.loc[val_mask, factors])

    clf = make_model()
    clf.fit(X_train_scaled, y_train_bal)
    y_pred_cell_enc = clf.predict(X_val_scaled)
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
fold_metrics.to_csv(os.path.join(args.output_dir, f"mlp_cv_fold_metrics{suffix}.csv"), index=False)
print(f"\nCV-Mittelwert (Patienten-Ebene, Mehrheitsentscheid): accuracy={fold_metrics['accuracy'].mean():.3f} "
      f"(+/-{fold_metrics['accuracy'].std():.3f}), "
      f"f1_macro={fold_metrics['f1_macro'].mean():.3f} (+/-{fold_metrics['f1_macro'].std():.3f})")
print(f"CV-Mittelwert (Zell-Ebene): accuracy={fold_metrics['accuracy_cell'].mean():.3f} "
      f"(+/-{fold_metrics['accuracy_cell'].std():.3f}), "
      f"f1_macro={fold_metrics['f1_macro_cell'].mean():.3f} (+/-{fold_metrics['f1_macro_cell'].std():.3f})")
print(f"Gespeichert: mlp_cv_fold_metrics{suffix}.csv")

# ── 3. Finales Modell (ganzer CV-Pool, alle Zellen) + Test-Holdout ──────────

print("\n=== Finales Modell + Test-Holdout ===")
test_mask = cell_df["holdout"] == "test"
test_df_patients = cell_df.loc[test_mask].drop_duplicates("patient_id")
print(f"CV-Pool (Training): {int(cv_mask.sum())} Zellen / {n_patients_cv.shape[0]} Patienten, "
      f"Test-Holdout: {int(test_mask.sum())} Zellen / {test_df_patients.shape[0]} Patienten")

X_cv_raw = cell_df.loc[cv_mask, factors]
y_cv_raw = cell_df.loc[cv_mask, "condition_enc"]
X_cv_bal, y_cv_bal = oversample_balanced(X_cv_raw, y_cv_raw, args.random_state)

final_scaler = StandardScaler().fit(X_cv_bal)
X_cv_scaled = final_scaler.transform(X_cv_bal)

final_clf = make_model()
final_clf.fit(X_cv_scaled, y_cv_bal)

X_test_scaled = final_scaler.transform(cell_df.loc[test_mask, factors])
y_pred_test_cell_enc = final_clf.predict(X_test_scaled)
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
print("Confusion matrix (Zeilen=wahr, Spalten=vorhergesagt, Patienten-Ebene):")
labels = sorted(cell_df["condition"].unique())
cm = confusion_matrix(y_true_test, y_pred_test, labels=labels)
print(pd.DataFrame(cm, index=labels, columns=labels))

with open(os.path.join(args.output_dir, f"mlp_test_holdout_report{suffix}.txt"), "w") as f:
    f.write(f"Split-Schema: {args.split}\n")
    f.write(f"Test-Holdout: {y_pred_test.shape[0]} Patienten "
            f"(Mehrheitsentscheid ueber {int(test_mask.sum())} Zellen, kein Pseudobulking)\n\n")
    f.write(report)
    f.write(f"\nZell-Ebene (zum Vergleich, ohne Mehrheitsentscheid): accuracy={acc_test_cell:.3f}, "
            f"f1_macro={f1_test_cell:.3f} (n={int(test_mask.sum())} Zellen)\n")
    f.write("\nConfusion matrix (Zeilen=wahr, Spalten=vorhergesagt, Patienten-Ebene):\n")
    f.write(pd.DataFrame(cm, index=labels, columns=labels).to_string())
print(f"\nGespeichert: mlp_test_holdout_report{suffix}.txt")

# ── Feature Importances via SHAP (KernelExplainer, CV-Pool, Zell-Ebene) ─────
# Gleiche "mean |SHAP value|"-Konvention wie RF/XGBoost/logistische
# Regression, damit alle vier Classifier im Paper dieselbe Feature-
# Importance-Methodik verwenden (Lisas Wunsch: nicht mehrere verschiedene
# Metriken erklaeren muessen). MLPClassifier hat keinen TreeExplainer --
# KernelExplainer ist modell-agnostisch, aber teuer, deshalb (anders als
# xgb_condition_classifier.py, das TreeExplainer auf dem GESAMTEN CV-Pool
# laufen lassen kann):
#   - Background: 50 kmeans-summierte Punkte statt des ganzen CV-Pools
#   - Erklaert wird nur eine zufaellige Stichprobe von Zellen (nicht alle
#     Zehntausende) -- bei Zell-Ebene-Daten sonst praktisch nicht rechenbar.
# Beide Groessen sind unten als Konstanten einstellbar.

print("\n=== SHAP Feature Importance (KernelExplainer, CV-Pool) ===")
SHAP_BACKGROUND_SIZE = 50
SHAP_SAMPLE_SIZE = 500

background = shap.kmeans(X_cv_scaled, SHAP_BACKGROUND_SIZE)
rng = np.random.default_rng(args.random_state)
sample_idx = rng.choice(X_cv_scaled.shape[0], size=min(SHAP_SAMPLE_SIZE, X_cv_scaled.shape[0]), replace=False)
X_shap_sample = X_cv_scaled[sample_idx]

explainer = shap.KernelExplainer(final_clf.predict_proba, background)
shap_values = explainer.shap_values(X_shap_sample)

# shap_values ist je nach SHAP-Version eine Liste (eine pro Klasse) oder ein
# 3D-Array. In ein einheitliches (n_classes, n_samples, n_features) Array
# bringen, damit wir -- wie in xgb_condition_classifier.py -- sowohl den
# Mittelwert pro Klasse (mlp_shap_values.csv) als auch den Gesamt-Mittelwert
# ueber alle Klassen (mlp_feature_importances.csv) berechnen koennen.
shap_values = np.asarray(shap_values)
if shap_values.ndim == 3 and shap_values.shape[0] != len(final_clf.classes_):
    # (n_samples, n_features, n_classes) -> (n_classes, n_samples, n_features)
    shap_values = np.moveaxis(shap_values, -1, 0)
elif shap_values.ndim == 2:
    # Binaerfall/eine gemeinsame Matrix -- als "eine Klasse" behandeln
    shap_values = shap_values[np.newaxis, ...]

mean_abs_per_class = np.stack([np.abs(sv).mean(axis=0) for sv in shap_values])  # (n_classes, n_features)
n_classes_out = mean_abs_per_class.shape[0]
class_labels = le.classes_ if n_classes_out == len(le.classes_) else [f"class_{i}" for i in range(n_classes_out)]

shap_df = pd.DataFrame(mean_abs_per_class.T, index=factors, columns=class_labels)
shap_df["mean_abs_shap"] = shap_df.mean(axis=1)
shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)
shap_df.to_csv(os.path.join(args.output_dir, f"mlp_shap_values{suffix}.csv"))
print(f"Gespeichert: mlp_shap_values{suffix}.csv")

# gleicher Dateiname/Spaltenname wie zuvor (importance), jetzt SHAP-basiert,
# damit plot_mlp_results.py unveraendert bleibt.
importances = shap_df["mean_abs_shap"]
importances.to_csv(os.path.join(args.output_dir, f"mlp_feature_importances{suffix}.csv"), header=["importance"])
print(f"\n(SHAP auf Zufallsstichprobe von {X_shap_sample.shape[0]} Zellen aus dem CV-Pool, "
      f"Background={SHAP_BACKGROUND_SIZE} kmeans-Punkte)")
print("\nTop-10 wichtigste Faktoren (mean |SHAP value|):")
print(importances.head(10))
print(f"Gespeichert: mlp_feature_importances{suffix}.csv")

model_path = os.path.join(args.output_dir, f"mlp_condition_classifier{suffix}.joblib")
joblib.dump({"model": final_clf, "scaler": final_scaler, "label_encoder": le, "factors": factors,
            "classes": le.inverse_transform(final_clf.classes_).tolist(), "split_scheme": args.split,
            "pseudobulk": False},
            model_path)
print(f"\nGespeichert: {model_path}")