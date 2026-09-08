#!/usr/bin/env python3
"""
MLP-Klassifikator: sagt den Zeitpunkt (TP1-TP4) einer Zelle aus ihren
aktiven DRVI-Faktoren voraus -- nur fuer ACS_sterile (classification ==
"acs_w_o_infection", 19 Patienten, 62 Proben).

MLP-Gegenstueck zu rf_timepoint_classifier.py und xgb_timepoint_classifier.py
-- gleiche Daten, gleiches Split-Schema, gleiche Zell-Ebene-/CV-/Holdout-
Logik (kein Pseudobulking mehr, siehe unten), nur der Classifier ist
ausgetauscht.

Nutzt data_for_practicum_post_integration_curated_activedims_split.h5ad,
Spalten `holdout_timepoint`/`cv_fold_timepoint`.

KEIN Pseudobulking mehr (gleiche Begruendung wie rf_timepoint_classifier.py/
xgb_timepoint_classifier.py): statt einer Zeile pro Probe (Mittelwert der
DRVI-Faktoren ueber alle ihre Zellen) wird JEDE EINZELNE ZELLE als eigene
Trainings-/Test-Zeile verwendet (Label = der Zeitpunkt der Probe, zu der die
Zelle gehoert). Die Split-/Fold-Zuordnung bleibt weiterhin auf Proben-Ebene
(aus `holdout_timepoint`/`cv_fold_timepoint`, pro Zelle identisch fuer alle
Zellen derselben Probe) -- keine Probe landet je in Training UND
Validation/Test, also kein Data Leakage.

Weil die Bewertung pro Zelle (viele stark korrelierte Zeilen pro Probe) ein
anderes Bild liefert als die klinisch relevante Frage "wird der Zeitpunkt
dieser Probe richtig klassifiziert", werden Vorhersagen zusaetzlich per
Mehrheitsentscheid ("majority vote") ueber alle Zellen einer Probe zu einer
Proben-Vorhersage aggregiert (identisch zum Prinzip in
xgb_condition_classifier.py, hier auf Proben- statt Patienten-Ebene, weil
der Zeitpunkt pro Probe und nicht pro Patient definiert ist). Report/
Confusion-Matrix/`accuracy`+`f1_macro` in den Ergebnisdateien beziehen sich
auf diese Proben-Ebene (fuer Vergleichbarkeit mit den bisherigen
Pseudobulk-Ergebnissen); die reinen Zell-Ebene-Metriken stehen zusaetzlich
als `accuracy_cell`/`f1_macro_cell` in mlp_timepoint_cv_fold_metrics.csv und
werden auf der Konsole ausgegeben.

Vier Unterschiede zu RF/XGBoost, technisch bedingt durch den MLP:
  1. Feature-Skalierung: MLPs sind (anders als Baum-Modelle) empfindlich
     gegenueber der Skala der Eingabe-Features -- StandardScaler wird pro
     Trainings-Fold NEU gefittet (kein Leakage aus Val/Test).
  2. Klassen-Balance: sklearn's MLPClassifier unterstuetzt kein
     sample_weight/class_weight (anders als RandomForest/XGBoost). Statt
     inverser Klassenhaeufigkeit wird daher im Training per Oversampling
     (Ziehen mit Zuruecklegen bis alle Klassen gleich gross sind)
     balanciert.
  3. Feature Importance: SHAP ueber KernelExplainer statt TreeExplainer --
     MLPClassifier hat kein Baum-Aequivalent, KernelExplainer ist
     modell-agnostisch, aber teuer. Deshalb wird (anders als bei
     xgb_condition_classifier.py, das TreeExplainer auf dem GESAMTEN
     CV-Pool laufen lassen kann) nur eine Zufallsstichprobe von Zellen
     erklaert, gegen einen kmeans-summierten Background (siehe unten).
  4. Label-Encoding: MLPClassifier(early_stopping=True) ruft intern
     predict() auf und prueft das Ergebnis mit np.isnan() -- das crasht mit
     String-Klassenlabels ("1".."4"). Deshalb LabelEncoder wie im
     XGBoost-Skript; Reports/Confusion-Matrix werden wieder in
     String-Labels zurueckdecodiert.

Ablauf:
  1. Zell-Ebene: jede Zelle einer eingeschlossenen Probe ist ein
     Trainingsbeispiel (Feature = aktive DRVI-Faktoren dieser Zelle, Label =
     Zeitpunkt ihrer Probe). Kein Aggregieren/Mitteln pro Probe.
  2. 5-fach Cross-Validation innerhalb des CV-Pools (`holdout_timepoint == "cv"`),
     Fold-Zuordnung aus `cv_fold_timepoint` uebernommen (proben-rein).
     Metriken auf Zell- UND (per Mehrheitsentscheid) Proben-Ebene.
  3. Finales Modell auf dem gesamten CV-Pool (alle Zellen) trainiert, einmalig
     auf dem nie angeruehrten Test-Holdout (`holdout_timepoint == "test"`)
     bewertet -- ebenfalls beide Ebenen.

Ergebnis (ueberschreibt die alten pseudobulk-basierten Outputs):
  - mlp_timepoint_classifier.joblib
  - mlp_timepoint_cv_fold_metrics.csv        (Proben- UND Zell-Ebene)
  - mlp_timepoint_test_holdout_report.txt    (Proben- UND Zell-Ebene)
  - mlp_timepoint_feature_importances.csv    (= mean |SHAP value|, Spaltenname
    bleibt "importance" fuer Kompatibilitaet mit plot_mlp_results.py)
  - mlp_timepoint_shap_values.csv            (rohe mean |SHAP value| pro
    Klasse + Mittel, gleiche Struktur wie xgb_shap_values.csv)

Verwendung:
    conda run -n mapra_cytokines python mlp_timepoint_classifier.py
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
parser.add_argument("--hidden-layer-sizes", default="32,16",
                     help="Kommagetrennt, z.B. '32,16' fuer zwei Hidden Layer")
parser.add_argument("--alpha", type=float, default=1e-2, help="L2-Regularisierung")
parser.add_argument("--learning-rate-init", type=float, default=1e-3)
parser.add_argument("--max-iter", type=int, default=2000)
parser.add_argument("--random-state", type=int, default=0)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
hidden_layer_sizes = tuple(int(x) for x in args.hidden_layer_sizes.split(","))


def majority_vote(df, group_col, pred_col):
    """Aggregiert Zell-Vorhersagen zu einer Vorhersage pro Gruppe (Probe) per
    Mehrheitsentscheid (gleiches Prinzip wie xgb_condition_classifier.py)."""
    return df.groupby(group_col, observed=True)[pred_col].agg(lambda s: s.value_counts().idxmax())


# ── 1. Daten laden (Zell-Ebene, KEIN Pseudobulk) ─────────────────────────────

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
n_samples = cell_df["sample_id"].nunique()
print(f"{n_before_cells - cell_df.shape[0]} Zellen ausgeschlossen (nicht ACS_sterile)")
print(f"\n{cell_df.shape[0]} Zellen von {n_samples} Proben x {len(factors)} Faktoren "
      f"(Zell-Ebene, kein Pseudobulk)")
print(cell_df["timepoint"].value_counts().sort_index())
print(cell_df.groupby("timepoint", observed=True)["sample_id"].nunique().rename("n_samples"))

# MLPClassifier(early_stopping=True) ruft intern predict() waehrend des
# Trainings auf und prueft das Ergebnis mit np.isnan() -- das crasht mit
# String-Klassenlabels ("1","2","3","4"). Deshalb hier (wie im
# XGBoost-Skript) auf Integer-codierte Labels umsteigen; fuer die
# Reports/Confusion-Matrix am Ende wieder zurueck in Strings decodieren.
le = LabelEncoder()
cell_df["timepoint_enc"] = le.fit_transform(cell_df["timepoint"])
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


# ── 2. 5-fach CV (Fold-Zuordnung aus cv_fold_timepoint uebernommen, proben-rein) ──

print("\n=== 5-fach Cross-Validation (CV-Pool) ===")
cv_mask = cell_df["holdout"] == "cv"
cv_df = cell_df[cv_mask]
n_folds = int(cv_df["cv_fold"].max()) + 1
n_samples_cv = cv_df.drop_duplicates("sample_id")
print(f"CV-Pool: {int(cv_mask.sum())} Zellen von {n_samples_cv.shape[0]} Proben, {n_folds} Folds")
print(n_samples_cv.groupby("cv_fold")["timepoint"].value_counts().unstack())

fold_rows = []
for fold in range(n_folds):
    train_mask = cv_mask & (cell_df["cv_fold"] != fold)
    val_mask = cv_mask & (cell_df["cv_fold"] == fold)

    X_train_raw = cell_df.loc[train_mask, factors]
    y_train_raw = cell_df.loc[train_mask, "timepoint_enc"]
    X_train_bal, y_train_bal = oversample_balanced(X_train_raw, y_train_raw, args.random_state)

    scaler = StandardScaler().fit(X_train_bal)
    X_train_scaled = scaler.transform(X_train_bal)
    X_val_scaled = scaler.transform(cell_df.loc[val_mask, factors])

    clf = make_model()
    clf.fit(X_train_scaled, y_train_bal)
    y_pred_cell_enc = clf.predict(X_val_scaled)
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
fold_metrics.to_csv(os.path.join(args.output_dir, "mlp_timepoint_cv_fold_metrics.csv"), index=False)
print(f"\nCV-Mittelwert (Proben-Ebene, Mehrheitsentscheid): accuracy={fold_metrics['accuracy'].mean():.3f} "
      f"(+/-{fold_metrics['accuracy'].std():.3f}), "
      f"f1_macro={fold_metrics['f1_macro'].mean():.3f} (+/-{fold_metrics['f1_macro'].std():.3f})")
print(f"CV-Mittelwert (Zell-Ebene): accuracy={fold_metrics['accuracy_cell'].mean():.3f} "
      f"(+/-{fold_metrics['accuracy_cell'].std():.3f}), "
      f"f1_macro={fold_metrics['f1_macro_cell'].mean():.3f} (+/-{fold_metrics['f1_macro_cell'].std():.3f})")
print("Gespeichert: mlp_timepoint_cv_fold_metrics.csv")

# ── 3. Finales Modell (ganzer CV-Pool, alle Zellen) + Test-Holdout ──────────

print("\n=== Finales Modell + Test-Holdout ===")
test_mask = cell_df["holdout"] == "test"
test_df_samples = cell_df.loc[test_mask].drop_duplicates("sample_id")
print(f"CV-Pool (Training): {int(cv_mask.sum())} Zellen / {n_samples_cv.shape[0]} Proben, "
      f"Test-Holdout: {int(test_mask.sum())} Zellen / {test_df_samples.shape[0]} Proben")

X_cv_raw = cell_df.loc[cv_mask, factors]
y_cv_raw = cell_df.loc[cv_mask, "timepoint_enc"]
X_cv_bal, y_cv_bal = oversample_balanced(X_cv_raw, y_cv_raw, args.random_state)

final_scaler = StandardScaler().fit(X_cv_bal)
X_cv_scaled = final_scaler.transform(X_cv_bal)

final_clf = make_model()
final_clf.fit(X_cv_scaled, y_cv_bal)

X_test_scaled = final_scaler.transform(cell_df.loc[test_mask, factors])
y_pred_test_cell_enc = final_clf.predict(X_test_scaled)
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
print("Confusion matrix (Zeilen=wahr, Spalten=vorhergesagt, Proben-Ebene):")
labels = sorted(cell_df["timepoint"].unique())
cm = confusion_matrix(y_true_test, y_pred_test, labels=labels)
print(pd.DataFrame(cm, index=labels, columns=labels))

with open(os.path.join(args.output_dir, "mlp_timepoint_test_holdout_report.txt"), "w") as f:
    f.write("Zielgroesse: Zeitpunkt (TP1-TP4), nur ACS_sterile\n")
    f.write(f"Test-Holdout: {y_pred_test.shape[0]} Proben "
            f"(Mehrheitsentscheid ueber {int(test_mask.sum())} Zellen, kein Pseudobulking)\n\n")
    f.write(report)
    f.write(f"\nZell-Ebene (zum Vergleich, ohne Mehrheitsentscheid): accuracy={acc_test_cell:.3f}, "
            f"f1_macro={f1_test_cell:.3f} (n={int(test_mask.sum())} Zellen)\n")
    f.write("\nConfusion matrix (Zeilen=wahr, Spalten=vorhergesagt, Proben-Ebene):\n")
    f.write(pd.DataFrame(cm, index=labels, columns=labels).to_string())
print("\nGespeichert: mlp_timepoint_test_holdout_report.txt")

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
# Mittelwert pro Klasse (mlp_timepoint_shap_values.csv) als auch den
# Gesamt-Mittelwert ueber alle Klassen (mlp_timepoint_feature_importances.csv)
# berechnen koennen.
shap_values = np.asarray(shap_values)
if shap_values.ndim == 3 and shap_values.shape[0] != len(final_clf.classes_):
    # (n_samples, n_features, n_classes) -> (n_classes, n_samples, n_features)
    shap_values = np.moveaxis(shap_values, -1, 0)
elif shap_values.ndim == 2:
    shap_values = shap_values[np.newaxis, ...]

mean_abs_per_class = np.stack([np.abs(sv).mean(axis=0) for sv in shap_values])  # (n_classes, n_features)
n_classes_out = mean_abs_per_class.shape[0]
class_labels = le.classes_ if n_classes_out == len(le.classes_) else [f"class_{i}" for i in range(n_classes_out)]

shap_df = pd.DataFrame(mean_abs_per_class.T, index=factors, columns=class_labels)
shap_df["mean_abs_shap"] = shap_df.mean(axis=1)
shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)
shap_df.to_csv(os.path.join(args.output_dir, "mlp_timepoint_shap_values.csv"))
print("Gespeichert: mlp_timepoint_shap_values.csv")

# gleicher Dateiname/Spaltenname wie zuvor (importance), jetzt SHAP-basiert,
# damit plot_mlp_results.py unveraendert bleibt.
importances = shap_df["mean_abs_shap"]
importances.to_csv(os.path.join(args.output_dir, "mlp_timepoint_feature_importances.csv"), header=["importance"])
print(f"\n(SHAP auf Zufallsstichprobe von {X_shap_sample.shape[0]} Zellen aus dem CV-Pool, "
      f"Background={SHAP_BACKGROUND_SIZE} kmeans-Punkte)")
print("\nTop-10 wichtigste Faktoren (mean |SHAP value|):")
print(importances.head(10))
print("Gespeichert: mlp_timepoint_feature_importances.csv")

model_path = os.path.join(args.output_dir, "mlp_timepoint_classifier.joblib")
joblib.dump({"model": final_clf, "scaler": final_scaler, "label_encoder": le, "factors": factors,
            "classes": le.inverse_transform(final_clf.classes_).tolist(), "pseudobulk": False}, model_path)
print(f"\nGespeichert: {model_path}")