#!/usr/bin/env python3
"""
DRVI Pseudobulk — Aggregation der Faktor-Aktivität von Zell- auf Patienten-Ebene

DRVI-Faktoren liegen pro Zelle vor, der Outcome (z.B. `classification`) aber
pro Patient. Dieses Skript fasst pro Faktor die Zell-Werte zu einer einzigen
Kennzahl zusammen (Mittelwert und Median) und erzeugt so eine Matrix
Einheit x Faktoren, die direkt fuer patientenbasierte Downstream-Analysen
(z.B. Klassifikation nach Outcome) genutzt werden kann.

Manche Patienten haben mehrere Proben zu verschiedenen Zeitpunkten (z.B.
`m6.1` .. `m6.4` fuer ACS-Patient 6, TP1-TP4; Kontrollen wie `k9` haben nur
eine Probe). `sample_id`/`display_name` kodieren daher "Patient.Zeitpunkt",
nicht den Patienten allein (gleiche Konvention wie in
`DEG/deg_conditions.ipynb`: `patient_id = display_name.split(".")[0]`).
Deshalb unterstuetzt dieses Skript zwei Aggregationsebenen (`--level`):

  - "sample":  eine Zeile pro Probe (Patient x Zeitpunkt). Zeitpunkt-Signal
               bleibt erhalten; fuer Repeated-Measures-Analysen muss
               `patient_id` (in den Metadaten enthalten) als Blocking-Faktor
               beruecksichtigt werden (siehe deg_conditions.ipynb).
  - "patient": eine Zeile pro echtem Patienten, ueber alle Zeitpunkte/Proben
               hinweg gemittelt. Verliert die zeitliche Aufloesung, liefert
               aber die vom Nutzer angefragte reine Patienten x Faktoren
               Matrix (ein Patient = eine Zeile).

Verwendung:
    conda run -n mapra_cytokines python 2_2_b_drvi_pseudobulk.py \
        --embed-input .../visualization/drvi_interpretation/embed.h5ad \
        --output-dir  .../visualization/drvi_interpretation \
        --level patient
"""

import argparse
import os

import anndata as ad
import pandas as pd

# ── Argparse ──────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument(
    "--embed-input",
    default="/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/visualization/drvi_interpretation/embed.h5ad",
    help="embed.h5ad aus 2_2_a_drvi_interpretation.py (Zellen x Faktoren)",
)
parser.add_argument(
    "--summary-csv",
    default="/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/visualization/drvi_interpretation/interpretation_summary.csv",
    help="interpretation_summary.csv aus 2_2_a — liefert Liste der aktiven Faktoren",
)
parser.add_argument(
    "--output-dir",
    default="/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/visualization/drvi_interpretation",
)
parser.add_argument("--sample-key", default="sample_id",
                    help="obs-Spalte, die die Probe (Patient.Zeitpunkt) identifiziert")
parser.add_argument("--outcome-key", default="classification",
                    help="obs-Spalte mit dem Patienten-Outcome (nur fuer Metadaten-Export)")
parser.add_argument("--factor-scope", choices=["active", "all"], default="active",
                    help="nur aktive Faktoren (aus summary-csv) oder alle Faktoren aggregieren")
parser.add_argument("--level", choices=["sample", "patient"], default="patient",
                    help="Aggregationsebene: 'sample' = pro Probe (Patient.Zeitpunkt), "
                         "'patient' = pro echtem Patienten (ueber Zeitpunkte gemittelt)")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ── Daten laden ───────────────────────────────────────────────────────────────

print("=== Embedding laden ===")
embed = ad.read_h5ad(args.embed_input)
print(f"Embedding: {embed.n_obs} Zellen x {embed.n_vars} Faktoren")

if args.factor_scope == "active" and os.path.exists(args.summary_csv):
    summary = pd.read_csv(args.summary_csv, header=None, index_col=0).squeeze("columns")
    active_factors = [f.strip() for f in str(summary["active_factors"]).split(",")]
    factors = [f for f in active_factors if f in embed.var_names]
    print(f"Aktive Faktoren (aus {os.path.basename(args.summary_csv)}): {len(factors)}")
else:
    factors = embed.var_names.tolist()
    print(f"Alle Faktoren: {len(factors)}")

if args.sample_key not in embed.obs.columns:
    raise ValueError(f"Proben-Schluessel '{args.sample_key}' nicht in embed.obs vorhanden.")

# ── Patient/Zeitpunkt aus Proben-ID ableiten ─────────────────────────────────
# Konvention wie in DEG/deg_conditions.ipynb: "m6.4" -> patient_id="m6", timepoint="4"
# Proben ohne Punkt (z.B. "k9", Einzelabnahme) -> timepoint=None.

sample_ids = embed.obs[args.sample_key].astype(str)
split       = sample_ids.str.split(".", n=1, expand=True)
patient_id  = split[0]
timepoint   = split[1] if split.shape[1] > 1 else pd.Series(pd.NA, index=sample_ids.index)

samples_per_patient = pd.DataFrame({"patient_id": patient_id.values, "sample": sample_ids.values}) \
    .drop_duplicates().groupby("patient_id", observed=True).size()
n_patients_multi = (samples_per_patient > 1).sum()
print(f"\n{sample_ids.nunique()} Proben -> {patient_id.nunique()} Patienten "
      f"({n_patients_multi} davon mit mehreren Zeitpunkt-Proben)")

group_key = args.sample_key if args.level == "sample" else "patient_id"

# ── Pseudobulk: Zelle -> Einheit (Probe oder Patient) ────────────────────────

print(f"\n=== Aggregation auf '{args.level}'-Ebene (Gruppierung: {group_key}) ===")

latent = embed[:, factors].X
if hasattr(latent, "toarray"):
    latent = latent.toarray()

factor_df = pd.DataFrame(latent, index=embed.obs_names, columns=factors)
factor_df["patient_id"] = patient_id.values
factor_df[args.sample_key] = sample_ids.values

grouped = factor_df.groupby(group_key, observed=True)

mean_matrix   = grouped[factors].mean()
median_matrix = grouped[factors].median()
n_cells       = grouped.size().rename("n_cells")

print(f"Matrix-Form ({args.level.capitalize()} x Faktoren): {mean_matrix.shape}")

# ── Speichern ─────────────────────────────────────────────────────────────────

mean_path   = os.path.join(args.output_dir, f"pseudobulk_{args.level}_factor_mean.csv")
median_path = os.path.join(args.output_dir, f"pseudobulk_{args.level}_factor_median.csv")
meta_path   = os.path.join(args.output_dir, f"pseudobulk_{args.level}_metadata.csv")

mean_matrix.to_csv(mean_path)
median_matrix.to_csv(median_path)
print(f"Gespeichert: {os.path.basename(mean_path)}")
print(f"Gespeichert: {os.path.basename(median_path)}")

# Metadaten separat (Outcome + Zellzahl + Zeitpunkt-Info), damit die
# Faktor-Matrizen selbst rein numerisch bleiben.
obs_extra = embed.obs.copy()
obs_extra["patient_id"] = patient_id.values
obs_extra[args.sample_key] = sample_ids.values

meta_cols = [c for c in [args.outcome_key, "library"] if c in obs_extra.columns]
metadata = (
    obs_extra[[group_key] + meta_cols]
    .drop_duplicates(subset=group_key)
    .set_index(group_key)
)
metadata = metadata.join(n_cells)

if args.level == "patient":
    n_timepoints = (
        obs_extra[["patient_id", args.sample_key]]
        .drop_duplicates()
        .groupby("patient_id", observed=True)
        .size()
        .rename("n_samples")
    )
    metadata = metadata.join(n_timepoints)
else:
    metadata["patient_id"] = obs_extra.drop_duplicates(subset=group_key).set_index(group_key)["patient_id"]

metadata.to_csv(meta_path)
print(f"Gespeichert: {os.path.basename(meta_path)}")

print(f"\nFertig. {mean_matrix.shape[0]} {args.level}s x {mean_matrix.shape[1]} Faktoren.")
print(f"Ergebnisse in: {args.output_dir}")
