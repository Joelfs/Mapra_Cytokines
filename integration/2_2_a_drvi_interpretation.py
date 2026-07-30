#!/usr/bin/env python3
"""
DRVI Factor Interpretation — Pipeline gemäß drvi_interpretability_pipeline.pdf

Schritte (nach PDF-Vorlage):
   1.  Modell laden + AnnData-Setup
   2.  Embedding (embed) erstellen + UMAP auf embed berechnen
   3.  Latente Dimensionsstatistiken (set_latent_dimension_stats)
   4.  drvi.utils.pl.plot_latent_dimension_stats (alle + nur aktive)
   5.  drvi.utils.pl.plot_latent_dims_in_umap
   6.  drvi.utils.pl.plot_latent_dims_in_heatmap
   7.  OOD Interpretability Scores: berechnen + plotten + als DataFrame
   8.  Top-Gene pro Faktor auf DRVI-UMAP
   9.  IND Interpretability Scores: berechnen + plotten + als DataFrame
  10.  Identifikation von Programmen (g:Profiler Gene-Set-Enrichment)
  11.  Faktor-Spezifität + Validierung mit Marker-Genen

Verwendung:
    conda run -n mapra_cytokines python drvi_interpretation.py \
        --drvi-input  .../drvi_results_consensus/drvi_representation.h5ad \
        --model-dir   .../drvi_results_consensus/drvi_model \
        --output-dir  .../repo/Mapra_Cytokines/visualization/drvi_interpretation
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import requests

import drvi.utils.plotting as drvi_pl
from drvi.model import DRVI

warnings.filterwarnings("ignore")

# ── Argparse ──────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument(
    "--drvi-input",
    default="/vol/disk/ubuntu/master_practicum_cytokines/joel/workdir/drvi_results_consensus/drvi_representation.h5ad",
)
parser.add_argument(
    "--model-dir",
    default="/vol/disk/ubuntu/master_practicum_cytokines/joel/workdir/drvi_results_consensus/drvi_model",
)
parser.add_argument(
    "--output-dir",
    default="/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/visualization/drvi_interpretation",
)
parser.add_argument("--hvg-col",         default="highly_variable")
parser.add_argument("--batch-key",       default="library")
parser.add_argument("--celltype-key",    default="cell_type_Scanorama")
parser.add_argument("--vanished-thresh", type=float, default=0.5)
parser.add_argument("--n-top-genes",     type=int,   default=10)
parser.add_argument("--n-gprofiler",     type=int,   default=5,
                    help="Anzahl Top-Faktoren für g:Profiler")
parser.add_argument("--leiden-key",      default="leiden_drvi")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
sc.settings.figdir  = args.output_dir
sc.settings.verbosity = 0
sc.settings.set_figure_params(dpi=100, facecolor="white", frameon=False)

embed_path = os.path.join(args.output_dir, "embed.h5ad")


def _save_current_figs(prefix: str) -> None:
    """Speichert alle offenen Matplotlib-Figuren und schließt sie."""
    figs = [plt.figure(i) for i in plt.get_fignums()]
    if not figs:
        return
    for j, fig in enumerate(figs):
        suffix = f"_{j}" if len(figs) > 1 else ""
        out = os.path.join(args.output_dir, f"{prefix}{suffix}.png")
        try:
            fig.savefig(out, bbox_inches="tight", dpi=100)
        except Exception:
            pass
    plt.close("all")
    base = os.path.basename(f"{prefix}.png")
    print(f"Gespeichert: {base}")


# ── 1. Modell laden + AnnData-Setup ──────────────────────────────────────────

print("=== 1. Modell und Daten laden ===")
adata = sc.read_h5ad(args.drvi_input)
adata_hvg = adata[:, adata.var[args.hvg_col]].copy()

DRVI.setup_anndata(
    adata_hvg,
    layer="counts",
    categorical_covariate_keys=[args.batch_key],
    is_count_data=True,
)
model = DRVI.load(args.model_dir, adata=adata_hvg)
print(f"Modell geladen: {model}")

# ── 2. Embedding erstellen + UMAP auf embed berechnen ────────────────────────

print("\n=== 2. Embedding + UMAP ===")

if os.path.exists(embed_path):
    embed = sc.read_h5ad(embed_path)
    if set(embed.obs_names) != set(adata.obs_names):
        print(f"Gecachtes embed passt nicht zu adata — wird neu berechnet.")
        embed = None
    else:
        print(f"Lade gespeichertes embed: {embed_path}")

if not os.path.exists(embed_path) or embed is None:
    latent_rep = model.get_latent_representation()
    n_latent   = latent_rep.shape[1]

    embed = sc.AnnData(
        X   = latent_rep,
        obs = adata.obs.copy(),
    )
    embed.var_names = [f"DR{i+1}" for i in range(n_latent)]

    # UMAP auf dem latenten Raum — wie im PDF
    sc.pp.neighbors(embed, n_neighbors=10, use_rep="X", n_pcs=n_latent)
    sc.tl.umap(embed, spread=1.0, min_dist=0.5, random_state=123)

    embed.write_h5ad(embed_path)
    print(f"embed gespeichert: {embed_path}")

print(f"Embedding shape: {embed.shape}  (Zellen × Faktoren)")

# DRVI-UMAP aus embed in adata kopieren (für Gen-Expression-Plots auf UMAP)
umap_df = pd.DataFrame(embed.obsm["X_umap"], index=embed.obs_names)
adata.obsm["X_drvi_umap"] = umap_df.loc[adata.obs_names].values

# adata.obsm["X_umap"] auf DRVI-UMAP setzen (für sc.pl.umap())
adata.obsm["X_umap"] = adata.obsm["X_drvi_umap"]

# UMAP auf embed gefärbt nach Batch + Zelltyp (wie im PDF)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sc.pl.umap(embed, color=args.batch_key,    ax=axes[0], show=False, title=args.batch_key)
sc.pl.umap(embed, color=args.celltype_key, ax=axes[1], show=False, title=args.celltype_key)
plt.suptitle("Embed UMAP (latenter Raum)", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "embed_umap.png"), bbox_inches="tight", dpi=120)
plt.close("all")
print("Gespeichert: embed_umap.png")

# ── 3. Latente Dimensionsstatistiken ─────────────────────────────────────────

print("\n=== 3. Latente Dimensionsstatistiken (set_latent_dimension_stats) ===")
model.set_latent_dimension_stats(embed, adata_hvg, vanished_threshold=args.vanished_thresh)

print("embed.var Spalten:", list(embed.var.columns))

if "vanished" in embed.var.columns:
    n_vanished  = int(embed.var["vanished"].sum())
    n_active    = int((~embed.var["vanished"]).sum())
    active_dims = embed.var_names[~embed.var["vanished"]].tolist()
else:
    latent_vals = embed.X if not hasattr(embed.X, "toarray") else embed.X.toarray()
    max_abs     = np.abs(latent_vals).max(axis=0)
    vanished_mask = max_abs < args.vanished_thresh
    embed.var["vanished"]      = vanished_mask
    embed.var["max_abs_value"] = max_abs
    n_vanished  = int(vanished_mask.sum())
    n_active    = int((~vanished_mask).sum())
    active_dims = embed.var_names[~vanished_mask].tolist()

print(f"Aktive Faktoren:   {n_active}")
print(f"Vanished Faktoren: {n_vanished}")

# Top-5 nach reconstruction_effect — wie im PDF
if "reconstruction_effect" in embed.var.columns:
    top5_cols = [c for c in ["reconstruction_effect", "vanished", "order"]
                 if c in embed.var.columns]
    print("\nTop-5 Faktoren nach Rekonstruktionseffekt:")
    print(embed.var.sort_values("reconstruction_effect", ascending=False).head(5)[top5_cols])

# Ranking der aktiven Faktoren
if "order" in embed.var.columns:
    ordered = embed.var[~embed.var["vanished"]].sort_values("order")
else:
    ordered = embed.var[~embed.var["vanished"]].copy()
    ordered["order"] = range(len(ordered))

ranked_active = ordered.index.tolist()
print(f"\nAktive Faktoren (nach Wichtigkeit): {ranked_active}")

# ── 4. plot_latent_dimension_stats ───────────────────────────────────────────

print("\n=== 4. plot_latent_dimension_stats ===")
try:
    drvi_pl.plot_latent_dimension_stats(embed, ncols=2)
    _save_current_figs("latent_dim_stats_all")
except Exception as e:
    print(f"  plot_latent_dimension_stats (alle): {e}")

try:
    drvi_pl.plot_latent_dimension_stats(embed, ncols=2, remove_vanished=True)
    _save_current_figs("latent_dim_stats_active")
except Exception as e:
    print(f"  plot_latent_dimension_stats (remove_vanished): {e}")

# ── 5. plot_latent_dims_in_umap ──────────────────────────────────────────────

print("\n=== 5. plot_latent_dims_in_umap ===")
try:
    drvi_pl.plot_latent_dims_in_umap(embed)
    _save_current_figs("latent_dims_in_umap")
except Exception as e:
    print(f"  plot_latent_dims_in_umap: {e}")

# ── 6. plot_latent_dims_in_heatmap ───────────────────────────────────────────

print("\n=== 6. plot_latent_dims_in_heatmap ===")
try:
    drvi_pl.plot_latent_dims_in_heatmap(embed, args.celltype_key)
    _save_current_figs("latent_dims_heatmap")
except Exception as e:
    print(f"  plot_latent_dims_in_heatmap: {e}")

# ── 7. OOD Interpretability Scores ───────────────────────────────────────────

print("\n=== 7. OOD Interpretability Scores ===")
model.calculate_interpretability_scores(embed, "OOD", directional=True, add_to_counts=1.0)

# Eingebaute Bar-Charts für OOD (wie im PDF)
for ood_key in ["OOD_combined", "OOD_max_possible", "OOD_min_possible"]:
    try:
        model.plot_interpretability_scores(embed, adata_hvg, key=ood_key)
        _save_current_figs(f"interp_scores_{ood_key}")
    except Exception as e:
        print(f"  plot_interpretability_scores({ood_key}): {e}")

# Scores als DataFrame (wie im PDF: model.get_interpretability_scores(embed, adata))
scores_df = model.get_interpretability_scores(embed, adata_hvg)
scores_df.to_csv(os.path.join(args.output_dir, "ood_scores.csv"))
print(f"Gespeichert: ood_scores.csv  ({scores_df.shape[0]} Gene × {scores_df.shape[1]} Faktoren)")
print("Vorschau (erste 5 Gene, erste 10 Faktoren):")
print(scores_df.iloc[:5, :10])

# Top-Gene pro Faktorrichtung
top_genes_per_factor: dict[str, list] = {}
for col in scores_df.columns:
    top_genes_per_factor[col] = scores_df[col].nlargest(args.n_top_genes).index.tolist()

# Übersicht Top-Gene für aktive Faktoren
print(f"\nTop-{args.n_top_genes} Gene pro aktivem Faktor:")
print("-" * 60)
for dim in ranked_active[:10]:
    num = dim.replace("DR", "")
    for direction in ["+", "-"]:
        col = f"DR {num}{direction}"
        if col in top_genes_per_factor:
            genes = top_genes_per_factor[col]
            print(f"  {col:<10}: {', '.join(str(g) for g in genes[:5])}")

# ── 8. Top-Gene pro Faktor auf DRVI-UMAP ─────────────────────────────────────

print("\n=== 8. Top-Gene auf DRVI-UMAP (pro Faktor) ===")
# PDF: adata.obsm['X_drvi_umap'] = embed[adata.obs.index].obsm['X_umap']  (bereits in Step 2)

for dim in ranked_active[:8]:
    num = dim.replace("DR", "")

    for direction in ["+", "-"]:
        dim_title = f"DR {num}{direction}"
        if dim_title not in top_genes_per_factor:
            continue

        top_genes     = top_genes_per_factor[dim_title][:4]
        genes_in_data = [g for g in top_genes if g in adata.var_names]
        if not genes_in_data:
            continue

        safe = dim_title.replace(" ", "").replace("+", "pos").replace("-", "neg")

        # Faktor-Aktivität auf embed-UMAP (wie im PDF)
        try:
            drvi_pl.plot_latent_dims_in_umap(embed, dim_subset=[dim_title])
            _save_current_figs(f"factor_{safe}_embed_umap")
        except Exception as e:
            print(f"  plot_latent_dims_in_umap({dim_title}): {e}")

        # Top-Gene auf DRVI-UMAP (sc.pl.embedding wie im PDF)
        try:
            sc.pl.embedding(adata, "X_drvi_umap", color=genes_in_data, show=False)
            plt.savefig(
                os.path.join(args.output_dir, f"factor_{safe}_top_genes.png"),
                bbox_inches="tight", dpi=100,
            )
            plt.close("all")
        except Exception as e:
            print(f"  sc.pl.embedding({dim_title}): {e}")

    print(f"  Faktor {dim}: Plots gespeichert")

# ── 9. IND Interpretability Scores ───────────────────────────────────────────

print("\n=== 9. IND Interpretability Scores (Within-Distribution) ===")
model.calculate_interpretability_scores(embed, "IND")

try:
    model.plot_interpretability_scores(embed, adata_hvg, key="IND_linear_weighted_mean")
    _save_current_figs("interp_scores_IND_linear_weighted_mean")
except Exception as e:
    print(f"  plot_interpretability_scores(IND): {e}")

scores_df_ind = model.get_interpretability_scores(
    embed, adata_hvg, key="IND_linear_weighted_mean"
)
scores_df_ind.to_csv(os.path.join(args.output_dir, "ind_scores.csv"))
print(f"Gespeichert: ind_scores.csv  ({scores_df_ind.shape[0]} Gene × {scores_df_ind.shape[1]} Faktoren)")
print("Vorschau (erste 5 Gene, erste 10 Faktoren):")
print(scores_df_ind.iloc[:5, :10])

# ── 10. Identifikation von Programmen: g:Profiler Enrichment ─────────────────

print(f"\n=== 10. Identifikation von Programmen (g:Profiler, Top-{args.n_gprofiler} Faktoren) ===")

top_factor_dirs: list[str] = []
for dim in ranked_active:
    num = dim.replace("DR", "")
    for direction in ["+", "-"]:
        col = f"DR {num}{direction}"
        if col in top_genes_per_factor:
            top_factor_dirs.append(col)
    if len(top_factor_dirs) >= args.n_gprofiler * 2:
        break

enrichment_results: dict = {}
for factor in top_factor_dirs[:args.n_gprofiler * 2]:
    genes = top_genes_per_factor[factor]
    if len(genes) < 3:
        continue
    try:
        resp = requests.post(
            "https://biit.cs.ut.ee/gprofiler/api/gost/profile/",
            json={
                "organism":     "hsapiens",
                "query":        genes,
                "sources":      ["GO:BP", "GO:MF", "KEGG", "REAC"],
                "user_threshold": 0.05,
                "significance_threshold_method": "fdr",
                "no_evidences": True,
            },
            timeout=30,
        )
        if resp.status_code == 200:
            results = resp.json().get("result", [])
            if results:
                top_terms = sorted(results, key=lambda x: x.get("p_value", 1))[:5]
                enrichment_results[factor] = top_terms
                print(f"\n  {factor}  (Gene: {genes[:5]})")
                for t in top_terms:
                    print(f"    [{t['source']}] {t['name'][:60]}  p={t['p_value']:.2e}")
            else:
                print(f"  {factor}: kein signifikanter Term")
        else:
            print(f"  {factor}: g:Profiler Fehler {resp.status_code}")
    except Exception as e:
        print(f"  {factor}: Fehler ({e})")

if enrichment_results:
    rows = []
    for factor, terms in enrichment_results.items():
        for t in terms:
            rows.append({
                "factor":            factor,
                "source":            t.get("source"),
                "term_id":           t.get("native"),
                "term_name":         t.get("name"),
                "p_value":           t.get("p_value"),
                "term_size":         t.get("term_size"),
                "intersection_size": t.get("intersection_size"),
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(args.output_dir, "gprofiler_enrichment.csv"), index=False
    )
    print("\nGespeichert: gprofiler_enrichment.csv")

# ── 11. Faktor-Spezifität + Validierung mit Marker-Genen ─────────────────────

print("\n=== 11. Faktor-Spezifität + Validierung ===")

# Faktorwerte in adata.obs kopieren (für Leiden-Coverage-Analyse)
latent_vals = embed.X if not hasattr(embed.X, "toarray") else embed.X.toarray()
for dim in active_dims:
    col_idx = embed.var_names.tolist().index(dim)
    adata.obs[f"DRVI_{dim}"] = latent_vals[:, col_idx]

# 11a. Faktor-Spezifität (Leiden-Cluster Coverage)
if args.leiden_key in adata.obs.columns:
    specificity: dict = {}
    n_clusters_total = adata.obs[args.leiden_key].nunique()
    for dim in active_dims:
        col  = f"DRVI_{dim}"
        vals = adata.obs[col].values
        q75  = np.percentile(vals, 75)
        high = adata.obs.loc[vals > q75, args.leiden_key]
        specificity[dim] = high.nunique() / n_clusters_total

    spec_df = (
        pd.DataFrame.from_dict(specificity, orient="index",
                               columns=["fraction_clusters_high"])
        .sort_values("fraction_clusters_high")
    )
    spec_df.to_csv(os.path.join(args.output_dir, "factor_specificity.csv"))

    celltype_factors = spec_df[spec_df["fraction_clusters_high"] <= 0.25].index.tolist()
    shared_factors   = spec_df[spec_df["fraction_clusters_high"] >  0.50].index.tolist()
    print(f"Zelltyp-spezifische Faktoren (≤25% Cluster): {celltype_factors}")
    print(f"Geteilte Prozesse      (>50% Cluster):        {shared_factors}")
    print("Gespeichert: factor_specificity.csv")

# 11b. Validierung: bekannte Marker außerhalb der HVGs
validation_markers = {
    "T-Zellen":       ["CD3D", "CD3E", "CD3G"],
    "B-Zellen":       ["CD19", "MS4A1", "CD79A"],
    "Monozyten":      ["CD14", "LYZ", "FCGR3A"],
    "NK-Zellen":      ["NCAM1", "GNLY", "NKG7"],
    "Fibroblasten":   ["COL1A1", "COL1A2", "DCN"],
    "Megakaryozyten": ["PF4", "PPBP", "GP1BB"],
}

non_hvg_genes: list[str] = []
print("\nMarker-Gen Status (HVG / Nicht-HVG / Fehlt im Datensatz):")
for celltype, genes in validation_markers.items():
    in_hvg      = [g for g in genes if g in adata_hvg.var_names]
    not_in_hvg  = [g for g in genes if g in adata.var_names and g not in adata_hvg.var_names]
    not_in_data = [g for g in genes if g not in adata.var_names]
    non_hvg_genes.extend(not_in_hvg)
    print(f"  {celltype:<15}: HVG={in_hvg}  Nicht-HVG={not_in_hvg}  Fehlt={not_in_data}")

if non_hvg_genes:
    adata.X = adata.layers["counts"]
    ncols = min(4, len(non_hvg_genes))
    nrows = (len(non_hvg_genes) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes_flat = np.array(axes).flatten() if nrows * ncols > 1 else [axes]
    for i, gene in enumerate(non_hvg_genes):
        sc.pl.umap(adata, color=gene, ax=axes_flat[i], show=False,
                   title=f"{gene} (nicht HVG)", color_map="RdBu_r", use_raw=False)
    for ax in axes_flat[len(non_hvg_genes):]:
        ax.set_visible(False)
    plt.suptitle("Validierung: Marker außerhalb der HVGs", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "validation_non_hvg_markers.png"),
                bbox_inches="tight", dpi=100)
    plt.close("all")
    print("Gespeichert: validation_non_hvg_markers.png")
else:
    print("Alle Validierungsmarker sind entweder HVGs oder nicht im Datensatz.")

# ── Zusammenfassung ───────────────────────────────────────────────────────────

print("\n=== Zusammenfassung ===")
summary = {
    "n_latent_total":    embed.n_vars,
    "n_active_factors":  n_active,
    "n_vanished_factors": n_vanished,
    "active_factors":    ", ".join(active_dims),
}
pd.Series(summary).to_csv(
    os.path.join(args.output_dir, "interpretation_summary.csv"), header=False
)
print(pd.Series(summary).to_string())
print(f"\nFertig. Ergebnisse in: {args.output_dir}")
