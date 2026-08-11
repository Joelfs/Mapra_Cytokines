#!/usr/bin/env python3
"""
DRVI: Disentangled Representation Variational Inference

Learns batch-corrected representations from raw scRNA-seq data (raw counts).
Serves as an alternative to PCA + Harmony and produces its own UMAP + Leiden.

Usage:
    conda run -n mapra_cytokines python drvi_analysis.py \
        --input  .../dimensionality_reduction.h5ad \
        --output-dir .../drvi_results
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import anndata as ad
from drvi.model import DRVI

warnings.filterwarnings("ignore")

# ── Argparse ──────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    default="/vol/disk/ubuntu/master_practicum_cytokines/data/data_for_practicum_preprocessed.h5ad",
)
parser.add_argument(
    "--output-dir",
    default="/vol/disk/ubuntu/master_practicum_cytokines/joel/workdir/drvi_results_prep_by_lisa",
)
parser.add_argument("--hvg-col",      default="highly_variable",
                    help="adata.var column used for HVG selection (highly_deviant, hvg_seurat_v3, …)")
parser.add_argument("--n-latent",     type=int,   default=64)
parser.add_argument("--max-epochs",   type=int,   default=400)
parser.add_argument("--leiden-res",   type=float, default=0.5)
parser.add_argument("--batch-key",    default="library")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
sc.settings.figdir   = args.output_dir
sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=100, facecolor="white", frameon=False)

# ── 1. Load data ────────────────────────────────────────────────────────────

print("=== 1. Load data ===")
adata = sc.read_h5ad(args.input)
print(adata)

if "counts" not in adata.layers:
    raise ValueError("adata.layers['counts'] missing — please use dimensionality_reduction.h5ad")
if args.hvg_col not in adata.var.columns:
    raise ValueError(f"adata.var['{args.hvg_col}'] missing — check the HVG column")

# ── 2. Restrict to HVGs ───────────────────────────────────────────────────────

print(f"\n=== 2. Subselect HVGs ({args.hvg_col}) ===")
adata_hvg = adata[:, adata.var[args.hvg_col]].copy()
print(f"Genes after subselect: {adata_hvg.n_vars}")

# ── 3. DRVI setup ─────────────────────────────────────────────────────────────

print(f"\n=== 3. DRVI Setup  (n_latent={args.n_latent}, batch={args.batch_key}) ===")
DRVI.setup_anndata(
    adata_hvg,
    layer="counts",
    categorical_covariate_keys=[args.batch_key],
    is_count_data=True,
)

model = DRVI(
    adata_hvg,
    n_latent=args.n_latent,
    encoder_dims=[128, 128],
    decoder_dims=[128, 128],
)
print(model)

# ── 4. Training ───────────────────────────────────────────────────────────────

print(f"\n=== 4. Training  (max_epochs={args.max_epochs}) ===")
model.train(
    max_epochs=args.max_epochs,
    early_stopping=False,
)

# Plot training loss
if hasattr(model, "history") and "elbo_train" in model.history:
    fig, ax = plt.subplots(figsize=(7, 3))
    history = model.history
    train_loss = history["elbo_train"].values.flatten()
    ax.plot(train_loss, label="Train ELBO")
    if "elbo_validation" in history:
        val_loss = history["elbo_validation"].values.flatten()
        ax.plot(val_loss, label="Val ELBO")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ELBO")
    ax.legend()
    plt.title("DRVI Training Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "drvi_training_loss.png"),
                bbox_inches="tight", dpi=120)
    plt.close()
    print("Saved: drvi_training_loss.png")

# ── 5. Latent representation ─────────────────────────────────────────────────

print("\n=== 5. Latent representation ===")
latent = model.get_latent_representation()
print(f"Latent shape: {latent.shape}")
adata.obsm["X_drvi"] = latent

# ── 6. Neighbors + UMAP on the DRVI embedding ─────────────────────────────────

print("\n=== 6. Neighbors + UMAP (DRVI) ===")
sc.pp.neighbors(adata, use_rep="X_drvi", key_added="neighbors_drvi")
sc.tl.umap(adata, neighbors_key="neighbors_drvi")
adata.obsm["X_umap_drvi"] = adata.obsm["X_umap"].copy()
print("UMAP computed → X_umap_drvi")

# ── 7. Leiden on the DRVI embedding ───────────────────────────────────────────

print(f"\n=== 7. Leiden  (res={args.leiden_res}) ===")
sc.tl.leiden(
    adata,
    neighbors_key="neighbors_drvi",
    key_added="leiden_drvi",
    resolution=args.leiden_res,
    flavor="igraph",
    n_iterations=2,
    directed=False,
)
n_clusters = adata.obs["leiden_drvi"].nunique()
print(f"Clusters (DRVI): {n_clusters}")

# ── 8. Visualizations ──────────────────────────────────────────────────────────

print("\n=== 8. Visualizations ===")

# UMAP: library (check batch correction)
sc.pl.embedding(
    adata, basis="umap_drvi", color=args.batch_key,
    title=f"DRVI UMAP — {args.batch_key}",
    save="_drvi_library.png", show=False,
)
print("Saved: umap_drvi_library.png")

# UMAP: Leiden
sc.pl.embedding(
    adata, basis="umap_drvi", color="leiden_drvi",
    legend_loc="on data",
    title=f"DRVI UMAP — Leiden (res={args.leiden_res}, {n_clusters} clusters)",
    save="_drvi_leiden.png", show=False,
)
print("Saved: umap_drvi_leiden.png")

# Comparison: PCA-UMAP vs DRVI-UMAP (if available)
pca_umap_key = "X_umap_deviance"
if pca_umap_key in adata.obsm:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    adata.obsm["X_umap"] = adata.obsm[pca_umap_key]
    sc.pl.umap(adata, color=args.batch_key, ax=axes[0], show=False,
               title=f"PCA-UMAP (Deviance HVGs) — {args.batch_key}")

    adata.obsm["X_umap"] = adata.obsm["X_umap_drvi"]
    sc.pl.umap(adata, color=args.batch_key, ax=axes[1], show=False,
               title=f"DRVI-UMAP — {args.batch_key}")

    plt.suptitle("PCA vs. DRVI: batch correction", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "comparison_pca_vs_drvi_library.png"),
                bbox_inches="tight", dpi=120)
    plt.close()
    print("Saved: comparison_pca_vs_drvi_library.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    adata.obsm["X_umap"] = adata.obsm[pca_umap_key]
    leiden_pca = "leiden" if "leiden" in adata.obs.columns else None
    if leiden_pca:
        sc.pl.umap(adata, color=leiden_pca, ax=axes[0], show=False,
                   legend_loc="on data", title="PCA-UMAP — Leiden")
    adata.obsm["X_umap"] = adata.obsm["X_umap_drvi"]
    sc.pl.umap(adata, color="leiden_drvi", ax=axes[1], show=False,
               legend_loc="on data", title="DRVI-UMAP — Leiden")
    plt.suptitle("PCA vs. DRVI: cluster structure", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "comparison_pca_vs_drvi_leiden.png"),
                bbox_inches="tight", dpi=120)
    plt.close()
    print("Saved: comparison_pca_vs_drvi_leiden.png")

    # Reset the active UMAP back to DRVI
    adata.obsm["X_umap"] = adata.obsm["X_umap_drvi"]

# ── 9. Plot latent dimensions (top 6) ─────────────────────────────────────────

print("\n=== 9. Latent dimensions ===")
n_show = min(6, args.n_latent)
latent_df = pd.DataFrame(
    latent[:, :n_show],
    index=adata.obs_names,
    columns=[f"DRVI_{i+1}" for i in range(n_show)],
)
for col in latent_df.columns:
    adata.obs[col] = latent_df[col].values

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes_flat = axes.flatten()
for i, ax in enumerate(axes_flat):
    col = f"DRVI_{i+1}"
    sc.pl.embedding(adata, basis="umap_drvi", color=col,
                    ax=ax, show=False, title=col,
                    color_map="RdBu_r", vcenter=0)
plt.suptitle("DRVI: first 6 latent dimensions", fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "drvi_latent_dims.png"),
            bbox_inches="tight", dpi=100)
plt.close()
print("Saved: drvi_latent_dims.png")

# ── 10. Save model + data ──────────────────────────────────────────────────────

print("\n=== 10. Save ===")

model_path = os.path.join(args.output_dir, "drvi_model")
model.save(model_path, overwrite=True)
print(f"Model saved: {model_path}/")

out_path = os.path.join(args.output_dir, "drvi_representation.h5ad")
adata.write_h5ad(out_path)
print(f"AnnData saved: {out_path}")
print(f"  obsm: X_drvi ({args.n_latent}D), X_umap_drvi")
print(f"  obs:  leiden_drvi")

print(f"\nDone. Results in: {args.output_dir}")
