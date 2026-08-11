#!/usr/bin/env python3
"""
Creates an annotated/cleaned copy of
data_for_practicum_post_integration.h5ad, based on the DRVI factor
threshold candidates from 2_7_a/2_7_c:

Newly annotated (new column `cell_type_curated`, original `cell_type_Scanorama`
kept unchanged):
  - DR9-  -> "Tregs"
  - DR17+ -> "pDC"
  - DR28+ -> "Plasma cells"
  (DR4+ / platelets deliberately NOT annotated — instead removed via PF4
   gene expression, see below.)

Removed (suspected doublets/batch artifacts):
  - DR21+ candidates within B cells (NK signal in B cells ->
    suspected B/NK doublets)
  - DR35+ candidates within Monocytes-CD14/Dendritic/Monocytes-CD16_FCGR3A
    (B-cell signal in myeloid cells -> suspected B/myeloid doublets)
  - All cells from candidate_cells_gene_IGHG3_in_Bcell.csv (IGHG3+ B cells,
    almost exclusively driven by one donor (k28) -> suspected
    batch artifact)
  - Entire Leiden cluster 13 (drvi_leiden, res=1.0): 363 cells, 361 of them
    B cells, 83.5% from a single donor (k11) -> suspected batch
    artifact (markers: ABCA6, FCRL2, FCER2, CLNK, TCF4, IGKC — same
    signature found in the B-cell subclustering check, see 2_8_c)
  - PF4+ cells (gene expression, aggressive manual threshold=0.12, instead
    of the data-driven detected valley at 0.95 — 0.12 and 0.2 yield almost
    identical candidate counts (4972 vs. 4966), the gap between them is
    essentially empty -> that is the actual valley floor) -> spread across
    all cell types (no donor dominance), suspected platelet/ambient-RNA
    contamination, see candidate_cells_gene_PF4_in_all.csv.
    Megakaryocytes are excluded from this (genuine, high PF4 expression is
    correct biology for this cell type, not contamination).

Usage:
    conda run -n mapra_cytokines python 2_8_a_drvi_create_annotated_copy.py
"""

import os

import numpy as np
import scanpy as sc
import pandas as pd

DRVI_INPUT = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/data/data_for_practicum_post_integration.h5ad"
OUTPUT_DIR = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/visualization/drvi_interpretation"
OUT_H5AD = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/data/data_for_practicum_post_integration_curated.h5ad"
CELLTYPE_KEY = "cell_type_Scanorama"

ANNOTATIONS = {
    "candidate_cells_DR9neg.csv": "Tregs",
    "candidate_cells_DR17pos.csv": "pDC",
    "candidate_cells_DR28pos.csv": "Plasma cells",
}
# These CSVs are already filtered to the respective cell type (from
# 2_7_a --celltype-filter or 2_7_c), so a plain union without re-masking by
# cell type is sufficient.
REMOVE_CSVS = {
    "candidate_cells_DR21pos_in_Bcell.csv": "DR21+ in B-cell (B/NK doublets)",
    "candidate_cells_DR35pos_in_MonocytesCD14_Dendritic_MonocytesCD16FCGR3A.csv":
        "DR35+ in Monocytes-CD14/Dendritic/Monocytes-CD16_FCGR3A (B/myeloid doublets)",
    "candidate_cells_gene_IGHG3_in_Bcell.csv": "IGHG3+ in B-cell (k28 mini-cluster)",
    "candidate_cells_gene_PF4_in_all.csv": "PF4+ (threshold=0.12, suspected platelets/ambient RNA)",
}
# Cell types excluded from certain removal reasons because the signal there
# is genuine biology rather than an artifact.
REMOVE_EXCEPTIONS = {
    "candidate_cells_gene_PF4_in_all.csv": ["Megakaryocytes"],
}
LEIDEN_KEY = "drvi_leiden"
REMOVE_LEIDEN_CLUSTERS = ["13"]  # k11-dominated B-cell mini-cluster, see 2_8_c


def load_barcodes(fname: str) -> set:
    path = os.path.join(OUTPUT_DIR, fname)
    return set(pd.read_csv(path)["cell_barcode"])


print("=== Load data ===")
adata = sc.read_h5ad(DRVI_INPUT)
print(f"{adata.n_obs} cells, {adata.n_vars} genes")

# ── Re-annotation ─────────────────────────────────────────────────────────────

print("\n=== Re-annotation ===")
new_celltype = adata.obs[CELLTYPE_KEY].astype(str).copy()

for fname, label in ANNOTATIONS.items():
    barcodes = load_barcodes(fname)
    mask = adata.obs_names.isin(barcodes)
    print(f"{fname} -> '{label}': {mask.sum()} cells "
          f"(before: {new_celltype[mask].value_counts().to_dict()})")
    new_celltype[mask] = label

adata.obs["cell_type_curated"] = pd.Categorical(new_celltype)
print("\ncell_type_curated distribution:")
print(adata.obs["cell_type_curated"].value_counts())

# ── Removal of suspected doublets/batch artifacts ────────────────────────────

print("\n=== Removal ===")

remove_masks = {}
for fname, desc in REMOVE_CSVS.items():
    barcodes = load_barcodes(fname)
    mask = adata.obs_names.isin(barcodes)
    exceptions = REMOVE_EXCEPTIONS.get(fname)
    if exceptions:
        is_exception = adata.obs[CELLTYPE_KEY].isin(exceptions).values
        n_spared = int((mask & is_exception).sum())
        mask = mask & ~is_exception
        print(f"  ({n_spared} cells from {exceptions} excluded from '{desc}')")
    remove_masks[desc] = mask
    print(f"{desc}: {mask.sum()} cells")

leiden_desc = f"{LEIDEN_KEY} in {REMOVE_LEIDEN_CLUSTERS} (k11 mini-cluster)"
leiden_mask = adata.obs[LEIDEN_KEY].astype(str).isin(REMOVE_LEIDEN_CLUSTERS).values
remove_masks[leiden_desc] = leiden_mask
print(f"{leiden_desc}: {leiden_mask.sum()} cells")

remove_mask = np.zeros(adata.n_obs, dtype=bool)
for mask in remove_masks.values():
    remove_mask |= mask

descs = list(remove_masks.keys())
for i in range(len(descs)):
    for j in range(i + 1, len(descs)):
        overlap = int((remove_masks[descs[i]] & remove_masks[descs[j]]).sum())
        if overlap:
            print(f"Overlap '{descs[i]}' & '{descs[j]}': {overlap}")

print(f"Total to remove: {int(remove_mask.sum())} cells "
      f"({100*remove_mask.sum()/adata.n_obs:.2f}% of all cells)")

# Sanity check: does this also remove newly annotated cells (Tregs/pDC/Plasma)?
for fname, label in ANNOTATIONS.items():
    barcodes = load_barcodes(fname)
    mask = adata.obs_names.isin(barcodes)
    overlap = int((mask & remove_mask).sum())
    if overlap:
        print(f"  WARNING: {overlap} '{label}' cells are also on the removal list!")

adata_curated = adata[~remove_mask].copy()
print(f"\nRemaining cells: {adata_curated.n_obs} / {adata.n_obs}")

# ── Save ────────────────────────────────────────────────────────────────────

print(f"\n=== Save ===")
adata_curated.write_h5ad(OUT_H5AD)
print(f"Saved: {OUT_H5AD}")
print(f"\ncell_type_curated distribution (final):")
print(adata_curated.obs["cell_type_curated"].value_counts())
