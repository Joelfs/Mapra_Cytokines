# Differential Gene Expression (DEG)

Pseudobulk DEG analysis following the [sc-best-practices tutorial](https://www.sc-best-practices.org/conditions/differential_gene_expression.html), using `decoupler` for pseudobulk aggregation and gene filtering, then `PyDESeq2` for testing.

This folder is set up so someone else can pick up the analysis without re-deriving anything already done.

## Contents

- `deg_conditions.ipynb` — main pipeline: pseudobulk aggregation (sum) → sample QC → gene filtering → one PyDESeq2 model per cell type → four comparisons, each repeated across all four cell-type groupings (Harmony/DRVI × named Scanorama-transferred cell types / raw Leiden clusters, both at resolution 1.0 — the h5ad's own `uns['harmony_leiden']['params']` / `uns['drvi_leiden']['params']` confirm this, it's the same clustering used everywhere in this folder). Rerun this to reproduce everything in `results/`.
- `replot_timepoint_vs_ccs.py` — regenerates the heatmap + volcano plots for Comparison 4 (ACS-per-timepoint vs CCS), all four groupings, in a different plot style than the rest of the notebook (matching `sudenaz/fix_and_replot.py`: faceted per-cluster volcano grid, "Cell Type"-labeled heatmap with non-symmetric colorbar, clusters-with-no-hits dropped). Reads the CSVs `deg_conditions.ipynb` already saved — does not rerun DESeq2. Run this after the notebook if Comparison 4's plots need regenerating.
- `dice_score_harmony_vs_drvi.py` — Dice similarity between `cell_type_Harmony` and `cell_type_DRVI` (both majority-vote label transfers from `cell_type_Scanorama`, onto different clustering solutions), overall and within each of the four ACS-timepoint-vs-CCS populations. Quantifies how much the two integration methods actually agree on cell identity — see `dice_score_Harmony_vs_DRVI_*` outputs.
- `cohort_overview_qc.ipynb` — cohort/sample overview and DEG-readiness QC (sampling completeness, cell-type composition, pseudobulk sample-size filtering, PCA of pseudobulks, batch confound check). Rerun this to reproduce `qc_panels/`.
- `deg_metadata_shared.parquet` — per-cell metadata table (patient ID, condition, timepoint, Scanorama/Harmony/DRVI cell-type labels). Join onto `adata.obs` to skip re-deriving cell-type labels or timepoint/condition parsing.
- `results/` — full gene-level DEG result tables (one CSV per grouping × comparison, all genes, not just significant ones) plus every plot generated from them (volcano, heatmap, fold-change, paired expression, pathway enrichment, pseudobulk QC).
- `qc_panels/` — cohort/sample-level QC plots (patient×timepoint completeness, cell-type composition per sample, pseudobulk sample-size filtering, PCA of pseudobulks + metadata correlation, batch×condition crosstab, per-sample QC metrics).

## Comparisons run

Cohort design follows Pekayvaz et al. 2024 (*Nat Med*); the DESeq2 pseudobulk analysis itself is independent of the paper's own MOFA-based approach.

1. **ACS vs CCS** — sterile ACS, all timepoints (TP1M–TP4M) vs CCS, blocked by `patient_id`
2. **ACS vs non-CCS** — sterile ACS (TP1M only) vs non-CCS + Sclerosis, pooled as Control
3. **Longitudinal** — sterile ACS, TP1M vs TP4M, paired within-patient
4. **ACS-per-timepoint vs CCS** — sterile ACS at a single timepoint (TP1M, TP2M, TP3M, or TP4M in turn) vs CCS, one comparison per timepoint. Unlike Comparison 1, each ACS patient contributes only one pseudobulk sample here, so no `patient_id` blocking covariate is needed — this also sidesteps the rank-deficiency issue noted below for Comparison 1. Shows whether the ACS-vs-CCS signal is stable across the post-event timeline or concentrated at particular timepoints. Run for all four groupings, so leiden-cluster and Scanorama-named results are directly comparable at each timepoint.

## Headline results (padj < 0.05, |log2FC| > 1)

| Grouping | ACS vs CCS | ACS vs non-CCS | Longitudinal | TP1M vs CCS | TP2M vs CCS | TP3M vs CCS | TP4M vs CCS |
|---|---|---|---|---|---|---|---|
| Harmony_leiden (25 clusters) | 260 | 51 | 157 | 105 | 161 | 37 | 136 |
| Harmony_named (Scanorama labels, 7 testable) | 290 | 57 | 110 | 108 | 195 | 54 | 153 |
| DRVI_leiden (32 clusters) | 378 | 52 | 152 | 136 | 237 | 39 | 192 |
| DRVI_named (Scanorama labels, 7 testable) | 344 | 58 | 127 | 126 | 213 | 49 | 182 |

Re-run in full 2026-08-07. Leiden consistently detects more significant genes than the matching named grouping (e.g. DRVI_leiden vs DRVI_named: 378 vs 344 overall, 136 vs 126 at TP1M, 237 vs 213 at TP2M) — expected, since raw clusters are finer-grained and more transcriptionally homogeneous than the 7 broad named types, giving DESeq2 more power per comparison, at the cost of testing more clusters (25–32 vs 7) and a correspondingly larger multiple-testing burden. Not a sign either grouping is wrong — see `dice_score_Harmony_vs_DRVI_*` for how much the two labelings actually agree cell-by-cell.

`results/{grouping}_{comparison}.csv` columns: `gene`, `baseMean`, `log2FoldChange`, `lfcSE`, `stat`, `pvalue`, `padj`, `cell_type`, `significant`.

## Heatmaps and volcano plots

- **Comparisons 1–3** (`heatmap_{grouping}_{comparison}.png`, `volcano_{grouping}_{comparison}.png`, all four groupings): top 5 most upregulated + top 5 most downregulated significant genes per cluster (ranked by log2FoldChange, not padj), log2FC colored `RdBu_r`, symmetric colorbar; volcano is a faceted grid, one panel per cluster, with the top genes by padj labeled.
- **Comparison 4** (all four groupings): same top-5-up/top-5-down selection, but plotted by `replot_timepoint_vs_ccs.py` in a different style — non-symmetric colorbar (`center=0`, auto-scaled to the data), clusters with no significant hits dropped from the heatmap entirely, and volcano panels titled `Cluster {name}` with `up (N)`/`down (N)` legend counts. Matches the look of `sudenaz/fix_and_replot.py`, used for the ACS-subgroup analysis. For the leiden groupings, "cluster name" is the raw numeric cluster ID (e.g. `0`–`24` for Harmony_leiden), not a cell type label.

## Known limitations / open items for whoever continues this

- **No demographic covariates available**: no age, sex, BMI, or medication data exists anywhere in the source object or the wider project — checked directly, not just missing from this analysis. Can't rule out demographic imbalance contributing to some DEG signal.
- **Batch confound**: libraries L11 (100% CCS) and L13 (100% non-CCS) are condition-pure — batch and biology are confounded for those specific samples. See `qc_panels/panel6_batch_condition_crosstab.png`.
- **ACS vs CCS design is not fully identifiable**: the `~ patient_id + condition` design for this comparison is rank-deficient by one degree of freedom, since no patient appears in both conditions (verified directly by building the design matrix and checking its rank). PyDESeq2 fits it without erroring, but the patient-blocking isn't as clean here as it is for the longitudinal comparison, where every patient does cross both levels. A cleaner fix would be restricting this comparison to one timepoint per patient (as comparison 2 already does), which would change the result and hasn't been done.
- **Untestable cell types excluded**: Progenitor and Megakaryocytes have too few cells (243 and 60 total, respectively, out of 109,504) to build reliable pseudobulk samples — excluded from all comparisons, not a bug.
- **Sum vs mean aggregation checked, not just assumed**: mean-aggregated pseudobulk was tested directly (same comparison, same data) and produces 0 significant genes, since averaging across cells produces fractional counts too small for DESeq2's count-based model. Sum is the correct choice; see `results/Harmony_named_ACS_vs_CCS.csv` (sum) for the real result.
- **Next natural step**: pathway enrichment (`dc.mt.ora` against MSigDB Hallmark gene sets) has only been run for two examples (Monocytes-CD14 and NK, ACS vs CCS) — see `results/enrichment_Harmony_named_*`. Extending this across all three comparisons and more cell types would move from gene lists to biological interpretation.
