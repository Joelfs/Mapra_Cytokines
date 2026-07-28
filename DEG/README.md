# Differential Gene Expression (DEG)

Pseudobulk DEG analysis following the [sc-best-practices tutorial](https://www.sc-best-practices.org/conditions/differential_gene_expression.html), using `decoupler` for pseudobulk aggregation and gene filtering, then `PyDESeq2` for testing.

## Contents

- `deg_conditions.ipynb` — main pipeline: pseudobulk aggregation (sum) → sample QC → gene filtering → one PyDESeq2 model per cell type → three comparisons, repeated across four cell-type groupings (Harmony/DRVI, named cell types + raw Leiden clusters). All plots (volcano, heatmap, fold-change, paired expression, pathway enrichment) are embedded as cell outputs.
- `cohort_overview_qc.ipynb` — cohort/sample overview and DEG-readiness QC (sampling completeness, cell-type composition, pseudobulk sample-size filtering, PCA of pseudobulks, batch confound check).
- `deg_results_slides.pptx` — presentation summarizing the full analysis and results.
- `deg_metadata_shared.parquet` — per-cell metadata table (patient ID, condition, timepoint, Scanorama/Harmony/DRVI cell-type labels) for reuse without re-deriving.

## Comparisons run

Cohort design follows Pekayvaz et al. 2024 (*Nat Med*); the DESeq2 pseudobulk analysis itself is independent of the paper's own MOFA-based approach.

1. **ACS vs CCS** — sterile ACS, all timepoints (TP1M–TP4M) vs CCS, blocked by `patient_id`
2. **ACS vs non-CCS** — sterile ACS (TP1M only) vs non-CCS + Sclerosis, pooled as Control
3. **Longitudinal** — sterile ACS, TP1M vs TP4M, paired within-patient

## Headline results (padj < 0.05, |log2FC| > 1)

| Grouping | ACS vs CCS | ACS vs non-CCS | Longitudinal |
|---|---|---|---|
| Harmony_leiden (25 clusters) | 264 | 83 | 173 |
| Harmony_named (Scanorama labels) | 351 | 57 | 128 |
| DRVI_leiden (32 clusters) | 333 | 70 | 147 |
| DRVI_named (Scanorama labels) | 339 | 60 | 128 |
