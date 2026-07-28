# Differential Gene Expression (DEG)

Pseudobulk DEG analysis following the [sc-best-practices tutorial](https://www.sc-best-practices.org/conditions/differential_gene_expression.html), using `decoupler` for pseudobulk aggregation and gene filtering, then `PyDESeq2` for testing.

This folder is set up so someone else can pick up the analysis without re-deriving anything already done.

## Contents

- `deg_conditions.ipynb` — main pipeline: pseudobulk aggregation (sum) → sample QC → gene filtering → one PyDESeq2 model per cell type → three comparisons, repeated across four cell-type groupings (Harmony/DRVI, named cell types + raw Leiden clusters). Rerun this to reproduce everything in `results/`.
- `cohort_overview_qc.ipynb` — cohort/sample overview and DEG-readiness QC (sampling completeness, cell-type composition, pseudobulk sample-size filtering, PCA of pseudobulks, batch confound check). Rerun this to reproduce `qc_panels/`.
- `deg_metadata_shared.parquet` — per-cell metadata table (patient ID, condition, timepoint, Scanorama/Harmony/DRVI cell-type labels). Join onto `adata.obs` to skip re-deriving cell-type labels or timepoint/condition parsing.
- `results/` — full gene-level DEG result tables (one CSV per grouping × comparison, all genes, not just significant ones) plus every plot generated from them (volcano, heatmap, fold-change, paired expression, pathway enrichment, pseudobulk QC).
- `qc_panels/` — cohort/sample-level QC plots (patient×timepoint completeness, cell-type composition per sample, pseudobulk sample-size filtering, PCA of pseudobulks + metadata correlation, batch×condition crosstab, per-sample QC metrics).

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

`results/{grouping}_{comparison}.csv` columns: `gene`, `baseMean`, `log2FoldChange`, `lfcSE`, `stat`, `pvalue`, `padj`, `cell_type`, `significant`.

## Known limitations / open items for whoever continues this

- **No demographic covariates available**: no age, sex, BMI, or medication data exists anywhere in the source object or the wider project — checked directly, not just missing from this analysis. Can't rule out demographic imbalance contributing to some DEG signal.
- **Batch confound**: libraries L11 (100% CCS) and L13 (100% non-CCS) are condition-pure — batch and biology are confounded for those specific samples. See `qc_panels/panel6_batch_condition_crosstab.png`.
- **ACS vs CCS design is not fully identifiable**: the `~ patient_id + condition` design for this comparison is rank-deficient by one degree of freedom, since no patient appears in both conditions (verified directly by building the design matrix and checking its rank). PyDESeq2 fits it without erroring, but the patient-blocking isn't as clean here as it is for the longitudinal comparison, where every patient does cross both levels. A cleaner fix would be restricting this comparison to one timepoint per patient (as comparison 2 already does), which would change the result and hasn't been done.
- **Untestable cell types excluded**: Progenitor and Megakaryocytes have too few cells (243 and 60 total, respectively, out of 109,504) to build reliable pseudobulk samples — excluded from all comparisons, not a bug.
- **Sum vs mean aggregation checked, not just assumed**: mean-aggregated pseudobulk was tested directly (same comparison, same data) and produces 0 significant genes, since averaging across cells produces fractional counts too small for DESeq2's count-based model. Sum is the correct choice; see `results/Harmony_named_ACS_vs_CCS.csv` (sum) for the real result.
- **Next natural step**: pathway enrichment (`dc.mt.ora` against MSigDB Hallmark gene sets) has only been run for two examples (Monocytes-CD14 and NK, ACS vs CCS) — see `results/enrichment_Harmony_named_*`. Extending this across all three comparisons and more cell types would move from gene lists to biological interpretation.
