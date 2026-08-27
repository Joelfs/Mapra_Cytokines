# Paper-comparison violin figures

Pseudobulk DESeq2 results (sum aggregation, ACS-per-timepoint vs CCS, sterile
ACS only), same setup as `deg_conditions.ipynb`, plotted as paper-style
violins (one point per patient-timepoint pseudobulk sample, not single
cells). Populations are the full `cell_type_Scanorama` (DRVI_named) labels,
pooled — no subclustering. Stars are the PyDESeq2 padj already computed for
that population/timepoint against CCS (`*` <0.05, `**` <0.01, `***` <0.001);
no test is run on the plotted values themselves.

- **`panelA_CD14_monocytes.png`** / **`panelB_CD4_T_cells.png`** — main
  14-gene panels, 3 rows x 6 columns, grouped by which Pekayvaz et al. 2024
  figure the genes come from: row 1 = IL6ST/JAK1/STAT3/SOCS3 (their Figs
  3c/3d), row 2 = EIF3E/HINT1/HMGB1/HIST1H4C (their Figs 2d/2e), row 3 =
  PIM1/VCAN/CD74/UBC/PSME2/ODC1 (their Fig 4d). Same gene order in both
  panels so the two are directly comparable.
- **`compact_6gene_main_figure.png`** — compact main-figure version, single
  2x6 figure (row 1 = CD14 monocytes, row 2 = CD4+ T cells), six genes only:
  IL6ST, CD74, HMGB1, HIST1H4C, PIM1, PSME2. Subset of the same underlying
  results as the two panels above, no re-testing.
- **`Pooled_CD4_14gene_summary.csv`** / **`Pooled_Monocytes_14gene_summary.csv`**
  — log2FoldChange/padj for all 14 genes x 4 ACS timepoints (vs CCS), the
  numbers behind the two main panels.

Headline finding: **HIST1H4C** is the strongest effect in either panel —
significant at all 4 timepoints in CD4+ T cells (log2FC as low as -1.12,
padj down to 7.5e-33 at TP1M), but only weakly/inconsistently significant in
CD14 monocytes (padj<0.05 at TP4M only, and log2FC changes sign across
timepoints). Not one of the paper's original headline genes, but the
strongest hit found here.
