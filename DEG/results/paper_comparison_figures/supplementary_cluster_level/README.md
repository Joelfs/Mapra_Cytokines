# Supplementary: cluster-level results (paper's own Leiden clusters)

Backs the "we also repeated this comparison using the original study's own,
finer clustering... included as supplementary material" statement in the
report. Grouped by `B2_Scanorama_Singlet_rb_mt_cluster` (confirmed to match
Pekayvaz et al. 2024's own published clustering: exact 19-cluster structure,
exact CD4/Treg split at clusters 0/2/5/11, exact excluded-ID range 14–18) —
not a reclustering of our own. Clusters 14–18 excluded throughout (too few
cells per patient-timepoint, same criterion the original study used).

Same PyDESeq2 pseudobulk setup as the main pooled comparison (sum
aggregation, ACS-per-timepoint vs CCS, sterile ACS only), just grouped by
these clusters instead of the pooled Scanorama cell-type label.

- **`violin_paper_cluster_{0,2,5,11,4,6,7}.png`** — CD4+ T cells (clusters
  0, 2, 5), Treg cells (cluster 11), CD14high monocytes (clusters 4, 6, 7).
  Cluster 4 has no CCS violin/stars: only 30 CCS cells total across 11
  patients (max 8 cells in any one sample), below the standard ≥10-cell
  pseudobulk threshold — a genuine data-sparsity gap, not a plotting error.
- **`violin_pooled_vs_cluster_comparison.png`** — PIM1/SOCS3, pooled
  Monocytes-CD14 vs. clusters 4/6/7 side by side; PIM1/cluster 7 at TP2M is
  the one case where cluster-level resolution crosses the fold-change
  threshold that the pooled comparison doesn't (padj significant either way,
  cluster log2FC=1.31 vs pooled 0.82).
- **`cluster_counts_summary.csv`** — cells and patient-timepoint samples per
  cluster, before/after the 14–18 exclusion.
- **`PaperClusters_ACS_{TP1-4}M_vs_CCS.csv`** — unfiltered PyDESeq2 output
  (every tested gene, no significance cutoff applied), one file per
  timepoint, all retained clusters (0–13) in each.
- **`Pooled_CD4_ACS_vs_CCS_per_timepoint.csv`** /
  **`Pooled_Monocytes_ACS_vs_CCS_per_timepoint.csv`** — the pooled
  (non-cluster-split) comparison this is benchmarked against, unfiltered,
  all genes tested (superset of the 14-gene summary in the parent folder).
- **`step5_full_gene_x_cluster_x_timepoint.csv`** — gene × cluster ×
  timepoint log2FC/padj for the 13-gene comparison panel specifically, plus
  the pooled-vs-cluster significance/direction comparison.

Net finding: cluster-level resolution added exactly one incremental result
(PIM1/cluster 7) beyond the pooled comparison, and lost nothing — pooled
captured everything else clusters showed. See the main report text for the
full discussion.
