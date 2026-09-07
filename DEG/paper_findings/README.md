# DEG paper findings — ACS vs CCS

`PAPER_FIGURES/` holds the figures, tables, and generating scripts actually
used in the manuscript's DEG section: the replication panel (Fig1a/1b), the
naive B-cell timepoint panel (Fig2), the classifier upset plot, and the
supplementary heatmap/volcano/library-confound/HLA-DQA2 figures and tables.

`source_data/` holds the small per-timepoint DEG tables the main-text
figures are built from.

## `_not_for_push/`

Anything under a folder named `_not_for_push/` (git-ignored, see
`.gitignore`) is superseded material kept for reference only: earlier drafts
of figures, retired versions, internal planning documents, and scripts from
analysis threads that didn't make it into the final paper (e.g. the
Harmony-vs-DRVI integration comparison, and the earlier GBP4/dendritic-cell
investigation). None of it is required to reproduce the current manuscript;
it's kept locally in case something needs to be revisited, not because it's
part of the reported results.
