"""
Top 10 genes per cell type ranked by adjusted p-value -- pooled ACS vs CCS.
Genuinely top-10-per-column, no gene silently dropped.

Two genes/gene-classes are flagged (shown, not excluded) because they are
interpreted in the Discussion rather than as disease findings:
  - HLA-DQA2 (‡): genotype-driven, not disease-associated -- see Discussion
  - sex-linked transcripts (†): likely index cohort sex composition -- see Discussion
"""
import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

BASE = os.path.dirname(os.path.abspath(__file__))
DEG = os.path.dirname(os.path.dirname(BASE))
RESULTS = os.path.join(DEG, "results", "DRVI_named_ACS_vs_CCS.csv")
OUT = os.path.join(BASE, "supplementary", "FigS3_padj_ranked_heatmap.png")

SEX = {"XIST","TSIX","RPS4Y1","UTY","ZFY","DDX3Y","EIF1AY","KDM5D","NLGN4Y","USP9Y"}
SEX_PREFIX = ("TTTY",)
GENOTYPE_FLAG = {"HLA-DQA2"}
N = 10
ORDER = ["Monocytes - CD14","Monocytes - CD16_FCGR3A","Dendritic","B-cell",
         "NK","T-cell-CD4","T-cell-CD8"]
HIGHLIGHT = {"IGHD","IGHM","FCER2","TCL1A"}

def is_sex(g):
    return g in SEX or g.startswith(SEX_PREFIX)

df = pd.read_csv(RESULTS)
sig = df[df.significant == True].copy()  # nothing removed

per_ct_top = {ct: sig[sig.cell_type == ct].nsmallest(N, "padj") for ct in ORDER if (sig.cell_type == ct).any()}
picked = pd.concat(per_ct_top.values()).drop_duplicates(["cell_type", "gene"])
print(f"genuine top-{N}-per-cell-type: {picked.gene.nunique()} unique genes, {len(picked)} gene-celltype rows")

mat = picked.pivot(index="gene", columns="cell_type", values="log2FoldChange")
mat = mat.reindex(columns=[c for c in ORDER if c in mat.columns])
mat = mat.loc[mat.mean(axis=1).sort_values().index]

vmax = np.nanmax(np.abs(mat.values))
fig, ax = plt.subplots(figsize=(7.4, 0.235 * len(mat) + 1.95))
im = ax.imshow(mat.values, cmap="RdBu_r",
               norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), aspect="auto")
ax.set_xticks(range(mat.shape[1]))
ax.set_xticklabels([c.replace("Monocytes - ","Mono ").replace("_FCGR3A","") for c in mat.columns],
                   rotation=40, ha="right", fontsize=8.5)
ax.set_yticks(range(mat.shape[0]))
ylabels = []
for g in mat.index:
    if g in GENOTYPE_FLAG:
        ylabels.append(f"{g} ‡")
    elif is_sex(g):
        ylabels.append(f"{g} †")
    else:
        ylabels.append(g)
labels = ax.set_yticklabels(ylabels, fontsize=7)
for lab, g in zip(labels, mat.index):
    if g in HIGHLIGHT:
        lab.set_color("#8B2E38"); lab.set_fontweight("bold")
    elif g in GENOTYPE_FLAG or is_sex(g):
        lab.set_color("#888888"); lab.set_fontstyle("italic")
ax.set_xticks(np.arange(-.5, mat.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-.5, mat.shape[0], 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.1)
ax.tick_params(which="minor", length=0)
cb = fig.colorbar(im, ax=ax, shrink=0.4, pad=0.02)
cb.set_label("log$_2$ Fold Change (ACS vs. CCS)", fontsize=8); cb.ax.tick_params(labelsize=7)
ax.set_title(f"Top {N} genes per cell type ranked by adjusted p-value\n"
             "pooled ACS vs. CCS",
             fontsize=10, fontweight="bold")
fig.text(0.02, -0.005,
          "$^{\\ddagger}$HLA-DQA2: genotype-driven, not disease-associated -- see Discussion\n"
          "$^{\\dagger}$Sex-linked transcripts: likely index cohort sex composition -- see Discussion",
          fontsize=6.5, color="#666666", style="italic")
fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print("Saved:", OUT, "-", len(mat), "rows")
print("HLA-DQA2 present:", "HLA-DQA2" in mat.index)
print("B-cell markers present:", [g for g in HIGHLIGHT if g in mat.index])
print("colour scale vmax (driven by):", mat.abs().max().max(), "->", mat.abs().stack().idxmax())
