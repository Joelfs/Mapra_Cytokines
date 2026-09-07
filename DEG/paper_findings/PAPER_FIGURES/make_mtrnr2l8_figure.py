"""
MTRNR2L8: per-sample expression, CCS vs ACS, coloured by sequencing library.
Built because the gene recurs across nearly every cell type in the top-10-by-
padj heatmap, so it needed the same scrutiny already applied to HLA-DQA2.

Verdict, from checking (not assuming): unlike HLA-DQA2 this is NOT bimodal
(zero zero-inflation in any group) and does NOT correlate with %mt QC
(r~0.1, n.s.) -- ruling out the obvious ambient-RNA explanation. But every
library in this comparison is condition-pure: CCS samples come only from
L11/L12, ACS_sterile only from the other 11 libraries. Condition and library
are completely confounded for this gene, and within ACS alone, library medians
range 66-236 CPM -- almost as wide as the CCS-vs-ACS gap itself. This plot
makes that visible directly.
"""
import os
import numpy as np, pandas as pd, scanpy as sc
import matplotlib.pyplot as plt
from scipy.sparse import issparse, csr_matrix

DEG = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/DEG"
META = os.path.join(DEG, "deg_metadata_shared.parquet")
DATA = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/data/data_for_practicum_post_integration.h5ad"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "supplementary", "FigS4_MTRNR2L8_library_confound.png")

CELLTYPES = ["B-cell", "NK", "T-cell-CD4", "T-cell-CD8"]

def pb(meta, ad, celltype):
    m = (meta["cell_type_DRVI"] == celltype) & meta["condition"].isin(["CCS", "ACS_sterile"])
    m &= meta.index.isin(ad.obs_names)
    cells = meta.index[m]
    sub = ad[cells]
    c = sub.layers["counts"]; c = c.tocsr() if issparse(c) else c
    s = sub.obs["display_name"].astype(str).values
    ids = pd.Index(pd.unique(s))
    ri = np.array([{x: i for i, x in enumerate(ids)}[x] for x in s])
    ind = csr_matrix((np.ones(len(ri)), (ri, np.arange(len(ri)))), shape=(len(ids), len(ri)))
    pbm = np.asarray((ind @ c).todense())
    ncell = pd.Series(s).value_counts().reindex(ids).values
    keep = (ncell >= 10) & (pbm.sum(1) >= 1000)
    cpm = pd.DataFrame(pbm / pbm.sum(1)[:, None] * 1e6, index=ids, columns=sub.var_names)[keep]
    sm = (meta.loc[cells, ["display_name", "condition"]]
          .assign(display_name=lambda d: d.display_name.astype(str))
          .drop_duplicates("display_name").set_index("display_name").loc[ids[keep]])
    lib = sub.obs.assign(display_name=sub.obs["display_name"].astype(str)).groupby("display_name")["library"].first().reindex(ids[keep])
    sm["library"] = lib.values
    return cpm, sm

meta = pd.read_parquet(META)
ad = sc.read_h5ad(DATA)

fig, axes = plt.subplots(1, 4, figsize=(11, 3.6), sharey=False)
rng = np.random.default_rng(0)
libs_all = sorted(meta["library"].dropna().unique(), key=lambda x: (len(x), x))
cmap = plt.get_cmap("tab20")
lib_color = {l: cmap(i % 20) for i, l in enumerate(libs_all)}

for ax, ct in zip(axes, CELLTYPES):
    cpm, sm = pb(meta, ad, ct)
    v = np.log1p(cpm["MTRNR2L8"])
    for xi, cond in enumerate(["CCS", "ACS_sterile"]):
        idx = sm.index[sm.condition == cond]
        y = v.loc[idx].values
        libs = sm.loc[idx, "library"].values
        x = np.full(len(y), xi) + rng.uniform(-0.12, 0.12, len(y))
        colors = [lib_color[l] for l in libs]
        ax.scatter(x, y, s=18, c=colors, alpha=0.85, linewidths=0.3, edgecolors="white")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["CCS", "ACS"], fontsize=9)
    ax.set_title(ct, fontsize=9.5, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("log(1 + CPM)\nMTRNR2L8", fontsize=9)

handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=lib_color[l], markersize=6, label=l)
           for l in libs_all]
fig.legend(handles=handles, title="library", loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7, title_fontsize=8, frameon=False)
fig.suptitle("MTRNR2L8: every point coloured by sequencing library\n"
             "library is completely confounded with condition in this cohort",
             fontsize=10.5, fontweight="bold", y=1.05)
fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print("Saved:", OUT)
