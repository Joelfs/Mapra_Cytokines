"""
B-cell sample-level heatmap: top 20 significant genes (by padj, no gene
singled out) x every pseudobulk sample, with condition AND library shown as
annotation bars. Replaces the volcano plot -- a volcano with individually
flagged/highlighted genes no longer fits an analysis where library and
condition are structurally confounded; this figure lets the reader see
directly whether expression tracks condition or library, rather than being
told which genes to trust.

Run: /vol/disk/miniforge3/envs/mapra_cytokines/bin/python make_bcell_sample_heatmap.py
"""
import os
import numpy as np, pandas as pd, scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.sparse import issparse, csr_matrix

DEG = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/DEG"
META = os.path.join(DEG, "deg_metadata_shared.parquet")
RESULTS = os.path.join(DEG, "results", "DRVI_named_ACS_vs_CCS.csv")
DATA = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/data/data_for_practicum_post_integration.h5ad"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "supplementary", "FigS3_bcell_sample_heatmap.png")

N_GENES = 20

res = pd.read_csv(RESULTS)
sig = res[(res.cell_type == "B-cell") & (res.significant == True)]
genes = sig.nsmallest(N_GENES, "padj")["gene"].tolist()

meta = pd.read_parquet(META)
ad = sc.read_h5ad(DATA)
m = (meta["cell_type_DRVI"] == "B-cell") & meta["condition"].isin(["CCS", "ACS_sterile"])
m &= meta.index.isin(ad.obs_names)
cells = meta.index[m]
sub = ad[cells]
c = sub.layers["counts"]; c = c.tocsr() if issparse(c) else c
s = sub.obs["display_name"].astype(str).values
ids = pd.Index(pd.unique(s))
ri = np.array([{x: i for i, x in enumerate(ids)}[x] for x in s])
ind = csr_matrix((np.ones(len(ri)), (ri, np.arange(len(ri)))), shape=(len(ids), len(ri)))
pb = np.asarray((ind @ c).todense())
ncell = pd.Series(s).value_counts().reindex(ids).values
keep = (ncell >= 10) & (pb.sum(1) >= 1000)
cpm = pd.DataFrame(pb / pb.sum(1)[:, None] * 1e6, index=ids, columns=sub.var_names)[keep]
sm = (meta.loc[cells, ["display_name", "condition"]]
      .assign(display_name=lambda d: d.display_name.astype(str))
      .drop_duplicates("display_name").set_index("display_name").loc[ids[keep]])
lib = sub.obs.assign(display_name=sub.obs["display_name"].astype(str)).groupby("display_name")["library"].first().reindex(ids[keep])
sm["library"] = lib.values

# order: condition (CCS then ACS), then library within each
sm["cond_rank"] = (sm.condition == "ACS_sterile").astype(int)
sm = sm.sort_values(["cond_rank", "library"])
mat = np.log1p(cpm.loc[sm.index, genes]).T
# z-score each gene (row) so all genes are visually comparable regardless of magnitude
z = mat.sub(mat.mean(axis=1), axis=0).div(mat.std(axis=1), axis=0)

libs_all = sorted(sm["library"].unique(), key=lambda x: (len(x), x))
cmap_lib = plt.get_cmap("tab20")
lib_color = {l: cmap_lib(i % 20) for i, l in enumerate(libs_all)}
cond_color = {"CCS": "#4C72B0", "ACS_sterile": "#C44E52"}

fig = plt.figure(figsize=(11, 0.32 * len(genes) + 2.3))
gs = fig.add_gridspec(3, 1, height_ratios=[0.4, 0.4, len(genes) * 0.32], hspace=0.05)

ax_cond = fig.add_subplot(gs[0])
ax_lib = fig.add_subplot(gs[1])
ax_hm = fig.add_subplot(gs[2])

for i, sample in enumerate(sm.index):
    ax_cond.add_patch(plt.Rectangle((i, 0), 1, 1, color=cond_color[sm.loc[sample, "condition"]]))
    ax_lib.add_patch(plt.Rectangle((i, 0), 1, 1, color=lib_color[sm.loc[sample, "library"]]))
for ax in (ax_cond, ax_lib):
    ax.set_xlim(0, len(sm)); ax.set_ylim(0, 1); ax.axis("off")
ax_cond.text(-2, 0.5, "condition", ha="right", va="center", fontsize=8)
ax_lib.text(-2, 0.5, "library", ha="right", va="center", fontsize=8)

vmax = np.nanmax(np.abs(z.values))
im = ax_hm.imshow(z.values, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
                   aspect="auto", extent=[0, len(sm), len(genes), 0])
ax_hm.set_yticks(np.arange(len(genes)) + 0.5)
ax_hm.set_yticklabels(genes, fontsize=7.5)
ax_hm.set_xticks([])
ax_hm.set_xlabel(f"pseudobulk samples (n={len(sm)}), grouped by condition then library", fontsize=8.5)

# boundary line between CCS and ACS blocks
split = (sm["cond_rank"] == 0).sum()
for ax in (ax_cond, ax_lib, ax_hm):
    ax.axvline(split, color="black", linewidth=1.2)

cond_handles = [plt.Rectangle((0,0),1,1,color=cond_color[k]) for k in ["CCS","ACS_sterile"]]
lib_handles = [plt.Rectangle((0,0),1,1,color=lib_color[l]) for l in libs_all]
fig.legend(cond_handles, ["CCS","ACS"], loc="upper left", bbox_to_anchor=(0.90, 0.88), fontsize=7.5, frameon=False, title="condition", title_fontsize=8)
fig.legend(lib_handles, libs_all, loc="upper left", bbox_to_anchor=(0.90, 0.65), fontsize=6.5, frameon=False, ncol=1, title="library", title_fontsize=8)

cb = fig.colorbar(im, ax=ax_hm, shrink=0.6, pad=0.14, location="left")
cb.set_label("row z-score\n(log1p CPM)", fontsize=7)
cb.ax.tick_params(labelsize=6.5)

fig.suptitle("B cells, top 20 significant genes by adjusted p-value: condition and library shown together",
             fontsize=10.5, fontweight="bold", y=0.995)
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print("Saved:", OUT)
print("genes:", genes)
