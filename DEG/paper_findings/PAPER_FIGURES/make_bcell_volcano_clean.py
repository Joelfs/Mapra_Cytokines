"""
B-cell volcano, cleaned: no individually flagged/highlighted genes (no black
outlines for contamination/sex-linked, no gene labels at all), since the text
now says these values cannot be individually trusted. This shows the overall
shape of the contrast, not a claim about specific genes.
"""
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

DEG = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/DEG"
RESULTS = os.path.join(DEG, "results", "DRVI_named_ACS_vs_CCS.csv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "supplementary", "FigS3b_bcell_volcano_clean.png")

UP_COLOR, DOWN_COLOR, NS_COLOR = "#C44E52", "#4C72B0", "#cccccc"
PADJ_T, LFC_T = 0.05, 1.0

df = pd.read_csv(RESULTS)
d = df[df["cell_type"] == "B-cell"].copy()
d["neglog10padj"] = -np.log10(d["padj"].clip(lower=1e-300))
is_sig = d["padj"] < PADJ_T
is_up = is_sig & (d["log2FoldChange"] > LFC_T)
is_down = is_sig & (d["log2FoldChange"] < -LFC_T)
is_ns = ~(is_up | is_down)

fig, ax = plt.subplots(figsize=(6.2, 5.3))
ax.scatter(d.loc[is_ns, "log2FoldChange"], d.loc[is_ns, "neglog10padj"],
           s=10, color=NS_COLOR, alpha=0.5, linewidths=0, label=f"n.s. ({is_ns.sum()})")
ax.scatter(d.loc[is_down, "log2FoldChange"], d.loc[is_down, "neglog10padj"],
           s=20, color=DOWN_COLOR, linewidths=0, label=f"down ({is_down.sum()})")
ax.scatter(d.loc[is_up, "log2FoldChange"], d.loc[is_up, "neglog10padj"],
           s=20, color=UP_COLOR, linewidths=0, label=f"up ({is_up.sum()})")

ax.axhline(-np.log10(PADJ_T), color="black", linestyle="--", linewidth=0.8, alpha=0.6)
ax.axvline(LFC_T, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
ax.axvline(-LFC_T, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
ax.set_xlabel("log$_2$ Fold Change (ACS vs. CCS)", fontsize=10)
ax.set_ylabel("-log$_{10}$(padj)", fontsize=10)
ax.set_title(f"B cells, ACS vs. CCS\n{len(d):,} genes tested, {(is_up|is_down).sum()} significant",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8.5, loc="upper right", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print("Saved:", OUT)
