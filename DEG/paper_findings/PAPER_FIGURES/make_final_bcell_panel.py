"""
Final Results-supplementary panel: (a) padj-ranked heatmap, (b) clean B-cell
volcano. Side by side, matching the original layout style, heatmap tall on
the left, volcano top-aligned on the right.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "supplementary")
OUT = os.path.join(S, "FigS3_bcell_evidence_panel.png")

heat = mpimg.imread(os.path.join(S, "FigS3_padj_ranked_heatmap.png"))
volc = mpimg.imread(os.path.join(S, "FigS3b_bcell_volcano_clean.png"))

fig = plt.figure(figsize=(13, 11))
fig.patch.set_facecolor("white")

hh, hw = heat.shape[:2]
ax1 = fig.add_axes([0.03, 0.02, 0.46, 0.96])
ax1.imshow(heat); ax1.axis("off")
fig.text(0.03, 0.99, "a", fontsize=16, fontweight="bold")

vh, vw = volc.shape[:2]
ax2_h = 0.46 * (vh / vw) * (13 / 11)  # match aspect given the axes width below
ax2 = fig.add_axes([0.53, 0.96 - ax2_h, 0.44, ax2_h])
ax2.imshow(volc); ax2.axis("off")
fig.text(0.53, 0.99, "b", fontsize=16, fontweight="bold")

fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print("Saved:", OUT)
