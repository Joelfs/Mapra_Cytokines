"""
UMAP (Harmony embedding), one figure per ACS timepoint, to visually explain
the Monocytes-CD14 named-vs-leiden heatmaps: are clusters 0 and 9 actually
inside the "Monocytes - CD14" named region, and how sparse are CCS cells in
cluster 9?

Three panels per timepoint, same cell subset as that timepoint's DEG
comparison (ACS-sterile-at-that-timepoint + CCS):
  (a) condition       -- ACS_sterile vs CCS
  (b) harmony_leiden   -- cluster 0 / cluster 9 highlighted, rest grey
  (c) cell_type_Harmony -- Monocytes - CD14 (majority-vote label) highlighted

Reads obs columns + the X_umap_harmony embedding directly via h5py (no
counts/X matrix needed) -- much lighter than a full sc.read_h5ad.
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

DATA_PATH = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/data/data_for_practicum_post_integration.h5ad"
OUTPUT_DIR = "/vol/disk/ubuntu/master_practicum_cytokines/lisa/monocytes_cd14_harmony_vs_leiden"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def read_cat(f, name):
    g = f["obs"][name]
    cats = np.array([c.decode() if isinstance(c, bytes) else c for c in g["categories"][()]])
    codes = g["codes"][()]
    return pd.Categorical.from_codes(codes, categories=cats)


f = h5py.File(DATA_PATH, "r")
harmony_leiden = read_cat(f, "harmony_leiden").astype(int)
classification = read_cat(f, "classification")
display_name = read_cat(f, "display_name")
scan = read_cat(f, "cell_type_Scanorama")
umap = f["obsm"]["X_umap_harmony"][()]

df = pd.DataFrame({
    "harmony_leiden": harmony_leiden,
    "classification": classification,
    "display_name": display_name,
    "cell_type_Scanorama": scan,
    "umap1": umap[:, 0],
    "umap2": umap[:, 1],
})
df["timepoint"] = df["display_name"].astype(str).apply(lambda s: f"TP{s.split('.')[-1]}M" if "." in s else "TP0M")
condition_map = {
    "acs_w_o_infection": "ACS_sterile", "acs_w_infection": "ACS_infection",
    "acs_subacute": "ACS_subacute", "ccs": "CCS",
    "vollstaendiger_ausschluss": "non-CCS", "koronarsklerose": "Sclerosis",
}
df["condition"] = df["classification"].map(condition_map)

# majority-vote label transfer, same as deg_conditions.ipynb's transfer_celltype_labels
majority = df.groupby("harmony_leiden", observed=True)["cell_type_Scanorama"].agg(lambda s: s.value_counts().idxmax())
df["cell_type_Harmony"] = df["harmony_leiden"].map(majority)

for tp in ["TP1M", "TP2M", "TP3M", "TP4M"]:
    mask = ((df["condition"] == "ACS_sterile") & (df["timepoint"] == tp)) | (df["condition"] == "CCS")
    sub = df[mask]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # (a) condition
    ax = axes[0]
    for cond, color in [("ACS_sterile", "#d62728"), ("CCS", "#1f77b4")]:
        s = sub[sub["condition"] == cond]
        ax.scatter(s["umap1"], s["umap2"], s=3, color=color, alpha=0.5, label=f"{cond} (n={len(s):,})")
    ax.set_title(f"Condition, {tp}")
    ax.legend(markerscale=4, fontsize=9, loc="upper right")

    # (b) harmony_leiden, 0 and 9 highlighted
    ax = axes[1]
    rest = sub[~sub["harmony_leiden"].isin([0, 9])]
    ax.scatter(rest["umap1"], rest["umap2"], s=3, color="lightgrey", alpha=0.4, label="other clusters")
    for cl, color in [(0, "#2ca02c"), (9, "#ff7f0e")]:
        s = sub[sub["harmony_leiden"] == cl]
        s_ccs = s[s["condition"] == "CCS"]
        ax.scatter(s["umap1"], s["umap2"], s=5, color=color, alpha=0.7, label=f"cluster {cl} (n={len(s):,}, CCS n={len(s_ccs):,})")
    ax.set_title(f"harmony_leiden 0 / 9, {tp}")
    ax.legend(markerscale=3, fontsize=9, loc="upper right")

    # (c) cell_type_Harmony, Monocytes-CD14 highlighted
    ax = axes[2]
    rest = sub[sub["cell_type_Harmony"] != "Monocytes - CD14"]
    mono = sub[sub["cell_type_Harmony"] == "Monocytes - CD14"]
    ax.scatter(rest["umap1"], rest["umap2"], s=3, color="lightgrey", alpha=0.4, label="other named types")
    ax.scatter(mono["umap1"], mono["umap2"], s=5, color="#9467bd", alpha=0.7, label=f"Monocytes - CD14 (n={len(mono):,})")
    ax.set_title(f"Named: Monocytes - CD14, {tp}")
    ax.legend(markerscale=3, fontsize=9, loc="upper right")

    for ax in axes:
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"Harmony UMAP -- ACS {tp} vs CCS cell subset", fontsize=13)
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"umap_Harmony_MonocytesCD14_ACS_{tp}_vs_CCS.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

print("\nDone.")
