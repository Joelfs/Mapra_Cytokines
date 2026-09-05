"""
Matched-cluster gene conservation: for each resolution-1.0 cluster and its
best-matching resolution-2.0 cluster (from cluster_correspondence_*.csv),
compares the actual significant gene sets between the two DEG runs --
unlike the earlier pooled "carry-over" table, this compares gene-for-gene
within the SAME matched cell population, not pooled across all clusters.

Requires (all should already exist on the cluster after running
resolution_correspondence_heatmap.py and deg_resolution2_reconstructed.py):
  - cluster_correspondence_Harmony.csv / cluster_correspondence_DRVI.csv
  - deg_conditions_leiden10/{Harmony_leiden,DRVI_leiden}_{comparison}.csv   (resolution 1.0)
  - deg_conditions_resolution2_reconstructed/{Harmony_leiden_res2,DRVI_leiden_res2}_{comparison}.csv   (resolution 2.0, reconstructed)

Fast: pure pandas, no h5ad/scanpy needed.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE = "/vol/disk/ubuntu/master_practicum_cytokines/sudenaz"
RES1_DIR = os.path.join(BASE, "deg_conditions_leiden10")
RES2_DIR = os.path.join(BASE, "deg_conditions_resolution2_reconstructed")
OUT_DIR = BASE

COMPARISONS = ["ACS_vs_CCS", "ACS_vs_nonCCS", "ACS_TP1M_vs_TP4M"]
GROUPINGS = {
    "Harmony": ("Harmony_leiden", "Harmony_leiden_res2", "cluster_correspondence_Harmony.csv"),
    "DRVI": ("DRVI_leiden", "DRVI_leiden_res2", "cluster_correspondence_DRVI.csv"),
}

all_rows = []

for grouping, (res1_name, res2_name, corr_file) in GROUPINGS.items():
    print(f"\n=== {grouping} ===")
    corr = pd.read_csv(os.path.join(BASE, corr_file))
    corr["res1_cluster"] = corr["res1_cluster"].astype(str)
    corr["best_matching_res2_cluster"] = corr["best_matching_res2_cluster"].astype(str)

    for comparison in COMPARISONS:
        res1_path = os.path.join(RES1_DIR, f"1.0{res1_name}_{comparison}.csv")
        res2_path = os.path.join(RES2_DIR, f"{res2_name}_{comparison}.csv")
        if not os.path.exists(res1_path) or not os.path.exists(res2_path):
            print(f"  MISSING for {comparison}: "
                  f"{res1_path if not os.path.exists(res1_path) else res2_path}")
            continue

        df1 = pd.read_csv(res1_path)
        df2 = pd.read_csv(res2_path)
        df1["cell_type"] = df1["cell_type"].astype(str)
        df2["cell_type"] = df2["cell_type"].astype(str)

        for _, row in corr.iterrows():
            c1, c2 = row["res1_cluster"], row["best_matching_res2_cluster"]
            overlap_pct = row["overlap_pct"]

            sig1 = df1[(df1["cell_type"] == c1) & (df1["significant"])]
            sig2 = df2[(df2["cell_type"] == c2) & (df2["significant"])]

            genes1 = set(sig1["gene"])
            genes2 = set(sig2["gene"])
            shared = genes1 & genes2
            lost = genes1 - genes2      # significant at res 1.0, not in matched res 2.0 cluster
            gained = genes2 - genes1    # significant at res 2.0, not at res 1.0

            if shared:
                dir1 = sig1.set_index("gene")["log2FoldChange"]
                dir2 = sig2.set_index("gene")["log2FoldChange"]
                agree = sum(1 for g in shared if (dir1[g] > 0) == (dir2[g] > 0))
                direction_agreement = agree / len(shared) * 100
            else:
                direction_agreement = float("nan")

            all_rows.append({
                "grouping": grouping,
                "comparison": comparison,
                "res1_cluster": c1,
                "res2_cluster": c2,
                "cluster_overlap_pct": overlap_pct,
                "sig_genes_res1": len(genes1),
                "sig_genes_res2": len(genes2),
                "conserved": len(shared),
                "lost": len(lost),
                "gained": len(gained),
                "conserved_pct_of_res1": round(len(shared) / len(genes1) * 100, 1) if genes1 else np.nan,
                "direction_agreement_pct": round(direction_agreement, 1) if shared else np.nan,
                "conserved_genes": ";".join(sorted(shared)),
                "lost_genes": ";".join(sorted(lost)),
                "gained_genes": ";".join(sorted(gained)),
            })

result = pd.DataFrame(all_rows)
result.to_csv(os.path.join(OUT_DIR, "matched_cluster_gene_conservation.csv"), index=False)
print(f"\nSaved: matched_cluster_gene_conservation.csv ({len(result)} rows)")

# Only show rows with at least some significant genes on either side -- most
# matched pairs will have 0/0, which isn't interesting to print or plot.
active = result[(result["sig_genes_res1"] > 0) | (result["sig_genes_res2"] > 0)]
print(f"\nMatched pairs with at least one significant gene on either side ({len(active)}):")
print(active[["grouping", "comparison", "res1_cluster", "res2_cluster", "cluster_overlap_pct",
               "sig_genes_res1", "sig_genes_res2", "conserved", "lost", "gained",
               "conserved_pct_of_res1", "direction_agreement_pct"]].to_string(index=False))

# ---------- heatmap per grouping: conserved % across matched pairs x comparisons ----------
for grouping in GROUPINGS:
    sub = active[active["grouping"] == grouping].copy()
    if sub.empty:
        print(f"\nNo active matched pairs for {grouping} -- skipping heatmap")
        continue
    sub["pair_label"] = "r1:" + sub["res1_cluster"] + " -> r2:" + sub["res2_cluster"]
    pivot = sub.pivot_table(index="pair_label", columns="comparison",
                             values="conserved_pct_of_res1", aggfunc="mean")
    pivot = pivot.reindex(columns=[c for c in COMPARISONS if c in pivot.columns])

    fig_w = max(6, 1.4 * pivot.shape[1] + 3)
    fig_h = max(4, 0.35 * pivot.shape[0] + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn", vmin=0, vmax=100, ax=ax,
                cbar_kws={"label": "% conserved (of res 1.0 sig. genes)"},
                linewidths=0.4, linecolor="white")
    ax.set_title(f"{grouping}: gene conservation, matched clusters (res 1.0 -> res 2.0)", fontsize=10)
    ax.set_xlabel("Comparison", fontsize=9)
    ax.set_ylabel("Matched cluster pair", fontsize=9)
    ax.tick_params(axis="x", labelsize=8, rotation=20)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"matched_cluster_conservation_{grouping}.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved matched_cluster_conservation_{grouping}.png")

print("\nDone.")