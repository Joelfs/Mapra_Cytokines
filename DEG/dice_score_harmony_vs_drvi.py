"""
Dice similarity between the Harmony- and DRVI-derived Scanorama-named cell
type labels (cell_type_Harmony vs cell_type_DRVI). Both are majority-vote
label transfers from the same reference annotation (cell_type_Scanorama)
onto two different clustering solutions (harmony_leiden vs drvi_leiden), so
even identically-named clusters don't necessarily contain the same cells --
this quantifies how much they actually agree.

Computed overall (all cells) and separately within each of the four
ACS-timepoint-vs-CCS populations used for Comparison 4 in deg_conditions.ipynb
(same subsetting logic), so agreement can be checked in the exact cell
populations each DEG comparison actually runs on.

Reads DEG/deg_metadata_shared.parquet -- no need to touch the full h5ad.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

METADATA_PATH = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/DEG/deg_metadata_shared.parquet"
OUTPUT_DIR = "/vol/disk/ubuntu/master_practicum_cytokines/lisa/deg_conditions"
COL_A = "cell_type_Harmony"
COL_B = "cell_type_DRVI"


def dice_table(df, col_a=COL_A, col_b=COL_B):
    """Per-cell-type Dice coefficient between two label columns on the same cells,
    plus macro (unweighted mean over cell types) and micro (global, count-weighted)
    summary rows."""
    labels = sorted(set(df[col_a].unique()) | set(df[col_b].unique()))
    rows = []
    for label in labels:
        a = df[col_a] == label
        b = df[col_b] == label
        n_a, n_b = int(a.sum()), int(b.sum())
        n_intersect = int((a & b).sum())
        denom = n_a + n_b
        dice = 2 * n_intersect / denom if denom > 0 else float("nan")
        rows.append((label, n_a, n_b, n_intersect, dice))

    table = pd.DataFrame(rows, columns=["cell_type", "n_Harmony", "n_DRVI", "n_intersect", "dice"])
    table = table.sort_values("dice", ascending=False).reset_index(drop=True)

    macro_dice = table["dice"].mean()
    micro_dice = 2 * table["n_intersect"].sum() / (table["n_Harmony"].sum() + table["n_DRVI"].sum())
    overall_match = (df[col_a] == df[col_b]).mean()

    return table, macro_dice, micro_dice, overall_match


def run(df, label, save_csv):
    table, macro_dice, micro_dice, overall_match = dice_table(df)
    print(f"\n=== {label} (n={len(df):,} cells) ===")
    print(table.to_string(index=False))
    print(f"Macro-average Dice: {macro_dice:.3f}  |  Micro (global) Dice: {micro_dice:.3f}  "
          f"|  Overall per-cell match: {overall_match:.3f}")
    table.to_csv(save_csv, index=False)
    return table, macro_dice, micro_dice, overall_match


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_parquet(METADATA_PATH)

    summary_rows = []
    per_celltype_wide = {}

    # Overall, all cells (all conditions pooled) -- baseline reference
    table, macro, micro, match = run(
        df, "Overall (all cells, all conditions)",
        os.path.join(OUTPUT_DIR, "dice_score_Harmony_vs_DRVI_overall.csv"),
    )
    summary_rows.append(("Overall", macro, micro, match, len(df)))
    per_celltype_wide["Overall"] = table.set_index("cell_type")["dice"]

    # Per ACS-timepoint vs CCS -- same subsetting as Comparison 4 in deg_conditions.ipynb
    for tp in ["TP1M", "TP2M", "TP3M", "TP4M"]:
        mask = ((df["condition"] == "ACS_sterile") & (df["timepoint"] == tp)) | (df["condition"] == "CCS")
        sub = df[mask]
        comparison = f"ACS_{tp}_vs_CCS"
        table, macro, micro, match = run(
            sub, f"ACS ({tp}) vs CCS",
            os.path.join(OUTPUT_DIR, f"dice_score_Harmony_vs_DRVI_{comparison}.csv"),
        )
        summary_rows.append((comparison, macro, micro, match, len(sub)))
        per_celltype_wide[comparison] = table.set_index("cell_type")["dice"]

    # Combined summary: one row per population, macro/micro/overall-match + n cells
    summary = pd.DataFrame(summary_rows, columns=["population", "macro_dice", "micro_dice", "overall_match", "n_cells"])
    summary_path = os.path.join(OUTPUT_DIR, "dice_score_Harmony_vs_DRVI_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")

    # Wide table: cell type x population, for the heatmap
    wide = pd.DataFrame(per_celltype_wide)
    wide_path = os.path.join(OUTPUT_DIR, "dice_score_Harmony_vs_DRVI_by_celltype.csv")
    wide.to_csv(wide_path)
    print(f"Saved per-cell-type-by-population table: {wide_path}")

    # Heatmap: cell type x population
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * wide.shape[1] + 2), max(4, 0.4 * wide.shape[0] + 2)))
    sns.heatmap(wide, cmap="RdYlGn", vmin=0, vmax=1, annot=True, fmt=".2f",
                linewidths=0.4, ax=ax, cbar_kws={"label": "Dice score"})
    ax.set_title("Harmony vs DRVI (Scanorama-named) label agreement", fontsize=11)
    ax.set_xlabel("Population")
    ax.set_ylabel("Cell type")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    heatmap_path = os.path.join(OUTPUT_DIR, "dice_score_Harmony_vs_DRVI_heatmap.png")
    plt.savefig(heatmap_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved heatmap: {heatmap_path}")

    # Same heatmap, timepoints only (drop the all-conditions-pooled "Overall" column)
    wide_tp = wide.drop(columns=["Overall"])
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * wide_tp.shape[1] + 2), max(4, 0.4 * wide_tp.shape[0] + 2)))
    sns.heatmap(wide_tp, cmap="RdYlGn", vmin=0, vmax=1, annot=True, fmt=".2f",
                linewidths=0.4, ax=ax, cbar_kws={"label": "Dice score"})
    ax.set_title("Harmony vs DRVI (Scanorama-named) label agreement, per ACS timepoint vs CCS", fontsize=11)
    ax.set_xlabel("ACS timepoint vs CCS")
    ax.set_ylabel("Cell type")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    heatmap_tp_path = os.path.join(OUTPUT_DIR, "dice_score_Harmony_vs_DRVI_heatmap_by_timepoint.png")
    plt.savefig(heatmap_tp_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved timepoint-only heatmap: {heatmap_tp_path}")


if __name__ == "__main__":
    main()
