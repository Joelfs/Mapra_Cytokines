"""
Visual version of the Step 5 divergence table: PIM1 (the one gene where a
cluster crossed significance that pooling missed -- cluster 7, TP2M) and
SOCS3 (cluster 7 shows a much larger raw effect size than pooled, though
neither reaches significance) -- each shown as Pooled Monocytes-CD14 vs.
clusters 4/6/7 individually, same x-axis/stats convention as everything else
in this folder. Answers "does grouping by subpopulation actually change the
picture" at a glance, for the one case where it did.

Two separate figures, one per gene, each a 2x2 grid of the four populations
(Pooled, Cluster 4, Cluster 6, Cluster 7) -- narrower/squarer than the
original single 4-row x 2-column combined figure, so each fits a
supplementary-figure panel on its own.
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.sparse import issparse, csr_matrix

DATA_PATH = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/data/data_for_practicum_post_integration.h5ad"
METADATA_PATH = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/DEG/deg_metadata_shared.parquet"
OUTPUT_DIR = "/vol/disk/ubuntu/master_practicum_cytokines/lisa/paper_leiden_clusters_deg"

GENES = ["PIM1", "SOCS3"]
ROWS = [
    ("Pooled Monocytes-CD14", "pooled", None),
    ("Cluster 4", "cluster", 4),
    ("Cluster 6", "cluster", 6),
    ("Cluster 7", "cluster", 7),
]
GROUP_ORDER = ["CCS", "ACS TP1M", "ACS TP2M", "ACS TP3M", "ACS TP4M"]
GROUP_COLORS = {"CCS": "#1f77b4", "ACS TP1M": "#fca082", "ACS TP2M": "#fb6a4a",
                "ACS TP3M": "#de2d26", "ACS TP4M": "#990000"}
TIMEPOINTS = ["TP1M", "TP2M", "TP3M", "TP4M"]
MIN_CELLS, MIN_COUNTS = 10, 1000


LFC_T = 1.0


def stars_for(padj, log2fc=None):
    """padj-only stars, unless log2fc is given -- then also requires
    |log2FC| > 1 to match the project's own "significant" definition
    (padj<0.05 & |log2FC|>1) used everywhere else (DEG README, results
    tables). Without this, e.g. pooled PIM1 at TP1M/TP2M shows *** from padj
    alone (0.0002) despite log2FC only 0.82-0.92 -- below the threshold that
    actually makes cluster 7's TP2M value (log2FC=1.31) the standout case."""
    if pd.isna(padj):
        return ""
    if log2fc is not None and (pd.isna(log2fc) or abs(log2fc) <= LFC_T):
        return ""
    if padj < 0.001:
        return "***"
    if padj < 0.01:
        return "**"
    if padj < 0.05:
        return "*"
    return ""


def load_sig(kind, cluster):
    rows = []
    if kind == "pooled":
        df = pd.read_csv(os.path.join(OUTPUT_DIR, "Pooled_Monocytes_ACS_vs_CCS_per_timepoint.csv"))
        for tp in TIMEPOINTS:
            sub = df[(df["gene"].isin(GENES)) & (df["timepoint"] == tp)]
            rows.append(sub[["gene", "timepoint", "log2FoldChange", "padj"]])
    else:
        for tp in TIMEPOINTS:
            df = pd.read_csv(os.path.join(OUTPUT_DIR, f"PaperClusters_ACS_{tp}_vs_CCS.csv"))
            sub = df[(df["cluster"] == cluster) & (df["gene"].isin(GENES))]
            rows.append(sub[["gene", "timepoint", "log2FoldChange", "padj"]])
    return pd.concat(rows, ignore_index=True)


def build_pseudobulk_cpm(adata_full, meta, kind, cluster):
    if kind == "pooled":
        cell_mask = (meta["cell_type_Scanorama"] == "Monocytes - CD14") & meta["condition"].isin(["CCS", "ACS_sterile"])
    else:
        cell_mask = (meta["paper_cluster"] == cluster) & meta["condition"].isin(["CCS", "ACS_sterile"])
    cell_mask = cell_mask & meta.index.isin(adata_full.obs_names)
    cells = meta.index[cell_mask]
    sub = adata_full[cells]
    counts = sub.layers["counts"]
    if issparse(counts):
        counts = counts.tocsr()
    samples = sub.obs["display_name"].astype(str).values
    sample_ids = pd.Index(pd.unique(samples))
    sample_to_row = {s: i for i, s in enumerate(sample_ids)}
    row_idx = np.array([sample_to_row[s] for s in samples])
    indicator = csr_matrix((np.ones(len(row_idx)), (row_idx, np.arange(len(row_idx)))),
                            shape=(len(sample_ids), len(row_idx)))
    pb_counts = indicator @ counts
    pb_counts = np.asarray(pb_counts.todense()) if issparse(pb_counts) else np.asarray(pb_counts)
    n_cells_per_sample = pd.Series(samples).value_counts().reindex(sample_ids).values
    total_counts_per_sample = pb_counts.sum(axis=1)
    keep = (n_cells_per_sample >= MIN_CELLS) & (total_counts_per_sample >= MIN_COUNTS)
    pb = pd.DataFrame(pb_counts[keep], index=sample_ids[keep], columns=sub.var_names)
    lib_size = pb.sum(axis=1)
    log1p_cpm = np.log1p(pb.div(lib_size, axis=0) * 1e6)
    sample_meta = (
        meta.loc[cells, ["display_name", "patient_id", "condition", "timepoint"]]
        .assign(display_name=lambda d: d["display_name"].astype(str))
        .drop_duplicates("display_name").set_index("display_name").loc[pb.index]
    )
    sample_meta["group"] = np.where(sample_meta["condition"] == "CCS", "CCS",
                                     "ACS " + sample_meta["timepoint"].astype(str))
    return log1p_cpm, sample_meta


def draw_gene_axis(ax, gene, log1p_cpm, sample_meta, sig_table, rng, title=None,
                    show_ylabel=False, title_fontsize=10):
    if gene not in log1p_cpm.columns:
        ax.axis("off")
        ax.set_title(f"{title or gene} (not tested)", fontsize=title_fontsize - 1)
        return
    gene_vals = log1p_cpm[gene]
    positions, violin_data = [], []
    for gi, group in enumerate(GROUP_ORDER):
        samples_in_group = sample_meta.index[sample_meta["group"] == group]
        vals = gene_vals.loc[samples_in_group].values
        if len(vals) == 0:
            continue
        positions.append(gi)
        violin_data.append(vals)
    if violin_data:
        parts = ax.violinplot(violin_data, positions=positions, showextrema=False, widths=0.8)
        for pi, body in zip(positions, parts["bodies"]):
            body.set_facecolor(GROUP_COLORS[GROUP_ORDER[pi]])
            body.set_edgecolor("none")
            body.set_alpha(0.6)
    for gi, group in enumerate(GROUP_ORDER):
        samples_in_group = sample_meta.index[sample_meta["group"] == group]
        vals = gene_vals.loc[samples_in_group].values
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), gi) + jitter, vals, s=8, color="black",
                   alpha=0.7, linewidths=0, zorder=3)

    ymax = gene_vals.max() if len(gene_vals) else 1.0
    ymax = ymax if ymax > 0 else 1.0

    # Comparison brackets: each ACS timepoint that is significant AND clears
    # |log2FC|>1 (the project's own "significant" definition -- padj alone
    # is not enough, see stars_for) gets a bracket back to CCS (x=0),
    # stacked upward so multiple brackets stay legible.
    sig_gis = []
    for gi, group in enumerate(GROUP_ORDER):
        if group == "CCS":
            continue
        tp = group.split(" ")[1]
        row = sig_table[(sig_table["gene"] == gene) & (sig_table["timepoint"] == tp)]
        if row.empty:
            continue
        s = stars_for(row["padj"].iloc[0], row["log2FoldChange"].iloc[0])
        if s:
            sig_gis.append((gi, s))

    n_sig = len(sig_gis)
    ax.set_ylim(0, ymax * (1.2 + 0.2 * n_sig))
    base_y = ymax * 1.08
    step = ymax * 0.2
    tick = step * 0.25
    for level, (gi, s) in enumerate(sig_gis):
        y = base_y + level * step
        ax.plot([0, 0, gi, gi], [y, y + tick, y + tick, y],
                color="black", linewidth=0.8, clip_on=False)
        ax.text(gi / 2, y + tick, s, ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_title(title or gene, fontsize=title_fontsize, fontweight="bold", pad=8)
    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels(GROUP_ORDER, fontsize=6, rotation=45, ha="right")
    ax.tick_params(axis="y", labelsize=7)
    if show_ylabel:
        ax.set_ylabel("Norm. expr.", fontsize=8, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)


def plot_gene(adata, meta, gene):
    fig, axes = plt.subplots(2, 2, figsize=(6.2, 6.0), sharex=False)
    rng = np.random.default_rng(0)

    for ri, (row_label, kind, cluster) in enumerate(ROWS):
        ax = axes[ri // 2][ri % 2]
        print(f"\n=== {gene}: {row_label} ===")
        log1p_cpm, sample_meta = build_pseudobulk_cpm(adata, meta, kind, cluster)
        print(f"{len(sample_meta)} pseudobulk samples: " + sample_meta["group"].value_counts().reindex(GROUP_ORDER).to_dict().__str__())
        sig_table = load_sig(kind, cluster)
        draw_gene_axis(ax, gene, log1p_cpm, sample_meta, sig_table, rng,
                        title=row_label, show_ylabel=(ri % 2 == 0))

    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g], alpha=0.6) for g in GROUP_ORDER]
    fig.legend(handles, GROUP_ORDER, loc="lower center", ncol=5, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Pooled Monocytes-CD14 vs. individual clusters: {gene}", fontsize=12, fontweight="bold", style="italic")
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    out_path = os.path.join(OUTPUT_DIR, f"violin_pooled_vs_cluster_{gene}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    print(f"\nSaved: {out_path}")


def plot_combined_2x2(adata, meta):
    """Both genes (PIM1, SOCS3) in one figure, 4 populations arranged 2x2 --
    two stacked (Pooled/Cluster 4) next to two more stacked (Cluster 6/
    Cluster 7) -- each population cell holding both genes side by side."""
    fig = plt.figure(figsize=(10.5, 10.0))
    outer = fig.add_gridspec(2, 2, hspace=0.65, wspace=0.35)
    rng = np.random.default_rng(0)
    pop_axes_pairs = []  # (row_label, [ax_pim1, ax_socs3]) for label placement after layout

    for ri, (row_label, kind, cluster) in enumerate(ROWS):
        print(f"\n=== combined: {row_label} ===")
        log1p_cpm, sample_meta = build_pseudobulk_cpm(adata, meta, kind, cluster)
        print(f"{len(sample_meta)} pseudobulk samples: " + sample_meta["group"].value_counts().reindex(GROUP_ORDER).to_dict().__str__())
        sig_table = load_sig(kind, cluster)

        inner = outer[ri // 2, ri % 2].subgridspec(1, len(GENES), wspace=0.5)
        gene_axes = []
        for gi, gene in enumerate(GENES):
            ax = fig.add_subplot(inner[gi])
            draw_gene_axis(ax, gene, log1p_cpm, sample_meta, sig_table, rng,
                            title=gene, show_ylabel=(gi == 0), title_fontsize=9)
            gene_axes.append(ax)
        pop_axes_pairs.append((row_label, gene_axes))

    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g], alpha=0.6) for g in GROUP_ORDER]
    fig.legend(handles, GROUP_ORDER, loc="lower center", ncol=5, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Pooled Monocytes-CD14 vs. individual clusters: PIM1 and SOCS3",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])

    # Population label centered above each pair of gene panels, placed after
    # tight_layout so the axes positions used are final.
    for row_label, gene_axes in pop_axes_pairs:
        x0 = min(ax.get_position().x0 for ax in gene_axes)
        x1 = max(ax.get_position().x1 for ax in gene_axes)
        y1 = max(ax.get_position().y1 for ax in gene_axes)
        fig.text((x0 + x1) / 2, y1 + 0.025, row_label, ha="center", fontsize=12, fontweight="bold")

    out_path = os.path.join(OUTPUT_DIR, "violin_pooled_vs_cluster_PIM1_SOCS3_2x2.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    print(f"\nSaved: {out_path}")
    return out_path


def main():
    print("Loading full h5ad + metadata...")
    adata = sc.read_h5ad(DATA_PATH)
    meta = pd.read_parquet(METADATA_PATH)
    meta = meta.copy()
    meta["paper_cluster"] = adata.obs["B2_Scanorama_Singlet_rb_mt_cluster"].astype(int).reindex(meta.index)

    for gene in GENES:
        plot_gene(adata, meta, gene)

    out_path = plot_combined_2x2(adata, meta)
    import shutil
    target = "/vol/disk/ubuntu/master_practicum_cytokines/lisa/FINAL_paper_comparison_plots/violin_pooled_vs_cluster_PIM1_SOCS3.png"
    shutil.copy(out_path, target)
    print(f"Copied to: {target}")


if __name__ == "__main__":
    main()
