"""
STEP 3: violin plots, one panel grid per paper cluster, matching Pekayvaz et
al. 2024's own figures. Pseudobulk log1p(CPM), one point per patient-
timepoint sample, stars from that cluster's own PyDESeq2 padj (no re-test).
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.sparse import issparse, csr_matrix

DATA_PATH = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/data/data_for_practicum_post_integration.h5ad"
METADATA_PATH = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/DEG/deg_metadata_shared.parquet"
INPUT_DIR = "/vol/disk/ubuntu/master_practicum_cytokines/lisa/paper_leiden_clusters_deg"
OUTPUT_DIR = INPUT_DIR

# gene rows mirror the paper's originating figures
GENE_ROWS = [
    ["IL6ST", "JAK1", "STAT3", "SOCS3"],          # Figs 3c, 3d
    ["EIF3E", "HINT1", "HMGB1"],                  # Fig 2e
    ["PIM1", "VCAN", "CD74", "UBC", "PSME2", "ODC1"],  # Fig 4d
]
GENES = [g for row in GENE_ROWS for g in row]
NCOLS = max(len(r) for r in GENE_ROWS)
NROWS = len(GENE_ROWS)

CLUSTERS = {
    0: "CD4+ T cells (cluster 0)",
    2: "CD4+ T cells (cluster 2)",
    5: "CD4+ T cells (cluster 5)",
    11: "Treg cells (cluster 11)",
    4: "CD14high monocytes (cluster 4)",
    6: "CD14high monocytes (cluster 6)",
    7: "CD14high monocytes (cluster 7)",
}

GROUP_ORDER = ["CCS", "ACS TP1M", "ACS TP2M", "ACS TP3M", "ACS TP4M"]
GROUP_COLORS = {"CCS": "#1f77b4", "ACS TP1M": "#fca082", "ACS TP2M": "#fb6a4a",
                "ACS TP3M": "#de2d26", "ACS TP4M": "#990000"}
TIMEPOINTS = ["TP1M", "TP2M", "TP3M", "TP4M"]
MIN_CELLS, MIN_COUNTS = 10, 1000


def stars_for(padj):
    if pd.isna(padj):
        return ""
    if padj < 0.001:
        return "***"
    if padj < 0.01:
        return "**"
    if padj < 0.05:
        return "*"
    return ""


def load_sig_table(cluster):
    rows = []
    for tp in TIMEPOINTS:
        df = pd.read_csv(os.path.join(INPUT_DIR, f"PaperClusters_ACS_{tp}_vs_CCS.csv"))
        sub = df[(df["cluster"] == cluster) & (df["gene"].isin(GENES))].copy()
        rows.append(sub[["gene", "timepoint", "log2FoldChange", "padj"]])
    return pd.concat(rows, ignore_index=True)


def build_pseudobulk_cpm(adata_full, meta, cluster):
    cell_mask = (
        (meta["paper_cluster"] == cluster)
        & (meta["condition"].isin(["CCS", "ACS_sterile"]))
        & meta.index.isin(adata_full.obs_names)
    )
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


def plot_cluster(log1p_cpm, sample_meta, sig_table, title, out_path):
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(1.4 * NCOLS, 1.15 * NCOLS + 1.6), sharex=False)
    rng = np.random.default_rng(0)

    for ri, row_genes in enumerate(GENE_ROWS):
        for ci in range(NCOLS):
            ax = axes[ri][ci]
            if ci >= len(row_genes):
                ax.axis("off")
                continue
            gene = row_genes[ci]
            if gene not in log1p_cpm.columns:
                ax.axis("off")
                ax.set_title(f"{gene}\n(not tested)", fontsize=7)
                continue
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
                ax.scatter(np.full(len(vals), gi) + jitter, vals, s=5, color="black",
                           alpha=0.7, linewidths=0, zorder=3)

            ymax = gene_vals.max() if len(gene_vals) else 1.0
            ymax = ymax if ymax > 0 else 1.0
            ax.set_ylim(0, ymax * 1.35)
            star_y = ymax * 1.1
            for gi, group in enumerate(GROUP_ORDER):
                if group == "CCS":
                    continue
                tp = group.split(" ")[1]
                row = sig_table[(sig_table["gene"] == gene) & (sig_table["timepoint"] == tp)]
                if row.empty:
                    continue
                s = stars_for(row["padj"].iloc[0])
                if s:
                    ax.text(gi, star_y, s, ha="center", va="bottom", fontsize=7, fontweight="bold")

            ax.set_title(gene, fontsize=8, fontweight="bold", style="italic", pad=14)
            ax.set_xticks(range(len(GROUP_ORDER)))
            if ri == NROWS - 1:
                ax.set_xticklabels(GROUP_ORDER, fontsize=5.5, rotation=45, ha="right")
            else:
                ax.set_xticklabels([])
            ax.tick_params(axis="y", labelsize=5.5)
            if ci == 0:
                ax.set_ylabel("Normalized\nexpression", fontsize=6)
            ax.spines[["top", "right"]].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g], alpha=0.6) for g in GROUP_ORDER]
    fig.legend(handles, GROUP_ORDER, loc="lower center", ncol=5, fontsize=6,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontsize=9, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    print("Loading full h5ad + metadata...")
    adata = sc.read_h5ad(DATA_PATH)
    meta = pd.read_parquet(METADATA_PATH)
    meta = meta.copy()
    meta["paper_cluster"] = adata.obs["B2_Scanorama_Singlet_rb_mt_cluster"].astype(int).reindex(meta.index)
    for col in ["condition", "timepoint", "display_name"]:
        pass

    for cluster, title in CLUSTERS.items():
        print(f"\n=== cluster {cluster} ({title}) ===")
        log1p_cpm, sample_meta = build_pseudobulk_cpm(adata, meta, cluster)
        print(f"{len(sample_meta)} pseudobulk samples after QC")
        print(sample_meta["group"].value_counts().reindex(GROUP_ORDER).to_string())
        sig_table = load_sig_table(cluster)
        out_path = os.path.join(OUTPUT_DIR, f"violin_paper_cluster_{cluster}.png")
        plot_cluster(log1p_cpm, sample_meta, sig_table, title, out_path)


if __name__ == "__main__":
    main()
