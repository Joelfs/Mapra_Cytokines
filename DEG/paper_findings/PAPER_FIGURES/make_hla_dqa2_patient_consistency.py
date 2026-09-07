"""
HLA-DQA2 within-patient consistency check, ACS patients, CD14 monocytes.

Question: does a patient's HLA-DQA2 on/off status stay the same across their
own repeat timepoints (consistent with a fixed genotype effect), or does it
switch within a patient over time (which would argue for a disease-driven
regulatory change instead)?

Pseudobulk per (patient_id, timepoint) sample, CD14 monocytes only, all ACS
clinical courses pooled (sterile/infection/subacute) plus CCS for reference.
"On" is defined as CPM > 1, matching the threshold used for the other
HLA-DQA2 zero-inflation checks.
"""
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.sparse import issparse, csr_matrix

DATA_PATH = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/data/data_for_practicum_post_integration.h5ad"
METADATA_PATH = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/DEG/deg_metadata_shared.parquet"
OUT = "/vol/disk/mofa_lung/tmp/claude-1000/-vol-disk-ubuntu-master-practicum-cytokines/9c1eaef7-ac91-459c-8966-313e6b45b553/scratchpad/HLA-DQA2_patient_consistency.png"

GENE = "HLA-DQA2"
CELL_TYPE = "Monocytes - CD14"
ON_THRESHOLD = 1.0  # CPM
MIN_CELLS = 10

TP_ORDER = ["TP0M", "TP1M", "TP2M", "TP3M", "TP4M"]

print("Loading data...")
adata = sc.read_h5ad(DATA_PATH)
meta = pd.read_parquet(METADATA_PATH)

mask = (meta["cell_type_DRVI"] == CELL_TYPE) & meta.index.isin(adata.obs_names)
mask &= meta["condition"].isin(["CCS", "ACS_sterile", "ACS_infection", "ACS_subacute"])
cells = meta.index[mask]
sub = adata[cells]
counts = sub.layers["counts"]
if issparse(counts):
    counts = counts.tocsr()

gene_idx = sub.var_names.get_loc(GENE)

sample_key = meta.loc[cells, "patient_id"].astype(str) + "|" + meta.loc[cells, "timepoint"].astype(str)
sample_key = sample_key.values
sample_ids = pd.Index(pd.unique(sample_key))
s2row = {s: i for i, s in enumerate(sample_ids)}
row_idx = np.array([s2row[s] for s in sample_key])
indicator = csr_matrix((np.ones(len(row_idx)), (row_idx, np.arange(len(row_idx)))),
                        shape=(len(sample_ids), len(row_idx)))
pb_counts = indicator @ counts
pb_counts = np.asarray(pb_counts.todense()) if issparse(pb_counts) else np.asarray(pb_counts)

n_cells = pd.Series(sample_key).value_counts().reindex(sample_ids).values
lib_size = pb_counts.sum(axis=1)
keep = (n_cells >= MIN_CELLS) & (lib_size > 0)

gene_counts = pb_counts[:, gene_idx]
cpm = gene_counts / lib_size * 1e6

df = pd.DataFrame({"sample_key": sample_ids, "cpm": cpm, "n_cells": n_cells, "lib_size": lib_size})
df = df[keep].copy()
df["patient_id"] = df["sample_key"].str.split("|").str[0]
df["timepoint"] = df["sample_key"].str.split("|").str[1]

pid_cond = meta.loc[cells, ["patient_id", "condition"]].drop_duplicates("patient_id").set_index("patient_id")["condition"]
df["condition"] = df["patient_id"].map(pid_cond.astype(str))
df["is_acs"] = df["condition"].str.startswith("ACS")
df["on"] = df["cpm"] > ON_THRESHOLD

acs = df[df["is_acs"]].copy()
tp_counts = acs.groupby("patient_id")["timepoint"].nunique()
multi_tp_patients = tp_counts[tp_counts >= 2].index
acs_multi = acs[acs["patient_id"].isin(multi_tp_patients)].copy()

def consistent(g):
    return g["on"].nunique() == 1

per_patient_consistency = acs_multi.groupby("patient_id").apply(consistent)
n_consistent = per_patient_consistency.sum()
n_total = len(per_patient_consistency)
print(f"ACS patients with >=2 CD14-monocyte timepoints: {n_total}")
print(f"Consistent on/off status across all their timepoints: {n_consistent} ({100*n_consistent/n_total:.0f}%)")
print(per_patient_consistency)

# ---- Plot: one line per patient across timepoints ----
fig, ax = plt.subplots(figsize=(7, 4.5))
tp_x = {tp: i for i, tp in enumerate(TP_ORDER)}
acs_multi["log1p_cpm"] = np.log1p(acs_multi["cpm"])
log_threshold = np.log1p(ON_THRESHOLD)

for pid, g in acs_multi.groupby("patient_id"):
    g = g.sort_values("timepoint", key=lambda s: s.map(tp_x))
    xs = g["timepoint"].map(tp_x).values
    ys = g["log1p_cpm"].values
    color = "#4C72B0" if consistent(g) else "#DD8452"
    ax.plot(xs, ys, color=color, alpha=0.6, linewidth=1.2, zorder=2)
    ax.scatter(xs, ys, color=color, s=28, zorder=3)

ax.axhline(log_threshold, color="grey", linestyle="--", linewidth=1, zorder=1)
ax.text(len(TP_ORDER) - 1.05, log_threshold + 0.15, "on/off threshold (1 CPM)",
        ha="right", va="bottom", fontsize=8, color="grey")

ax.set_xticks(range(len(TP_ORDER)))
ax.set_xticklabels(TP_ORDER)
ax.set_xlim(-0.3, len(TP_ORDER) - 0.7)
ax.set_ylabel("HLA-DQA2, CD14 monocytes\nlog(1 + CPM)")
ax.set_title(f"HLA-DQA2 on/off status is mostly stable within a patient over time\n"
             f"{n_consistent}/{n_total} ACS patients ({100*n_consistent/n_total:.0f}%) never switch status "
             f"across their sampled timepoints",
             fontsize=11, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)

from matplotlib.lines import Line2D
handles = [Line2D([0], [0], color="#4C72B0", lw=2, label="stays on or stays off"),
           Line2D([0], [0], color="#DD8452", lw=2, label="switches status")]
ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=False)

fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT}")
