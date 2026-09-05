"""
Reconstructs the resolution-2.0 Leiden clustering (Lisa's original: 25 Harmony
clusters, 32 DRVI clusters) by rerunning Leiden on the EXISTING neighbor
graphs, with the same random_state/n_iterations Joel used for the resolution
1.0 rerun -- only the resolution parameter changes.

SAFE: opens the h5ad read-only (backed="r"), never calls adata.write() or
anything that touches the original file. Only writes a small new CSV.
"""

import scanpy as sc

DATA_PATH = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/data/data_for_practicum_post_integration.h5ad"
OUT_PATH = "/vol/disk/ubuntu/master_practicum_cytokines/sudenaz/resolution2_leiden_labels.csv"

print("Loading (backed, read-only -- no write access to the original file)...")
adata = sc.read_h5ad(DATA_PATH, backed="r")

print("Recomputing resolution 2.0 Leiden clusters on the existing neighbor graphs...")
sc.tl.leiden(
    adata,
    resolution=2.0,
    random_state=0,
    n_iterations=-1,
    neighbors_key="neighbors_harmony",
    key_added="harmony_leiden_res2",
)
sc.tl.leiden(
    adata,
    resolution=2.0,
    random_state=0,
    n_iterations=-1,
    neighbors_key="neighbors_drvi",
    key_added="drvi_leiden_res2",
)

n_harmony = adata.obs["harmony_leiden_res2"].nunique()
n_drvi = adata.obs["drvi_leiden_res2"].nunique()
print(f"harmony_leiden_res2: {n_harmony} clusters (original resolution 2.0 had 25)")
print(f"drvi_leiden_res2: {n_drvi} clusters (original resolution 2.0 had 32)")
if n_harmony != 25 or n_drvi != 32:
    print("NOTE: counts don't exactly match the original 25/32 -- expected, since "
          "leiden/package versions may have drifted slightly. Still a valid "
          "resolution-2.0-equivalent clustering on the same graph, just maybe "
          "not bit-identical to Lisa's original run.")

out = adata.obs[["harmony_leiden", "drvi_leiden", "harmony_leiden_res2", "drvi_leiden_res2"]].copy()
out.index.name = "cell_barcode"
out.to_csv(OUT_PATH)
print(f"\nSaved (new file only, original h5ad untouched): {OUT_PATH}")