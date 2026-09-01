import nbformat

SRC = "DEG/deg_conditions.ipynb"
DST = "/vol/disk/ubuntu/master_practicum_cytokines/sudenaz/deg_conditions_leiden10.ipynb"

NEW_CONFIG_SOURCE = '''DATA_PATH  = "/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/data/data_for_practicum_post_integration.h5ad"
OUTPUT_DIR = "/vol/disk/ubuntu/master_practicum_cytokines/sudenaz/deg_conditions_leiden10"

CELL_TYPE_COL = "cell_type_Scanorama"

CELL_TYPE_COLS = {
    "Harmony_leiden": "harmony_leiden",
    "DRVI_leiden":    "drvi_leiden",
}

SAMPLE_COL    = "display_name"
CONDITION_COL = "classification"
TIMEPOINT_COL = "timepoint"

PADJ_THRESH = 0.05
LFC_THRESH  = 1.0

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output: {OUTPUT_DIR}")
'''

nb = nbformat.read(SRC, as_version=4)

found_config = False
for cell in nb.cells:
    if cell.get("id") == "02-config":
        found_config = True
        cell["source"] = NEW_CONFIG_SOURCE
        cell["outputs"] = []
        cell["execution_count"] = None
        break

if not found_config:
    raise RuntimeError("Could not find config cell (id='02-config') -- stopping, nothing modified.")

nbformat.write(nb, DST)
print(f"Wrote: {DST}")
