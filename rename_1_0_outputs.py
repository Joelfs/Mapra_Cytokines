import os
import shutil

OUTPUT_DIR = "/vol/disk/ubuntu/master_practicum_cytokines/sudenaz/deg_conditions_leiden10"
MARKERS = ["Harmony_leiden", "DRVI_leiden"]
PREFIX = "1.0"

renamed = []
for fname in sorted(os.listdir(OUTPUT_DIR)):
    full_path = os.path.join(OUTPUT_DIR, fname)
    if not os.path.isfile(full_path) or fname.startswith(PREFIX):
        continue
    if any(marker in fname for marker in MARKERS):
        new_path = os.path.join(OUTPUT_DIR, PREFIX + fname)
        shutil.move(full_path, new_path)
        renamed.append((fname, PREFIX + fname))

print(f"Renamed {len(renamed)} files:")
for old, new in renamed:
    print(f"  {old}  ->  {new}")
