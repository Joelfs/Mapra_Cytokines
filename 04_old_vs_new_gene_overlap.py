import pandas as pd
from pathlib import Path

OLD_DIR = Path("/vol/disk/ubuntu/master_practicum_cytokines/repo/Mapra_Cytokines/DEG/results")
NEW_DIR = Path("/vol/disk/ubuntu/master_practicum_cytokines/sudenaz/deg_conditions_leiden10")
OUT_CSV = NEW_DIR / "old_vs_new_gene_overlap.csv"

GROUPINGS = ["Harmony_leiden", "DRVI_leiden"]
COMPARISONS = ["ACS_vs_CCS", "ACS_vs_nonCCS", "ACS_TP1M_vs_TP4M"]
PADJ_THRESH = 0.05
LFC_THRESH = 1.0

def sig_genes_with_direction(path):
    df = pd.read_csv(path)
    if "significant" not in df.columns:
        df["significant"] = (df["padj"] < PADJ_THRESH) & (df["log2FoldChange"].abs() > LFC_THRESH)
    sig = df[df["significant"]].sort_values("padj")
    first_per_gene = sig.drop_duplicates(subset="gene", keep="first")
    genes = set(first_per_gene["gene"])
    fc_by_gene = first_per_gene.set_index("gene")["log2FoldChange"]
    return genes, fc_by_gene

rows = []
for grouping in GROUPINGS:
    for comparison in COMPARISONS:
        old_path = OLD_DIR / f"{grouping}_{comparison}.csv"
        new_path = NEW_DIR / f"1.0{grouping}_{comparison}.csv"
        if not old_path.exists():
            print(f"MISSING old file: {old_path}")
            continue
        if not new_path.exists():
            print(f"MISSING new file: {new_path}")
            continue
        old_genes, old_fc = sig_genes_with_direction(old_path)
        new_genes, new_fc = sig_genes_with_direction(new_path)
        shared = old_genes & new_genes
        old_only = old_genes - new_genes
        new_only = new_genes - old_genes
        agree = sum(1 for g in shared if (old_fc[g] > 0) == (new_fc[g] > 0))
        rows.append({
            "grouping": grouping,
            "comparison": comparison,
            "n_sig_old_unique_genes": len(old_genes),
            "n_sig_new_unique_genes": len(new_genes),
            "n_shared": len(shared),
            "n_old_only": len(old_only),
            "n_new_only": len(new_only),
            "pct_of_old_still_significant": round(len(shared) / len(old_genes) * 100, 1) if old_genes else None,
            "pct_shared_agree_direction": round(agree / len(shared) * 100, 1) if shared else None,
        })

summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
summary.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")
