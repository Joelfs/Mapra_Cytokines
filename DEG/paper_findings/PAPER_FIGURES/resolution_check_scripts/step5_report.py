import pandas as pd

INPUT_DIR = '/vol/disk/ubuntu/master_practicum_cytokines/lisa/paper_leiden_clusters_deg'
GENES = ["IL6ST", "JAK1", "STAT3", "SOCS3", "EIF3E", "HINT1", "HMGB1", "PIM1", "VCAN", "CD74", "UBC", "PSME2", "ODC1"]
TIMEPOINTS = ["TP1M", "TP2M", "TP3M", "TP4M"]
CD4_CLUSTERS = [0, 2, 5, 11]
MONO_CLUSTERS = [4, 6, 7]
PADJ_T, LFC_T = 0.05, 1.0

# ── load cluster-level results ──────────────────────────────────────────
cl_rows = []
for tp in TIMEPOINTS:
    df = pd.read_csv(f'{INPUT_DIR}/PaperClusters_ACS_{tp}_vs_CCS.csv')
    d = df[df['gene'].isin(GENES) & df['cluster'].isin(CD4_CLUSTERS + MONO_CLUSTERS)].copy()
    cl_rows.append(d)
cluster_all = pd.concat(cl_rows, ignore_index=True)
cluster_all['sig'] = (cluster_all['padj'] < PADJ_T) & (cluster_all['log2FoldChange'].abs() > LFC_T)

pooled_cd4 = pd.read_csv(f'{INPUT_DIR}/Pooled_CD4_ACS_vs_CCS_per_timepoint.csv')
pooled_cd4 = pooled_cd4[pooled_cd4['gene'].isin(GENES)]
pooled_mono = pd.read_csv(f'{INPUT_DIR}/Pooled_Monocytes_ACS_vs_CCS_per_timepoint.csv')
pooled_mono = pooled_mono[pooled_mono['gene'].isin(GENES)]
pooled_all = pd.concat([pooled_cd4.assign(pop='CD4'), pooled_mono.assign(pop='Monocytes')], ignore_index=True)
pooled_all['sig'] = (pooled_all['padj'] < PADJ_T) & (pooled_all['log2FoldChange'].abs() > LFC_T)

# ── full table ───────────────────────────────────────────────────────────
full_table = cluster_all[['gene','cluster','timepoint','log2FoldChange','padj','sig']].copy()
full_table.to_csv(f'{INPUT_DIR}/step5_full_gene_x_cluster_x_timepoint.csv', index=False)
print(f"Saved full table ({len(full_table)} rows): step5_full_gene_x_cluster_x_timepoint.csv\n")

def pooled_lookup(pop, gene, tp):
    row = pooled_all[(pooled_all['population']==f'Pooled_{pop}') & (pooled_all['gene']==gene) & (pooled_all['timepoint']==tp)]
    return None if row.empty else row.iloc[0]

# ── condensed print: log2FC/padj per gene x cluster, all 4 timepoints ─────
print("="*100)
print("Cluster-level log2FC/padj (13-gene panel), and the matching pooled-population number")
print("="*100)
for gene in GENES:
    print(f"\n--- {gene} ---")
    for cl in CD4_CLUSTERS + MONO_CLUSTERS:
        pop = 'CD4' if cl in CD4_CLUSTERS else 'Monocytes'
        parts = []
        for tp in TIMEPOINTS:
            row = cluster_all[(cluster_all['gene']==gene) & (cluster_all['cluster']==cl) & (cluster_all['timepoint']==tp)]
            parts.append(f"{tp}: NT" if row.empty else f"{tp}: LFC={row['log2FoldChange'].iloc[0]:.2f},padj={row['padj'].iloc[0]:.3g}")
        print(f"  cluster {cl:2d} ({pop:9s}): " + " | ".join(parts))
    for pop in (['CD4'] if any(c in CD4_CLUSTERS for c in [0]) else []):
        pass
    parts = []
    for tp in TIMEPOINTS:
        row = pooled_lookup('CD4', gene, tp)
        parts.append(f"{tp}: NT" if row is None else f"{tp}: LFC={row['log2FoldChange']:.2f},padj={row['padj']:.3g}")
    print(f"  POOLED CD4          : " + " | ".join(parts))
    parts = []
    for tp in TIMEPOINTS:
        row = pooled_lookup('Monocytes', gene, tp)
        parts.append(f"{tp}: NT" if row is None else f"{tp}: LFC={row['log2FoldChange']:.2f},padj={row['padj']:.3g}")
    print(f"  POOLED Monocytes    : " + " | ".join(parts))

# ── significant at cluster but not pooled, and vice versa ────────────────
print("\n" + "="*100)
print("Cluster-significant (padj<0.05 & |LFC|>1) but NOT significant in the matching pooled population")
print("="*100)
any_found = False
for _, row in cluster_all[cluster_all['sig']].iterrows():
    pop = 'CD4' if row['cluster'] in CD4_CLUSTERS else 'Monocytes'
    prow = pooled_lookup(pop, row['gene'], row['timepoint'])
    pooled_sig = prow is not None and prow['sig']
    if not pooled_sig:
        any_found = True
        pooled_str = "not tested" if prow is None else f"LFC={prow['log2FoldChange']:.2f},padj={prow['padj']:.3g} (ns)"
        print(f"  {row['gene']:8s} cluster {row['cluster']:2d} ({pop:9s}) {row['timepoint']}  "
              f"LFC={row['log2FoldChange']:.2f} padj={row['padj']:.3g}   <- pooled {pop}: {pooled_str}")
if not any_found:
    print("  (none)")

print("\n" + "="*100)
print("Pooled-significant but NOT significant in ANY corresponding cluster, same timepoint")
print("="*100)
any_found = False
for _, prow in pooled_all[pooled_all['sig']].iterrows():
    pop = prow['population'].replace('Pooled_', '')
    clusters = CD4_CLUSTERS if pop == 'CD4' else MONO_CLUSTERS
    matches = cluster_all[(cluster_all['gene']==prow['gene']) & (cluster_all['timepoint']==prow['timepoint']) & (cluster_all['cluster'].isin(clusters))]
    if not matches['sig'].any():
        any_found = True
        print(f"  {prow['gene']:8s} POOLED {pop:9s} {prow['timepoint']}  LFC={prow['log2FoldChange']:.2f} padj={prow['padj']:.3g}"
              f"   <- no cluster in {clusters} reached significance")
if not any_found:
    print("  (none)")

# ── direction disagreements ───────────────────────────────────────────────
print("\n" + "="*100)
print("DIRECTION disagreements: cluster log2FC sign vs pooled log2FC sign (same gene/timepoint/population),")
print("restricted to cases where at least one side has |LFC|>0.2 (to skip near-zero noise)")
print("="*100)
any_found = False
for _, row in cluster_all.iterrows():
    pop = 'CD4' if row['cluster'] in CD4_CLUSTERS else 'Monocytes'
    prow = pooled_lookup(pop, row['gene'], row['timepoint'])
    if prow is None or pd.isna(row['log2FoldChange']) or pd.isna(prow['log2FoldChange']):
        continue
    c_lfc, p_lfc = row['log2FoldChange'], prow['log2FoldChange']
    if max(abs(c_lfc), abs(p_lfc)) < 0.2:
        continue
    if (c_lfc > 0) != (p_lfc > 0):
        any_found = True
        print(f"  {row['gene']:8s} cluster {row['cluster']:2d} ({pop:9s}) {row['timepoint']}: "
              f"cluster LFC={c_lfc:+.2f} (padj={row['padj']:.3g})  vs  pooled LFC={p_lfc:+.2f} (padj={prow['padj']:.3g})")
if not any_found:
    print("  (none)")
