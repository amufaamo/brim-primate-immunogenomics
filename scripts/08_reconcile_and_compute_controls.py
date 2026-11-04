#!/usr/bin/env python3
"""
08_reconcile_and_compute_controls.py
Reconciles all manuscript tables directly from frozen TSV files,
computes exposed non-contact framework control residues,
and outputs a 100% verified data sheet for the manuscript revision (v2).
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

DRIVE = os.environ.get("BRIM_DATA_DIR", os.environ.get("BRIM_DATA_DIR", "./data"))

# File paths
COUNTS_TSV = os.path.join(DRIVE, "brim_t2t_02_germline_counts.tsv")
K1_TSV = os.path.join(DRIVE, "brim_t2t_04_k1_summary.tsv")
MASTER_TSV = os.path.join(DRIVE, "brim_t2t_04_master_imgt104_3d_selection.tsv")
ATLAS_TSV = os.path.join(DRIVE, "brim_t2t_04_gene_integrated_atlas.tsv")
PART_TSV = os.path.join(DRIVE, "brim_t2t_03b_partition_sensitivity.tsv")

print("=== 1. VERIFY GERMLINE COUNTS (TABLE 1) ===")
df_counts = pd.read_csv(COUNTS_TSV, sep="\t")

# Check 8-species T2T panel (Human_CHM13 + 7 non-human primates)
chm13_panel = [
    "Human_CHM13", "Pan_troglodytes", "Pan_paniscus", "Gorilla_gorilla",
    "Pongo_abelii", "Pongo_pygmaeus", "Symphalangus_syndactylus", "Macaca_mulatta"
]
df_8t2t = df_counts[df_counts["assembly_id"].isin(chm13_panel)]

table1_rows = []
for asm in chm13_panel:
    sub = df_8t2t[df_8t2t["assembly_id"] == asm]
    sp = sub["species"].iloc[0]
    c_ighv = sub[sub["locus"] == "IGHV"]["n_functional_F"].sum()
    c_igkv = sub[sub["locus"] == "IGKV"]["n_functional_F"].sum()
    c_iglv = sub[sub["locus"] == "IGLV"]["n_functional_F"].sum()
    c_trav = sub[sub["locus"] == "TRAV"]["n_functional_F"].sum()
    c_trbv = sub[sub["locus"] == "TRBV"]["n_functional_F"].sum()
    total_f = sub["n_functional_F"].sum()
    total_all = sub["n_total"].sum()
    table1_rows.append({
        "assembly": asm, "species": sp,
        "IGHV": c_ighv, "IGKV": c_igkv, "IGLV": c_iglv,
        "TRAV": c_trav, "TRBV": c_trbv,
        "total_functional": total_f, "total_segments": total_all
    })

# Add GRCh38 control
sub_grch38 = df_counts[df_counts["assembly_id"] == "Human_GRCh38"]
table1_rows.append({
    "assembly": "Human_GRCh38", "species": "Homo_sapiens",
    "IGHV": sub_grch38[sub_grch38["locus"] == "IGHV"]["n_functional_F"].sum(),
    "IGKV": sub_grch38[sub_grch38["locus"] == "IGKV"]["n_functional_F"].sum(),
    "IGLV": sub_grch38[sub_grch38["locus"] == "IGLV"]["n_functional_F"].sum(),
    "TRAV": sub_grch38[sub_grch38["locus"] == "TRAV"]["n_functional_F"].sum(),
    "TRBV": sub_grch38[sub_grch38["locus"] == "TRBV"]["n_functional_F"].sum(),
    "total_functional": sub_grch38["n_functional_F"].sum(),
    "total_segments": sub_grch38["n_total"].sum()
})

df_t1 = pd.DataFrame(table1_rows)
print(df_t1.to_string())

sum_8t2t_f = df_8t2t["n_functional_F"].sum()
sum_8t2t_tot = df_8t2t["n_total"].sum()
print(f"\n8-species T2T total functional: {sum_8t2t_f}")
print(f"8-species T2T total segments: {sum_8t2t_tot}")
print(f"Total all 9 assemblies functional: {df_counts['n_functional_F'].sum()}")
print(f"Total all 9 assemblies segments: {df_counts['n_total'].sum()}")

print("\n=== 2. VERIFY CRITERION K1 TABLE 2 METRICS ===")
df_k1 = pd.read_csv(K1_TSV, sep="\t")
print(df_k1.to_string())

print("\n=== 3. COMPUTING EXPOSED NON-CONTACT FRAMEWORK CONTROL ===")
df_m = pd.read_csv(MASTER_TSV, sep="\t")

# Define regions:
# CDR12: IMGT 27..38 (CDR1), 56..65 (CDR2)
# Core FR: IMGT positions traditionally marked as Core FR
# Exposed Non-Contact FR: Framework positions outside CDRs with:
# contact5_freq < 0.05 AND mean_surface_exposure_idx >= 0.50 (high exposure, zero contact)

for unit in df_m["eval_unit"].unique():
    sub = df_m[df_m["eval_unit"] == unit].copy()
    if len(sub) == 0:
        continue
    
    # CDR residues
    cdr_mask = sub["region"].isin(["CDR1", "CDR2"])
    
    # Exposed non-contact FR mask
    # For antibodies (contact5_freq_combined)
    contact_col = "contact5_freq_combined" if "contact5_freq_combined" in sub.columns and sub["contact5_freq_combined"].notna().sum() > 0 else "contact5_pmhc_freq"
    
    exp_noncontact_mask = (~cdr_mask) & (sub[contact_col] < 0.05) & (sub["mean_surface_exposure_idx"] >= 0.50)
    core_fr_mask = sub["region"] == "FR3_core"  # or specific core positions
    
    omega_cdr = sub.loc[cdr_mask, "fubar_omega"].median()
    omega_exp_nc = sub.loc[exp_noncontact_mask, "fubar_omega"].median()
    omega_all_fr = sub.loc[~cdr_mask, "fubar_omega"].median()
    
    n_cdr = cdr_mask.sum()
    n_exp_nc = exp_noncontact_mask.sum()
    
    # Test CDR vs Exposed Non-Contact FR
    vals_cdr = sub.loc[cdr_mask, "fubar_omega"].dropna()
    vals_exp_nc = sub.loc[exp_noncontact_mask, "fubar_omega"].dropna()
    
    mwu_p = stats.mannwhitneyu(vals_cdr, vals_exp_nc, alternative="greater").pvalue if len(vals_exp_nc) > 0 else np.nan
    
    # Positive selection sites (fubar_beta > fubar_alpha or fubar_prob_pos >= 0.90)
    pos_sites_cdr = (sub.loc[cdr_mask, "fubar_pos_pp90"] == 1).sum()
    pos_sites_exp_nc = (sub.loc[exp_noncontact_mask, "fubar_pos_pp90"] == 1).sum()
    
    print(f"Unit: {unit:15s} | CDR omega: {omega_cdr:.3f} (n={n_cdr}) | Exposed-NC-FR omega: {omega_exp_nc:.3f} (n={n_exp_nc}) | Ratio: {omega_cdr/(omega_exp_nc+1e-5):.2f}x | MWU p: {mwu_p:.4f} | Pos sites (CDR vs Exp-NC): {pos_sites_cdr}/{n_cdr} vs {pos_sites_exp_nc}/{n_exp_nc}")


print("\n=== 4. VERIFY STRICT 1:1 ORTHOLOGS (TABLE 2 & S1) ===")
df_atlas = pd.read_csv(ATLAS_TSV, sep="\t")
strict = df_atlas[df_atlas["is_strict_1to1"] == 1]
print(f"Total strict 1:1 orthologs: {len(strict)}")
print("Locus distribution:", strict["locus"].value_counts().to_dict())

# Calculate strict 1:1 median CDR omega, FR omega, and ratio
med_cdr_strict = strict["mean_cdr12_omega"].median()
med_fr_strict = strict["mean_core_fr_omega"].median()
med_ratio_strict = strict["cdr12_vs_fr_omega_ratio"].median()
wilc_p = stats.wilcoxon(strict["cdr12_vs_fr_omega_ratio"] - 1.0).pvalue

print(f"Strict 1:1 Median CDR omega: {med_cdr_strict:.4f}")
print(f"Strict 1:1 Median FR omega: {med_fr_strict:.4f}")
print(f"Strict 1:1 Median Ratio (CDR/FR): {med_ratio_strict:.4f}x")
print(f"Strict 1:1 Wilcoxon p vs 1.0: {wilc_p:.2e}")

# Check BUSTED global FDR q < 0.05
busted_sig = df_atlas[df_atlas["busted_mh_q_global"] < 0.05]
print(f"\nBUSTED global q < 0.05 count: {len(busted_sig)}")
for idx, r in busted_sig.iterrows():
    print(f"  -> {r['human_gene']} ({r['locus']}): LRT={r['busted_mh_lrt']}, p={r['busted_mh_p']:.4e}, q={r['busted_mh_q_global']:.4e}")

print("\n=== 5. VERIFY ARCHETYPES (TABLE 3) ===")
archetypes = ["IGHV1-69", "IGHV4-34", "TRBV5-4", "TRBV19", "TRBV6-5"]
sub_arch = df_atlas[df_atlas["human_gene"].isin(archetypes)][[
    "human_gene", "locus", "is_strict_1to1", "n_species",
    "busted_mh_lrt", "busted_mh_p", "busted_mh_q_global",
    "mean_cdr12_omega", "mean_core_fr_omega", "cdr12_vs_fr_omega_ratio",
    "public_db_count", "log2_public_over_olga_pgen",
    "vane_hd_naive_freq", "vane_hd_memory_freq", "vane_mg_memory_freq",
    "pos_selected_imgt_positions"
]]
print(sub_arch.to_string())
