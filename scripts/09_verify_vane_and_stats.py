#!/usr/bin/env python3
"""
09_verify_vane_and_stats.py
Verify vane cohort structure, donor breakdown, and exact donor-level & library-level
statistics for IGHV4-34 and IGHV1-69 across HD Naive, HD Memory, and MG Memory B cells.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

def compute_stats(series, name=""):
    s = series.dropna()
    n = len(s)
    if n == 0:
        return {}
    mean = float(s.mean())
    std = float(s.std(ddof=1)) if n > 1 else 0.0
    median = float(s.median())
    q25 = float(s.quantile(0.25))
    q75 = float(s.quantile(0.75))
    iqr = q75 - q25
    min_val = float(s.min())
    max_val = float(s.max())
    return {
        "name": name,
        "n": n,
        "mean": mean,
        "std": std,
        "median": median,
        "q25": q25,
        "q75": q75,
        "iqr": iqr,
        "min": min_val,
        "max": max_val,
    }

def print_summary(st, unit="%"):
    mult = 100.0 if unit == "%" else 1.0
    print(f"  [{st['name']}] (n={st['n']}):")
    print(f"    Mean ± SD: {st['mean']*mult:.3f}{unit} ± {st['std']*mult:.3f}{unit}")
    print(f"    Median (IQR): {st['median']*mult:.3f}{unit} (IQR: {st['q25']*mult:.3f} - {st['q75']*mult:.3f}, IQR_width={st['iqr']*mult:.3f}{unit})")
    print(f"    Range: [{st['min']*mult:.3f}{unit}, {st['max']*mult:.3f}{unit}]")

def run_analysis(tsv_path):
    print("=" * 80)
    print(f"Loading vane summary dataset from: {tsv_path}")
    print("=" * 80)
    df = pd.read_csv(tsv_path, sep="\t")

    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check cohort and donor composition
    donors = df[['donor', 'cohort']].drop_duplicates()
    print("\n[Donor Breakdown by Cohort]")
    print(donors['cohort'].value_counts())
    print("\nDonors list:")
    for cohort, grp in donors.groupby('cohort'):
        print(f"  {cohort} (n={len(grp)}): {grp['donor'].tolist()}")

    # Libraries breakdown
    print("\n[Library Breakdown]")
    # Distinct cell-subset libraries: (donor, cohort, subset)
    subset_libs = df[['donor', 'cohort', 'subset']].drop_duplicates()
    print(f"Total donor-subset cell populations: {len(subset_libs)}")
    print(subset_libs.groupby(['cohort', 'subset']).size())

    # Distinct locus libraries: (donor, cohort, subset, locus)
    locus_libs = df[['donor', 'cohort', 'subset', 'locus']].drop_duplicates()
    print(f"\nTotal (donor, cohort, subset, locus) libraries: {len(locus_libs)}")
    print(locus_libs.groupby(['locus', 'subset']).size())

    # Check why n=62 libraries was reported
    # If 31 cell-subset libraries * 2 or 31 heavy + 31 light?
    print(f"\nBreakdown of locus_libs by locus:")
    print(locus_libs['locus'].value_counts())

    # Analyze target genes: IGHV4-34 and IGHV1-69
    for target_gene in ["IGHV4-34", "IGHV1-69"]:
        print("\n" + "#" * 80)
        print(f"STATISTICAL ANALYSIS FOR: {target_gene}")
        print("#" * 80)

        gdf = df[df['v_gene'] == target_gene].copy()
        print(f"Total rows for {target_gene}: {len(gdf)}")

        for metric, metric_name in [("freq_seq", "Sequence Frequency (freq_seq)"), ("freq_umi", "UMI Frequency (freq_umi)")]:
            print(f"\n--- Metric: {metric_name} ---")

            # 1. Library-level (each row in gdf is a library for IGH locus)
            igh_gdf = gdf[gdf['locus'] == 'IGH'].copy()

            hd_naive_lib = igh_gdf[(igh_gdf['cohort'] == 'HD') & (igh_gdf['subset'] == 'N')][metric]
            hd_mem_lib = igh_gdf[(igh_gdf['cohort'] == 'HD') & (igh_gdf['subset'] == 'M')][metric]
            mg_mem_lib = igh_gdf[(igh_gdf['cohort'] != 'HD') & (igh_gdf['subset'] == 'M')][metric]

            st_hd_n_lib = compute_stats(hd_naive_lib, f"{target_gene} HD Naive (Library-level)")
            st_hd_m_lib = compute_stats(hd_mem_lib, f"{target_gene} HD Memory (Library-level)")
            st_mg_m_lib = compute_stats(mg_mem_lib, f"{target_gene} MG Memory (Library-level)")

            print("\n[Library-Level Summary Statistics]")
            print_summary(st_hd_n_lib)
            print_summary(st_hd_m_lib)
            print_summary(st_mg_m_lib)

            # Hypothesis tests Library-level
            # HD Naive vs HD Mem
            u_n_m, p_mwu_n_m = stats.mannwhitneyu(hd_naive_lib, hd_mem_lib, alternative='two-sided')
            # HD Mem vs MG Mem
            u_m_mg, p_mwu_m_mg = stats.mannwhitneyu(hd_mem_lib, mg_mem_lib, alternative='two-sided')

            print("\n[Library-Level Two-Sided Mann-Whitney U Tests]")
            print(f"  HD Naive vs HD Mem: U = {u_n_m:.1f}, two-sided p = {p_mwu_n_m:.5e} (p = {p_mwu_n_m:.4f})")
            print(f"  HD Mem vs MG Mem:   U = {u_m_mg:.1f}, two-sided p = {p_mwu_m_mg:.5e} (p = {p_mwu_m_mg:.4f})")

            # 2. Donor-level (aggregate across runs per donor if any)
            donor_df = igh_gdf.groupby(['donor', 'cohort', 'subset'])[metric].mean().reset_index()

            hd_naive_dnr = donor_df[(donor_df['cohort'] == 'HD') & (donor_df['subset'] == 'N')][metric]
            hd_mem_dnr = donor_df[(donor_df['cohort'] == 'HD') & (donor_df['subset'] == 'M')][metric]
            mg_mem_dnr = donor_df[(donor_df['cohort'] != 'HD') & (donor_df['subset'] == 'M')][metric]

            st_hd_n_dnr = compute_stats(hd_naive_dnr, f"{target_gene} HD Naive (Donor-level)")
            st_hd_m_dnr = compute_stats(hd_mem_dnr, f"{target_gene} HD Memory (Donor-level)")
            st_mg_m_dnr = compute_stats(mg_mem_dnr, f"{target_gene} MG Memory (Donor-level)")

            print("\n[Donor-Level Summary Statistics]")
            print_summary(st_hd_n_dnr)
            print_summary(st_hd_m_dnr)
            print_summary(st_mg_m_dnr)

            # Hypothesis tests Donor-level
            u_n_m_d, p_mwu_n_m_d = stats.mannwhitneyu(hd_naive_dnr, hd_mem_dnr, alternative='two-sided')
            u_m_mg_d, p_mwu_m_mg_d = stats.mannwhitneyu(hd_mem_dnr, mg_mem_dnr, alternative='two-sided')

            print("\n[Donor-Level Two-Sided Mann-Whitney U Tests]")
            print(f"  HD Naive vs HD Mem (unpaired): U = {u_n_m_d:.1f}, two-sided p = {p_mwu_n_m_d:.5e} (p = {p_mwu_n_m_d:.4f})")
            print(f"  HD Mem vs MG Mem (unpaired):   U = {u_m_mg_d:.1f}, two-sided p = {p_mwu_m_mg_d:.5e} (p = {p_mwu_m_mg_d:.4f})")

            # Also check Paired Wilcoxon signed-rank for HD Naive vs HD Mem if matched donors exist
            paired_hd = donor_df[donor_df['cohort'] == 'HD'].pivot(index='donor', columns='subset', values=metric).dropna()
            if len(paired_hd) > 0:
                w_stat, p_wilc = stats.wilcoxon(paired_hd['N'], paired_hd['M'], alternative='two-sided')
                print(f"  HD Naive vs HD Mem (paired Wilcoxon, n={len(paired_hd)} pairs): W = {w_stat:.1f}, two-sided p = {p_wilc:.5e} (p = {p_wilc:.4f})")

            # Check Subtype breakdown within MG (AChR-MG vs MuSK-MG)
            mg_achr = donor_df[(donor_df['cohort'] == 'AChR_MG') & (donor_df['subset'] == 'M')][metric]
            mg_musk = donor_df[(donor_df['cohort'] == 'MuSK_MG') & (donor_df['subset'] == 'M')][metric]
            print(f"\n  [MG Subtypes Memory B cell Frequencies]")
            st_achr = compute_stats(mg_achr, "AChR_MG Mem")
            st_musk = compute_stats(mg_musk, "MuSK_MG Mem")
            print_summary(st_achr)
            print_summary(st_musk)
            if len(mg_achr) > 0 and len(mg_musk) > 0:
                u_sub, p_sub = stats.mannwhitneyu(mg_achr, mg_musk, alternative='two-sided')
                print(f"  AChR-MG Mem vs MuSK-MG Mem: U = {u_sub:.1f}, p = {p_sub:.4f}")

if __name__ == "__main__":
    tsv = "./data/vane_bcr_v_usage_summary.tsv"
    if not os.path.exists(tsv):
        tsv = "G:\\data\\brim_t2t_scripts\\vane_bcr_v_usage_summary.tsv"
    if len(sys.argv) > 1:
        tsv = sys.argv[1]
    run_analysis(tsv)
