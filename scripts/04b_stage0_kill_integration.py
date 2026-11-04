#!/usr/bin/env python3
"""
04b_stage0_kill_integration.py
Master statistical evaluation of Stage 0 Pre-Registered Go/No-Go Criteria (K1, K2, K3)
across all 8 T2T primate species and all 5 adaptive immune receptor loci (IGHV, IGKV, IGLV, TRAV, TRBV).

All numbers reported in manuscript/logs are produced deterministically by this script from:
- 02_germline_v/t2t_primate_germline_v_catalog.tsv & t2t_primate_germline_v_counts_summary.tsv
- 03_selection/tier1_site_selection_imgt104.tsv, tier2_orthogroup_selection.tsv, tier2_orthogroup_sites.tsv
- 04_structure_repertoire/structural_contacts_bcr_imgt104.tsv & structural_contacts_tcr_imgt104.tsv
- 04_structure_repertoire/olga_human_v_pgen.tsv, CoV-AbDab_080224.csv, vdjdb.txt, vane_bcr_v_usage_summary.tsv
"""
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

ROOT = os.environ.get("BRIM_DATA_DIR", os.environ.get("BRIM_DATA_DIR", "./data"))
OUT_DIR = os.path.join(ROOT, "04_structure_repertoire", "kill_eval")
os.makedirs(OUT_DIR, exist_ok=True)


def partial_spearman(x, y, z):
    """Exact partial Spearman rank correlation between x and y controlling for z."""
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    n = len(x)
    if n < 6:
        return np.nan, np.nan
    rx = sp_stats.rankdata(x)
    ry = sp_stats.rankdata(y)
    rz = sp_stats.rankdata(z)
    # Residualize rx and ry on rz
    bx = np.polyfit(rz, rx, 1)
    by = np.polyfit(rz, ry, 1)
    res_x = rx - np.polyval(bx, rz)
    res_y = ry - np.polyval(by, rz)
    r, _ = sp_stats.pearsonr(res_x, res_y)
    t_stat = r * math.sqrt((n - 3) / max(1e-12, 1.0 - r * r))
    p_val = 2.0 * sp_stats.t.sf(abs(t_stat), df=n - 3)
    return float(r), float(p_val)


def evaluate_k1():
    """Criterion K1: Germline CDR1+CDR2 (and HV4) vs Core FR positive selection across Tier 1 & Strict 1:1 Tier 2."""
    t1 = pd.read_csv(os.path.join(ROOT, "03_selection", "tier1_site_selection_imgt104.tsv"), sep="\t")
    t2 = pd.read_csv(os.path.join(ROOT, "03_selection", "tier2_orthogroup_selection.tsv"), sep="\t")
    t2_sites = pd.read_csv(os.path.join(ROOT, "03_selection", "tier2_orthogroup_sites.tsv"), sep="\t")

    k1_rows = []
    for unit, df in t1.groupby("unit"):
        cdr = df[df["region"].isin(["CDR1", "CDR2"])]
        fr = df[df["region"].isin(["FR1", "FR2", "FR3_core"])]
        hv4 = df[df["region"] == "HV4"]

        # Mann-Whitney U (one-sided: CDR > FR) on FUBAR omega and FEL omega
        u_fub, p_mwu_fub = sp_stats.mannwhitneyu(cdr["fubar_omega"], fr["fubar_omega"], alternative="greater")
        u_fel, p_mwu_fel = sp_stats.mannwhitneyu(cdr["fel_omega_reg"], fr["fel_omega_reg"], alternative="greater")

        # Fisher exact test on MEME p<0.05 sites
        a_m = int(cdr["meme_pos_p05"].sum())
        b_m = len(cdr) - a_m
        c_m = int(fr["meme_pos_p05"].sum())
        d_m = len(fr) - c_m
        or_meme, p_fish_meme = sp_stats.fisher_exact([[a_m, b_m], [c_m, d_m]], alternative="greater")

        # Fisher exact test on FUBAR PP>=0.90 sites
        a_f = int(cdr["fubar_pos_pp90"].sum())
        b_f = len(cdr) - a_f
        c_f = int(fr["fubar_pos_pp90"].sum())
        d_f = len(fr) - c_f
        or_fub, p_fish_fub = sp_stats.fisher_exact([[a_f, b_f], [c_f, d_f]], alternative="greater")

        # Fisher exact test on FEL pervasive positive p<0.05
        a_fl = int(cdr["fel_pos_p05"].sum())
        b_fl = len(cdr) - a_fl
        c_fl = int(fr["fel_pos_p05"].sum())
        d_fl = len(fr) - c_fl
        or_fel, p_fish_fel = sp_stats.fisher_exact([[a_fl, b_fl], [c_fl, d_fl]], alternative="greater")

        med_cdr = float(cdr["fubar_omega"].median())
        med_fr = float(fr["fubar_omega"].median())
        med_hv4 = float(hv4["fubar_omega"].median()) if len(hv4) else np.nan

        k1_rows.append({
            "dataset": unit,
            "locus": df["locus"].iloc[0],
            "n_unique_seqs": int(df["n_unique_seqs"].iloc[0]),
            "n_cdr12_codons": len(cdr),
            "n_core_fr_codons": len(fr),
            "median_omega_cdr12": round(med_cdr, 4),
            "median_omega_core_fr": round(med_fr, 4),
            "median_omega_hv4": round(med_hv4, 4),
            "omega_median_ratio_cdr_vs_fr": round(med_cdr / max(med_fr, 1e-6), 3),
            "mwu_p_fubar_omega": p_mwu_fub,
            "mwu_p_fel_omega": p_mwu_fel,
            "meme_p05_cdr12": f"{a_m}/{len(cdr)} ({100*a_m/len(cdr):.1f}%)",
            "meme_p05_core_fr": f"{c_m}/{len(fr)} ({100*c_m/len(fr):.1f}%)",
            "meme_or": round(float(or_meme), 3) if np.isfinite(or_meme) else "Inf",
            "meme_fisher_p": p_fish_meme,
            "fel_pos_cdr12": f"{a_fl}/{len(cdr)}",
            "fel_pos_core_fr": f"{c_fl}/{len(fr)}",
            "fel_fisher_p": p_fish_fel,
            "fubar_pp90_cdr12": f"{a_f}/{len(cdr)}",
            "fubar_pp90_core_fr": f"{c_f}/{len(fr)}",
            "fubar_fisher_p": p_fish_fub,
        })

    # Strict 1:1 syntenic orthologs pooled site check (paralog-free control!)
    for label, sub_sites in [
        ("Strict1to1_AllLoci", t2_sites[t2_sites["is_strict_1to1"] == 1]),
        ("Strict1to1_TRBV", t2_sites[(t2_sites["is_strict_1to1"] == 1) & (t2_sites["locus"] == "TRBV")]),
        ("Strict1to1_IGHV", t2_sites[(t2_sites["is_strict_1to1"] == 1) & (t2_sites["locus"] == "IGHV")]),
        ("Strict1to1_TRAV", t2_sites[(t2_sites["is_strict_1to1"] == 1) & (t2_sites["locus"] == "TRAV")]),
        ("AllOrthogroups_Pooled", t2_sites),
    ]:
        if len(sub_sites) < 30:
            continue
        cdr = sub_sites[sub_sites["region"].isin(["CDR1", "CDR2"])]
        fr = sub_sites[sub_sites["region"].isin(["FR1", "FR2", "FR3_core"])]
        hv4 = sub_sites[sub_sites["region"] == "HV4"]
        _, p_mwu = sp_stats.mannwhitneyu(cdr["fubar_omega"], fr["fubar_omega"], alternative="greater")
        a_m = int((cdr["meme_p"] < 0.05).sum())
        c_m = int((fr["meme_p"] < 0.05).sum())
        or_m, p_fm = sp_stats.fisher_exact([[a_m, len(cdr) - a_m], [c_m, len(fr) - c_m]], alternative="greater")
        a_f = int((cdr["fubar_prob_pos"] >= 0.90).sum())
        c_f = int((fr["fubar_prob_pos"] >= 0.90).sum())
        _, p_ff = sp_stats.fisher_exact([[a_f, len(cdr) - a_f], [c_f, len(fr) - c_f]], alternative="greater")
        med_c, med_f = float(cdr["fubar_omega"].median()), float(fr["fubar_omega"].median())
        k1_rows.append({
            "dataset": label,
            "locus": label.split("_")[-1],
            "n_unique_seqs": int(sub_sites["orthogroup"].nunique()),
            "n_cdr12_codons": len(cdr),
            "n_core_fr_codons": len(fr),
            "median_omega_cdr12": round(med_c, 4),
            "median_omega_core_fr": round(med_f, 4),
            "median_omega_hv4": round(float(hv4["fubar_omega"].median()), 4) if len(hv4) else np.nan,
            "omega_median_ratio_cdr_vs_fr": round(med_c / max(med_f, 1e-6), 3),
            "mwu_p_fubar_omega": p_mwu,
            "mwu_p_fel_omega": np.nan,
            "meme_p05_cdr12": f"{a_m}/{len(cdr)} ({100*a_m/len(cdr):.2f}%)",
            "meme_p05_core_fr": f"{c_m}/{len(fr)} ({100*c_m/len(fr):.2f}%)",
            "meme_or": round(float(or_m), 3) if np.isfinite(or_m) else "Inf",
            "meme_fisher_p": p_fm,
            "fel_pos_cdr12": "NA",
            "fel_pos_core_fr": "NA",
            "fel_fisher_p": np.nan,
            "fubar_pp90_cdr12": f"{a_f}/{len(cdr)}",
            "fubar_pp90_core_fr": f"{c_f}/{len(fr)}",
            "fubar_fisher_p": p_ff,
        })

    k1_df = pd.DataFrame(k1_rows)
    k1_df.to_csv(os.path.join(OUT_DIR, "k1_cdr_vs_fr_selection_summary.tsv"), sep="\t", index=False)
    return k1_df, t1, t2, t2_sites


def evaluate_k2(t1: pd.DataFrame):
    """Criterion K2: Primate site selection (IMGT 1..104) vs Empirical 3D Antigen / Peptide / MHC Contact Frequency."""
    bcr_st = pd.read_csv(os.path.join(ROOT, "04_structure_repertoire", "structural_contacts_bcr_imgt104.tsv"), sep="\t")
    tcr_st = pd.read_csv(os.path.join(ROOT, "04_structure_repertoire", "structural_contacts_tcr_imgt104.tsv"), sep="\t")

    k2_rows = []
    merged_profiles = []

    # BCR: IGHV (MetaClans + ClanI/II/III), IGKV_All, IGLV_All
    bcr_units = [
        ("IGHV_MetaClans", "IGHV"),
        ("IGHV_ClanI", "IGHV"),
        ("IGHV_ClanII", "IGHV"),
        ("IGHV_ClanIII", "IGHV"),
        ("IGKV_All", "IGKV"),
        ("IGLV_All", "IGLV"),
    ]
    for uname, loc in bcr_units:
        sub_sel = t1[t1["unit"] == uname]
        sub_str = bcr_st[bcr_st["locus"] == loc]
        m = pd.merge(sub_sel, sub_str, on=["locus", "imgt_pos", "region"], how="inner")
        m = m[m["n_combined"] >= 20].copy()
        if len(m) < 20:
            continue
        merged_profiles.append(m.assign(eval_unit=uname))

        for contact_col, label in [
            ("contact5_freq_combined", "AbAg_Combined_5A"),
            ("contact5_freq_sabdab_noncov", "SAbDab2_NonCoV_General_5A"),
            ("contact5_freq_covabdab", "CoVAbDab_5A"),
        ]:
            if m[contact_col].isna().all() or (m[contact_col] == 0).all():
                continue
            rho_om, p_om = sp_stats.spearmanr(m["fubar_omega"], m[contact_col])
            rho_fel, p_fel = sp_stats.spearmanr(m["fel_omega_reg"], m[contact_col])
            prho_om, pp_om = partial_spearman(
                m["fubar_omega"].to_numpy(dtype=float),
                m[contact_col].to_numpy(dtype=float),
                m["mean_surface_exposure_idx"].to_numpy(dtype=float),
            )
            # Restrict strictly to surface-exposed top 50% positions
            med_exp = m["mean_surface_exposure_idx"].median()
            surf = m[m["mean_surface_exposure_idx"] >= med_exp]
            rho_surf, p_surf = sp_stats.spearmanr(surf["fubar_omega"], surf[contact_col])

            k2_rows.append({
                "unit": uname,
                "locus": loc,
                "contact_metric": label,
                "n_imgt_sites": len(m),
                "spearman_rho_fubar_omega": round(float(rho_om), 4),
                "spearman_p_fubar_omega": float(p_om),
                "spearman_rho_fel_omega": round(float(rho_fel), 4),
                "spearman_p_fel_omega": float(p_fel),
                "partial_rho_controlling_surface_exposure": round(prho_om, 4),
                "partial_p_controlling_surface_exposure": pp_om,
                "surface_only_top50pct_rho": round(float(rho_surf), 4),
                "surface_only_top50pct_p": float(p_surf),
            })

    # TCR: TRBV_All & TRAV_All vs pMHC total, Peptide-only, MHC-only!
    for uname, loc in [("TRBV_All", "TRBV"), ("TRAV_All", "TRAV")]:
        sub_sel = t1[t1["unit"] == uname]
        sub_str = tcr_st[tcr_st["locus"] == loc]
        m = pd.merge(sub_sel, sub_str, on=["locus", "imgt_pos", "region"], how="inner")
        m = m[m["n_structures"] >= 20].copy()
        if len(m) < 20:
            continue
        merged_profiles.append(m.assign(eval_unit=uname))

        for contact_col, label in [
            ("contact5_pmhc_freq", "TCR_pMHC_Total_5A"),
            ("contact5_mhc_freq", "TCR_MHC_Helix_5A"),
            ("contact5_peptide_freq", "TCR_Peptide_5A"),
        ]:
            rho_om, p_om = sp_stats.spearmanr(m["fubar_omega"], m[contact_col])
            rho_fel, p_fel = sp_stats.spearmanr(m["fel_omega_reg"], m[contact_col])
            prho_om, pp_om = partial_spearman(
                m["fubar_omega"].to_numpy(dtype=float),
                m[contact_col].to_numpy(dtype=float),
                m["mean_surface_exposure_idx"].to_numpy(dtype=float),
            )
            med_exp = m["mean_surface_exposure_idx"].median()
            surf = m[m["mean_surface_exposure_idx"] >= med_exp]
            rho_surf, p_surf = sp_stats.spearmanr(surf["fubar_omega"], surf[contact_col])

            k2_rows.append({
                "unit": uname,
                "locus": loc,
                "contact_metric": label,
                "n_imgt_sites": len(m),
                "spearman_rho_fubar_omega": round(float(rho_om), 4),
                "spearman_p_fubar_omega": float(p_om),
                "spearman_rho_fel_omega": round(float(rho_fel), 4),
                "spearman_p_fel_omega": float(p_fel),
                "partial_rho_controlling_surface_exposure": round(prho_om, 4),
                "partial_p_controlling_surface_exposure": pp_om,
                "surface_only_top50pct_rho": round(float(rho_surf), 4),
                "surface_only_top50pct_p": float(p_surf),
            })

    k2_df = pd.DataFrame(k2_rows)
    k2_df.to_csv(os.path.join(OUT_DIR, "k2_3d_contact_correlations.tsv"), sep="\t", index=False)
    if merged_profiles:
        pd.concat(merged_profiles, ignore_index=True).to_csv(
            os.path.join(OUT_DIR, "master_imgt104_selection_and_3d_contacts.tsv"), sep="\t", index=False
        )
    return k2_df


def evaluate_k3(t2: pd.DataFrame, t2_sites: pd.DataFrame):
    """
    Criterion K3: Human Public Repertoire (CoV-AbDab, SAbDab2, VDJdb) & FACS Naive/Memory (`vane` HD & MG)
    controlling for OLGA baseline recombination probability P_gen(V).
    """
    olga = pd.read_csv(os.path.join(ROOT, "04_structure_repertoire", "olga_human_v_pgen.tsv"), sep="\t")
    olga_gene = olga[olga["level"] == "gene"].set_index(["locus", "gene"])["pgen_v"].to_dict()

    # 1. CoV-AbDab human antibody V-gene counts
    cov_counts = defaultdict(Counter)
    cov_neut_counts = defaultdict(Counter)
    with open(os.path.join(ROOT, "04_structure_repertoire", "CoV-AbDab_080224.csv"), "r", encoding="utf-8-sig", errors="replace") as fh:
        for r in csv.DictReader(fh):
            if (r.get("Ab or Nb") or "").strip() != "Ab":
                continue
            origin = (r.get("Origin") or "").lower()
            h_raw = (r.get("Heavy V Gene") or "").strip()
            l_raw = (r.get("Light V Gene") or "").strip()
            is_neut = bool((r.get("Neutralising Vs") or "").strip())
            if "human" in h_raw.lower() or ("human" in origin and "mouse" not in h_raw.lower() and "macaque" not in h_raw.lower()):
                m = re.search(r"(IGHV\d+-[0-9A-Za-z\-]+)", h_raw)
                if m:
                    g = m.group(1)
                    cov_counts["IGHV"][g] += 1
                    if is_neut:
                        cov_neut_counts["IGHV"][g] += 1
            if "human" in l_raw.lower() or ("human" in origin and "mouse" not in l_raw.lower()):
                m = re.search(r"(IG[KL]V\d+-[0-9A-Za-z\-]+)", l_raw)
                if m:
                    g = m.group(1)
                    cov_counts[g[:4]][g] += 1
                    if is_neut:
                        cov_neut_counts[g[:4]][g] += 1

    # 2. VDJdb human antigen-specific TCR counts (unique CDR3 + Epitope clonotypes per V gene)
    vdj_path = os.path.join(ROOT, "04_structure_repertoire", "vdjdb_2026_06_03", "vdjdb-2026-06-03", "vdjdb.txt")
    vdj_clones = defaultdict(set)
    vdj_viral_clones = defaultdict(set)
    viral_kw = ("sars", "influenza", "ebv", "cmv", "hiv", "kov", "yf", "hcv", "hbv", "dengue", "virus")
    with open(vdj_path, "r", encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if (r.get("species") or "").strip() != "HomoSapiens":
                continue
            gene_type = (r.get("gene") or "").strip()  # TRA or TRB
            loc = "TRAV" if gene_type == "TRA" else ("TRBV" if gene_type == "TRB" else None)
            if not loc:
                continue
            v_seg = (r.get("v.segm") or "").split("*")[0].strip()
            if not v_seg.startswith(loc[:4]):
                continue
            cdr3 = (r.get("cdr3") or "").strip()
            epi = (r.get("antigen.epitope") or "").strip()
            sp_ag = (r.get("antigen.species") or "").lower()
            vdj_clones[(loc, v_seg)].add((cdr3, epi))
            if any(k in sp_ag for k in viral_kw):
                vdj_viral_clones[(loc, v_seg)].add((cdr3, epi))

    # 3. `vane` FACS-sorted Human Naive vs Memory & MG vs HD BCR usage
    vane = pd.read_csv(os.path.join(ROOT, "04_structure_repertoire", "vane_bcr_v_usage_summary.tsv"), sep="\t")
    vane["locus_v"] = vane["locus"] + "V"
    hd_n = vane[(vane["cohort"] == "HD") & (vane["subset"] == "N")].groupby(["locus_v", "v_gene"])["freq_seq"].mean().to_dict()
    hd_m = vane[(vane["cohort"] == "HD") & (vane["subset"] == "M")].groupby(["locus_v", "v_gene"])["freq_seq"].mean().to_dict()
    hd_m_shm = vane[(vane["cohort"] == "HD") & (vane["subset"] == "M")].groupby(["locus_v", "v_gene"])["mean_shm_freq"].mean().to_dict()
    mg_m = vane[(vane["cohort"] != "HD") & (vane["subset"] == "M")].groupby(["locus_v", "v_gene"])["freq_seq"].mean().to_dict()

    # Build integrated gene-level table linking Primate Evolution (Tier 2) <-> Human Public Repertoires
    tot_cov = {loc: max(sum(cov_counts[loc].values()), 1) for loc in ["IGHV", "IGKV", "IGLV"]}
    tot_vdj = {loc: max(sum(len(s) for (l, _), s in vdj_clones.items() if l == loc), 1) for loc in ["TRAV", "TRBV"]}

    gene_rows = []
    for _, r in t2.iterrows():
        loc = r["locus"]
        g = r["human_gene"]
        pgen = float(olga_gene.get((loc, g), np.nan))
        naive_f = float(hd_n.get((loc, g), np.nan))
        mem_f = float(hd_m.get((loc, g), np.nan))
        mem_shm = float(hd_m_shm.get((loc, g), np.nan))
        mg_mem_f = float(mg_m.get((loc, g), np.nan))

        if loc in ("IGHV", "IGKV", "IGLV"):
            pub_n = cov_counts[loc].get(g, 0)
            pub_freq = pub_n / tot_cov[loc]
            neut_n = cov_neut_counts[loc].get(g, 0)
        else:
            pub_n = len(vdj_clones.get((loc, g), set()))
            pub_freq = pub_n / tot_vdj[loc]
            neut_n = len(vdj_viral_clones.get((loc, g), set()))

        baseline = naive_f if (np.isfinite(naive_f) and naive_f > 1e-5) else pgen
        log2_pub_over_pgen = math.log2((pub_freq + 1e-4) / (pgen + 1e-4)) if np.isfinite(pgen) and pgen > 0 else np.nan
        log2_pub_over_naive = math.log2((pub_freq + 1e-4) / (naive_f + 1e-4)) if np.isfinite(naive_f) and naive_f > 0 else np.nan
        log2_mem_over_naive = math.log2((mem_f + 1e-4) / (naive_f + 1e-4)) if (np.isfinite(mem_f) and np.isfinite(naive_f)) else np.nan
        log2_mg_over_hd_mem = math.log2((mg_mem_f + 1e-4) / (mem_f + 1e-4)) if (np.isfinite(mg_mem_f) and np.isfinite(mem_f)) else np.nan

        # List exact positively selected CDR1/CDR2/HV4 IMGT positions in this orthogroup
        sub_s = t2_sites[(t2_sites["orthogroup"] == r["orthogroup"]) & (t2_sites["locus"] == loc)]
        pos_sites = sub_s[(sub_s["meme_p"] < 0.05) | (sub_s["fubar_prob_pos"] >= 0.90)]["imgt_pos"].tolist()

        gene_rows.append({
            **r.to_dict(),
            "olga_pgen_v": pgen,
            "vane_hd_naive_freq": naive_f,
            "vane_hd_memory_freq": mem_f,
            "vane_hd_memory_shm": mem_shm,
            "vane_mg_memory_freq": mg_mem_f,
            "public_db_count": pub_n,
            "public_db_viral_or_neut_count": neut_n,
            "public_db_freq": pub_freq,
            "log2_public_over_olga_pgen": log2_pub_over_pgen,
            "log2_public_over_hd_naive": log2_pub_over_naive,
            "log2_hd_memory_over_naive": log2_mem_over_naive,
            "log2_mg_over_hd_memory": log2_mg_over_hd_mem,
            "pos_selected_imgt_positions": ",".join(map(str, pos_sites)),
        })

    gdf = pd.DataFrame(gene_rows)
    gdf.to_csv(os.path.join(OUT_DIR, "k3_gene_level_evolution_repertoire_integrated.tsv"), sep="\t", index=False)

    # Statistical tests for K3 across loci
    k3_tests = []
    # Test 1: Does BUSTED[S]+MH positive selection (q < 0.05 or p < 0.05) or CDR12 selection associate with Public Recruitment / SHM / Usage?
    for loc_filter, name in [("ALL", "All_5_Loci"), ("IGHV", "IGHV"), ("TRBV", "TRBV"), ("TRAV", "TRAV"), ("IGKV", "IGKV"), ("IGLV", "IGLV")]:
        sub = gdf if loc_filter == "ALL" else gdf[gdf["locus"] == loc_filter]
        sub_v = sub.dropna(subset=["olga_pgen_v"]).copy()
        if len(sub_v) < 8:
            continue
        sel_mask = (sub_v["busted_mh_p"].astype(float) < 0.05) | (sub_v["n_cdr12_meme_p05"].astype(int) >= 1) | (sub_v["n_cdr12_fubar_pp90"].astype(int) >= 1)
        pos_g = sub_v[sel_mask]
        neg_g = sub_v[~sel_mask]

        # Compare public recruitment count & enrichment controlling for P_gen(V)
        r_pub_cdr, p_pub_cdr = sp_stats.spearmanr(sub_v["cdr12_vs_fr_omega_ratio"].astype(float), sub_v["public_db_freq"])
        pr_pub, pp_pub = partial_spearman(
            sub_v["cdr12_vs_fr_omega_ratio"].astype(float).to_numpy(),
            sub_v["public_db_freq"].to_numpy(),
            sub_v["olga_pgen_v"].to_numpy(),
        )
        mwu_enr_p = (
            float(sp_stats.mannwhitneyu(pos_g["log2_public_over_olga_pgen"].dropna(), neg_g["log2_public_over_olga_pgen"].dropna(), alternative="greater")[1])
            if len(pos_g["log2_public_over_olga_pgen"].dropna()) >= 3 and len(neg_g["log2_public_over_olga_pgen"].dropna()) >= 3
            else np.nan
        )
        # Check germline readiness (lower Memory SHM needed when germline CDR1/2 already shaped by positive selection!)
        shm_rho, shm_p = (np.nan, np.nan)
        if sub_v["vane_hd_memory_shm"].notna().sum() >= 8:
            sub_s = sub_v.dropna(subset=["vane_hd_memory_shm"])
            shm_rho, shm_p = sp_stats.spearmanr(sub_s["mean_cdr12_omega"].astype(float), sub_s["vane_hd_memory_shm"])

        k3_tests.append({
            "subset": name,
            "n_genes_with_pgen": len(sub_v),
            "n_genes_positively_selected": int(sel_mask.sum()),
            "spearman_rho_cdr_fr_ratio_vs_public_freq": round(float(r_pub_cdr), 4),
            "spearman_p": float(p_pub_cdr),
            "partial_rho_controlling_olga_pgen": round(pr_pub, 4),
            "partial_p_controlling_olga_pgen": pp_pub,
            "mwu_p_pos_vs_neg_log2_public_over_pgen": mwu_enr_p,
            "spearman_rho_cdr12_omega_vs_memory_shm": round(float(shm_rho), 4) if np.isfinite(shm_rho) else np.nan,
            "spearman_p_memory_shm": float(shm_p) if np.isfinite(shm_p) else np.nan,
        })

    k3_df = pd.DataFrame(k3_tests)
    k3_df.to_csv(os.path.join(OUT_DIR, "k3_public_repertoire_tests.tsv"), sep="\t", index=False)
    return k3_df, gdf


def main():
    k1_df, t1, t2, t2_sites = evaluate_k1()
    k2_df = evaluate_k2(t1)
    k3_df, gdf = evaluate_k3(t2, t2_sites)

    print("=== K1 SUMMARY (Germline CDR1+CDR2 vs Core FR Positive Selection) ===")
    print(k1_df.to_string(index=False))
    print("\n=== K2 SUMMARY (Primate Selection vs 3D Crystal Structure Contacts) ===")
    print(k2_df.to_string(index=False))
    print("\n=== K3 SUMMARY (Human Public Antiviral & FACS Repertoire Integration) ===")
    print(k3_df.to_string(index=False))

    print("\n=== ARCHETYPE PUBLIC ANTIVIRAL & AUTOIMMUNE GENES ===")
    archetypes = ["IGHV1-69", "IGHV3-53", "IGHV3-66", "IGHV3-30", "IGHV3-23", "IGHV1-2", "IGHV4-34", "IGHV5-51", "TRBV19", "TRBV6-5", "TRBV20-1", "TRBV28", "TRBV7-2", "TRAV12-2", "TRAV27"]
    sub_a = gdf[gdf["human_gene"].isin(archetypes)][[
        "locus", "human_gene", "is_strict_1to1", "n_species", "busted_mh_p", "busted_mh_q_global",
        "mean_cdr12_omega", "mean_core_fr_omega", "cdr12_vs_fr_omega_ratio",
        "public_db_count", "log2_public_over_olga_pgen", "vane_hd_memory_shm", "pos_selected_imgt_positions"
    ]]
    print(sub_a.to_string(index=False))
    print("\n[STAGE0_KILL_EVAL_COMPLETE]", flush=True)


if __name__ == "__main__":
    main()
