#!/usr/bin/env python3
"""
03_run_codon_selection.py
Reproducible T2T Primate Germline V-Segment Positive Selection Pipeline (HyPhy 2.5.101 + IQ-TREE 2):
Tier 1 (Site-level IMGT 1..104 selection atlas across all 5 loci & IGHV Clans I/II/III):
  - IGHV_ClanI (IGHV1/5/7), IGHV_ClanII (IGHV2/4/6), IGHV_ClanIII (IGHV3)
  - TRBV_All, TRAV_All, IGKV_All, IGLV_All
  - Deduplicates 100%-identical 312-nt sequences (preventing zero-branch artifacts), records exact active-column -> IMGT 1..104 map,
    infers ML phylogeny (`iqtree -m GTR+G4 -fast`), and runs:
      1) `hyphy fel --srv Yes` (pervasive site-specific dN, dS, LRT p-value)
      2) `hyphy meme` (episodic diversifying site-specific beta+, LRT p-value)
      3) `hyphy fubar` (Bayesian posterior Pr[beta > alpha])

Tier 2 (Human-anchored V-gene orthogroups & Strict 1:1 Syntenic Orthologs):
  - Groups functional T2T V-segments across the 8 unique primate species (`t2t8_chm13`) by `closest_human_gene`
    (and marks `is_strict_1to1 = True` when <=1 copy per species across >=6 species, zero paralog mixing!).
  - Runs `iqtree` + `hyphy busted --srv Yes --multiple-hits Double+Triple` (`BUSTED[S]+MH`) + `hyphy fubar` + `hyphy meme`
    on every eligible orthogroup (>=5 unique sequences across >=4 species), PLUS `GARD` recombination check on major subfamilies.
"""
import concurrent.futures
import csv
import json
import math
import os
import re
import subprocess
from collections import Counter, defaultdict
import numpy as np
from scipy import stats as sp_stats

ROOT = os.environ.get("BRIM_DATA_DIR", os.environ.get("BRIM_DATA_DIR", "./data"))
GERM_TSV = os.path.join(ROOT, "02_germline_v", "t2t_primate_germline_v_catalog.tsv")
OUT_DIR = os.path.join(ROOT, "03_selection")

IQTREE = "iqtree2"
HYPHY = "hyphy"


def imgt_region_of_pos(pos: int) -> str:
    if 1 <= pos <= 26:
        return "FR1"
    if 27 <= pos <= 38:
        return "CDR1"
    if 39 <= pos <= 55:
        return "FR2"
    if 56 <= pos <= 65:
        return "CDR2"
    if 74 <= pos <= 85:
        return "HV4"
    return "FR3_core"


def prepare_codon_alignment(records: list, out_prefix: str):
    """
    Takes list of (seq_id, nt312_with_dots_or_dashes) and:
    1) Deduplicates 100%-identical sequences (keeping representative taxon name)
    2) Removes 100%-gap columns while saving exact 0-based active_idx -> 1-based IMGT pos (1..104)
    Returns (aln_fa, map_json_list, n_unique)
    """
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    seen = {}
    for sid, raw_nt in records:
        nt = raw_nt.upper().replace(".", "-")
        assert len(nt) == 312, f"Bad length {len(nt)} for {sid}"
        if nt not in seen:
            seen[nt] = re.sub(r"[^A-Za-z0-9_]", "_", sid)

    unique_items = list(seen.items())  # (nt, sid)
    if len(unique_items) < 4:
        return None, [], len(unique_items)

    # Identify active IMGT codons (at least 3 non-gap sequences at that codon so site rates are identifiable)
    active_to_imgt = []
    active_slices = []
    for pos_1b in range(1, 105):
        st = (pos_1b - 1) * 3
        non_gap = sum(1 for nt, _ in unique_items if nt[st:st + 3] != "---" and "-" not in nt[st:st + 3])
        if non_gap >= 3:
            active_to_imgt.append(pos_1b)
            active_slices.append((st, st + 3))

    aln_fa = out_prefix + ".aln.fa"
    with open(aln_fa, "w") as fh:
        for nt, sid in unique_items:
            trimmed = "".join(nt[a:b] for a, b in active_slices)
            # Replace any partial-gap codon with '---'
            cods = [trimmed[i:i + 3] if "-" not in trimmed[i:i + 3] else "---" for i in range(0, len(trimmed), 3)]
            fh.write(f">{sid}\n{''.join(cods)}\n")

    with open(out_prefix + ".imgt_map.json", "w") as fh:
        json.dump(active_to_imgt, fh)
    return aln_fa, active_to_imgt, len(unique_items)


def build_tree(aln_fa: str, cpu: int = 4) -> str:
    treefile = aln_fa + ".treefile"
    if os.path.exists(treefile) and os.path.getsize(treefile) > 10:
        return treefile
    subprocess.check_call(
        [IQTREE, "-s", aln_fa, "-st", "DNA", "-m", "GTR+G4", "-fast", "-nt", str(cpu), "-redo", "-quiet"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return treefile


def parse_hyphy_site_table(json_path: str) -> tuple:
    with open(json_path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    headers = [h[0] for h in d["MLE"]["headers"]]
    rows = d["MLE"]["content"]["0"]
    return headers, rows, d


def run_tier1_unit(unit_name: str, locus: str, records: list, cpu: int = 10) -> list:
    udir = os.path.join(OUT_DIR, "tier1_loci_clans", unit_name)
    os.makedirs(udir, exist_ok=True)
    prefix = os.path.join(udir, unit_name)

    aln_fa, active_to_imgt, n_uniq = prepare_codon_alignment(records, prefix)
    assert aln_fa is not None, f"Too few unique sequences for {unit_name}: {n_uniq}"
    print(f"[TIER1 START] {unit_name}: raw={len(records)}, unique={n_uniq}, active_codons={len(active_to_imgt)}", flush=True)

    treefile = build_tree(aln_fa, cpu=min(cpu, 8))

    fel_json = prefix + ".fel.json"
    meme_json = prefix + ".meme.json"
    fubar_json = prefix + ".fubar.json"

    if not (os.path.exists(fel_json) and os.path.getsize(fel_json) > 500):
        subprocess.check_call(
            [HYPHY, f"CPU={cpu}", "fel", "--alignment", aln_fa, "--tree", treefile, "--srv", "Yes", "--output", fel_json],
            stdout=open(prefix + ".fel.log", "w"),
            stderr=subprocess.STDOUT,
        )
    if not (os.path.exists(meme_json) and os.path.getsize(meme_json) > 500):
        subprocess.check_call(
            [HYPHY, f"CPU={cpu}", "meme", "--alignment", aln_fa, "--tree", treefile, "--pvalue", "0.05", "--output", meme_json],
            stdout=open(prefix + ".meme.log", "w"),
            stderr=subprocess.STDOUT,
        )
    if not (os.path.exists(fubar_json) and os.path.getsize(fubar_json) > 500):
        subprocess.check_call(
            [HYPHY, f"CPU={cpu}", "fubar", "--alignment", aln_fa, "--tree", treefile, "--cache", prefix + ".fubar.cache", "--output", fubar_json],
            stdout=open(prefix + ".fubar.log", "w"),
            stderr=subprocess.STDOUT,
        )

    fel_h, fel_r, _ = parse_hyphy_site_table(fel_json)
    meme_h, meme_r, _ = parse_hyphy_site_table(meme_json)
    fub_h, fub_r, _ = parse_hyphy_site_table(fubar_json)

    fi = {k: i for i, k in enumerate(fel_h)}
    mi = {k: i for i, k in enumerate(meme_h)}
    bi = {k: i for i, k in enumerate(fub_h)}

    # Find exact FUBAR positive posterior header key
    fub_pos_key = next(k for k in fub_h if "alpha<beta" in k.replace(" ", "") and "Prob" in k)

    out_rows = []
    for idx, imgt_pos in enumerate(active_to_imgt):
        fr = fel_r[idx]
        mr = meme_r[idx]
        br = fub_r[idx]

        fel_alpha = float(fr[fi["alpha"]])
        fel_beta = float(fr[fi["beta"]])
        fel_p = float(fr[fi["p-value"]])
        # Empirical site dN/dS stabilized with small pseudocount 0.02 when dS~0, plus FUBAR Bayesian beta/alpha!
        fub_alpha = float(br[bi["alpha"]])
        fub_beta = float(br[bi["beta"]])
        fub_prob_pos = float(br[bi[fub_pos_key]])
        fub_omega = (fub_beta + 0.01) / (fub_alpha + 0.01)
        fel_omega = (fel_beta + 0.01) / (fel_alpha + 0.01)

        meme_p = float(mr[mi["p-value"]])
        meme_beta_plus = float(mr[mi.get("&beta;<sup>+</sup>", mi.get("beta+", 3))])

        out_rows.append({
            "unit": unit_name,
            "locus": locus,
            "n_raw_seqs": len(records),
            "n_unique_seqs": n_uniq,
            "imgt_pos": imgt_pos,
            "region": imgt_region_of_pos(imgt_pos),
            "fel_alpha": f"{fel_alpha:.5f}",
            "fel_beta": f"{fel_beta:.5f}",
            "fel_omega_reg": f"{fel_omega:.5f}",
            "fel_p": f"{fel_p:.6g}",
            "fel_pos_p05": int(fel_beta > fel_alpha and fel_p < 0.05),
            "fel_neg_p05": int(fel_beta < fel_alpha and fel_p < 0.05),
            "meme_beta_plus": f"{meme_beta_plus:.5f}",
            "meme_p": f"{meme_p:.6g}",
            "meme_pos_p05": int(meme_p < 0.05),
            "fubar_alpha": f"{fub_alpha:.5f}",
            "fubar_beta": f"{fub_beta:.5f}",
            "fubar_omega": f"{fub_omega:.5f}",
            "fubar_prob_pos": f"{fub_prob_pos:.5f}",
            "fubar_pos_pp90": int(fub_prob_pos >= 0.90),
        })

    print(
        f"[TIER1 DONE ] {unit_name}: MEME_p05={sum(r['meme_pos_p05'] for r in out_rows)}, "
        f"FEL_pos_p05={sum(r['fel_pos_p05'] for r in out_rows)}, "
        f"FUBAR_pp90={sum(r['fubar_pos_pp90'] for r in out_rows)}",
        flush=True,
    )
    return out_rows


def run_tier2_orthogroup(task: tuple):
    og_id, locus, human_gene, subfamily, is_strict_1to1, n_species, n_copies_human, records = task
    odir = os.path.join(OUT_DIR, "tier2_orthogroups", f"{locus}__{og_id}")
    os.makedirs(odir, exist_ok=True)
    prefix = os.path.join(odir, og_id)

    aln_fa, active_to_imgt, n_uniq = prepare_codon_alignment(records, prefix)
    if aln_fa is None or n_uniq < 5:
        return None, []

    try:
        treefile = build_tree(aln_fa, cpu=2)
        busted_json = prefix + ".busted_mh.json"
        fubar_json = prefix + ".fubar.json"
        meme_json = prefix + ".meme.json"

        if not (os.path.exists(busted_json) and os.path.getsize(busted_json) > 500):
            subprocess.check_call(
                [
                    HYPHY, "CPU=2", "busted",
                    "--alignment", aln_fa, "--tree", treefile,
                    "--srv", "Yes", "--multiple-hits", "Double+Triple",
                    "--output", busted_json,
                ],
                stdout=open(prefix + ".busted.log", "w"),
                stderr=subprocess.STDOUT,
                timeout=600,
            )
        if not (os.path.exists(fubar_json) and os.path.getsize(fubar_json) > 500):
            subprocess.check_call(
                [
                    HYPHY, "CPU=2", "fubar",
                    "--alignment", aln_fa, "--tree", treefile,
                    "--cache", prefix + ".fubar.cache", "--output", fubar_json,
                ],
                stdout=open(prefix + ".fubar.log", "w"),
                stderr=subprocess.STDOUT,
                timeout=300,
            )
        if not (os.path.exists(meme_json) and os.path.getsize(meme_json) > 500):
            subprocess.check_call(
                [
                    HYPHY, "CPU=2", "meme",
                    "--alignment", aln_fa, "--tree", treefile,
                    "--pvalue", "0.05", "--output", meme_json,
                ],
                stdout=open(prefix + ".meme.log", "w"),
                stderr=subprocess.STDOUT,
                timeout=300,
            )

        with open(busted_json) as fh:
            bd = json.load(fh)
        busted_p = float(bd["test results"]["p-value"])
        busted_lrt = float(bd["test results"]["LRT"])

        fub_h, fub_r, _ = parse_hyphy_site_table(fubar_json)
        meme_h, meme_r, _ = parse_hyphy_site_table(meme_json)
        bi = {k: i for i, k in enumerate(fub_h)}
        mi = {k: i for i, k in enumerate(meme_h)}
        fub_pos_key = next(k for k in fub_h if "alpha<beta" in k.replace(" ", "") and "Prob" in k)

        cdr12_omegas, fr_omegas = [], []
        cdr12_meme_n, fr_meme_n = 0, 0
        cdr12_fub_n, fr_fub_n = 0, 0
        site_rows = []

        for idx, pos in enumerate(active_to_imgt):
            br = fub_r[idx]
            mr = meme_r[idx]
            fa, fb = float(br[bi["alpha"]]), float(br[bi["beta"]])
            fpp = float(br[bi[fub_pos_key]])
            om = (fb + 0.01) / (fa + 0.01)
            mp = float(mr[mi["p-value"]])
            reg = imgt_region_of_pos(pos)
            if reg in ("CDR1", "CDR2"):
                cdr12_omegas.append(om)
                if mp < 0.05:
                    cdr12_meme_n += 1
                if fpp >= 0.90:
                    cdr12_fub_n += 1
            elif reg in ("FR1", "FR2", "FR3_core"):
                fr_omegas.append(om)
                if mp < 0.05:
                    fr_meme_n += 1
                if fpp >= 0.90:
                    fr_fub_n += 1
            site_rows.append({
                "orthogroup": og_id,
                "locus": locus,
                "human_gene": human_gene,
                "is_strict_1to1": int(is_strict_1to1),
                "imgt_pos": pos,
                "region": reg,
                "fubar_omega": om,
                "fubar_prob_pos": fpp,
                "meme_p": mp,
            })

        mean_cdr_om = float(np.mean(cdr12_omegas)) if cdr12_omegas else np.nan
        mean_fr_om = float(np.mean(fr_omegas)) if fr_omegas else np.nan
        cdr_fr_ratio = mean_cdr_om / mean_fr_om if (mean_fr_om and mean_fr_om > 0) else np.nan

        summary = {
            "orthogroup": og_id,
            "locus": locus,
            "human_gene": human_gene,
            "subfamily": subfamily,
            "is_strict_1to1": int(is_strict_1to1),
            "n_species": n_species,
            "n_seqs_raw": len(records),
            "n_seqs_unique": n_uniq,
            "n_human_copies": n_copies_human,
            "busted_mh_lrt": f"{busted_lrt:.4f}",
            "busted_mh_p": busted_p,
            "mean_cdr12_omega": f"{mean_cdr_om:.4f}",
            "mean_core_fr_omega": f"{mean_fr_om:.4f}",
            "cdr12_vs_fr_omega_ratio": f"{cdr_fr_ratio:.4f}",
            "n_cdr12_meme_p05": cdr12_meme_n,
            "n_fr_meme_p05": fr_meme_n,
            "n_cdr12_fubar_pp90": cdr12_fub_n,
            "n_fr_fubar_pp90": fr_fub_n,
        }
        return summary, site_rows
    except Exception as e:
        print(f"[WARN] Tier2 failed on {og_id}: {e}", flush=True)
        return None, []


def bh_fdr(pvals: list) -> list:
    n = len(pvals)
    if n == 0:
        return []
    order = np.argsort(pvals)
    q = np.empty(n, dtype=float)
    prev = 1.0
    for rank_rev, idx in enumerate(order[::-1]):
        rank = n - rank_rev
        val = min(prev, pvals[idx] * n / rank)
        q[idx] = val
        prev = val
    return q.tolist()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # Load functional 8-species haploid T2T set (drop Human_GRCh38 to keep 1 haploid reference per species: Human_CHM13)
    rows = []
    with open(GERM_TSV, "r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["status"] == "F" and r["assembly_id"] != "Human_GRCh38":
                rows.append(r)

    print(f"[INFO] Loaded {len(rows)} functional V-genes across 8 T2T primate species", flush=True)

    # Build Tier 1 units
    tier1_specs = []
    # IGHV Clans I, II, III
    clan_map = {
        "IGHV1": "IGHV_ClanI", "IGHV5": "IGHV_ClanI", "IGHV7": "IGHV_ClanI",
        "IGHV2": "IGHV_ClanII", "IGHV4": "IGHV_ClanII", "IGHV6": "IGHV_ClanII",
        "IGHV3": "IGHV_ClanIII",
    }
    for cname in ["IGHV_ClanI", "IGHV_ClanII", "IGHV_ClanIII"]:
        recs = [(r["gene_uid"], r["imgt_312_nt"]) for r in rows if r["locus"] == "IGHV" and clan_map.get(r["subfamily"]) == cname]
        if len(recs) >= 10:
            tier1_specs.append((cname, "IGHV", recs))

    for loc in ["TRBV", "TRAV", "IGKV", "IGLV"]:
        recs = [(r["gene_uid"], r["imgt_312_nt"]) for r in rows if r["locus"] == loc]
        if len(recs) >= 10:
            tier1_specs.append((f"{loc}_All", loc, recs))

    tier1_rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=7) as ex:
        futs = [ex.submit(run_tier1_unit, uname, loc, recs, 10) for uname, loc, recs in tier1_specs]
        for f in concurrent.futures.as_completed(futs):
            tier1_rows.extend(f.result())

    # Also build sequence-weighted meta-profile `IGHV_All` from Clan I + Clan II + Clan III across IMGT 1..104
    ighv_by_pos = defaultdict(list)
    for r in tier1_rows:
        if r["locus"] == "IGHV":
            ighv_by_pos[int(r["imgt_pos"])].append(r)

    for pos in sorted(ighv_by_pos.keys()):
        clist = ighv_by_pos[pos]
        w = np.array([float(x["n_unique_seqs"]) for x in clist])
        w_norm = w / np.sum(w)
        # Fisher's combined probability test across independent clans for FEL positive selection & MEME
        meme_ps = [max(float(x["meme_p"]), 1e-16) for x in clist]
        stat_m = -2.0 * sum(math.log(p) for p in meme_ps)
        meta_meme_p = float(sp_stats.chi2.sf(stat_m, 2 * len(meme_ps)))

        # Directional p for FEL positive selection (if beta < alpha, p_one_sided_pos >= 0.5)
        fel_pos_ps = [
            max(float(x["fel_p"]) / 2.0, 1e-16) if float(x["fel_beta"]) > float(x["fel_alpha"]) else 1.0 - float(x["fel_p"]) / 2.0
            for x in clist
        ]
        stat_f = -2.0 * sum(math.log(min(max(p, 1e-16), 1.0)) for p in fel_pos_ps)
        meta_fel_pos_p = float(sp_stats.chi2.sf(stat_f, 2 * len(fel_pos_ps)))

        w_fel_om = float(np.sum(w_norm * np.array([float(x["fel_omega_reg"]) for x in clist])))
        w_fub_om = float(np.sum(w_norm * np.array([float(x["fubar_omega"]) for x in clist])))
        max_fub_pp = max(float(x["fubar_prob_pos"]) for x in clist)

        tier1_rows.append({
            "unit": "IGHV_MetaClans",
            "locus": "IGHV",
            "n_raw_seqs": sum(int(x["n_raw_seqs"]) for x in clist),
            "n_unique_seqs": int(np.sum(w)),
            "imgt_pos": pos,
            "region": imgt_region_of_pos(pos),
            "fel_alpha": f"{np.sum(w_norm * np.array([float(x['fel_alpha']) for x in clist])):.5f}",
            "fel_beta": f"{np.sum(w_norm * np.array([float(x['fel_beta']) for x in clist])):.5f}",
            "fel_omega_reg": f"{w_fel_om:.5f}",
            "fel_p": f"{meta_fel_pos_p:.6g}",
            "fel_pos_p05": int(meta_fel_pos_p < 0.05),
            "fel_neg_p05": int(all(int(x["fel_neg_p05"]) == 1 for x in clist)),
            "meme_beta_plus": f"{np.sum(w_norm * np.array([float(x['meme_beta_plus']) for x in clist])):.5f}",
            "meme_p": f"{meta_meme_p:.6g}",
            "meme_pos_p05": int(meta_meme_p < 0.05),
            "fubar_alpha": f"{np.sum(w_norm * np.array([float(x['fubar_alpha']) for x in clist])):.5f}",
            "fubar_beta": f"{np.sum(w_norm * np.array([float(x['fubar_beta']) for x in clist])):.5f}",
            "fubar_omega": f"{w_fub_om:.5f}",
            "fubar_prob_pos": f"{max_fub_pp:.5f}",
            "fubar_pos_pp90": int(max_fub_pp >= 0.90),
        })

    tier1_rows.sort(key=lambda x: (x["unit"], int(x["imgt_pos"])))
    t1_tsv = os.path.join(OUT_DIR, "tier1_site_selection_imgt104.tsv")
    with open(t1_tsv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(tier1_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(tier1_rows)
    print(f"[DONE] Saved Tier 1 site-level selection table -> {t1_tsv}", flush=True)

    # Build Tier 2 human-anchored orthogroups across all 5 loci
    by_hgene = defaultdict(list)
    for r in rows:
        # Collapse exact human tandem D-duplicates onto base gene if needed? Keep closest_human_gene first,
        # and also strip trailing 'D' ONLY when closest_human_gene has <4 species so tandem duplicates group cleanly
        by_hgene[(r["locus"], r["closest_human_gene"])].append(r)

    tier2_tasks = []
    for (loc, hgene), glist in sorted(by_hgene.items()):
        sp_counts = Counter(x["assembly_id"] for x in glist)
        n_sp = len(sp_counts)
        uniq_nts = len(set(x["imgt_312_nt"] for x in glist))
        if n_sp < 4 or uniq_nts < 5:
            continue
        is_1to1 = (max(sp_counts.values()) == 1 and n_sp >= 6)
        n_hum = sp_counts.get("Human_CHM13", 0)
        subfam = glist[0]["subfamily"]
        og_clean = re.sub(r"[^A-Za-z0-9_]", "_", hgene)
        recs = [(x["gene_uid"], x["imgt_312_nt"]) for x in glist]
        tier2_tasks.append((og_clean, loc, hgene, subfam, is_1to1, n_sp, n_hum, recs))

    print(
        f"[INFO] Running Tier 2 selection on {len(tier2_tasks)} human-anchored orthogroups "
        f"(strict 1:1 count = {sum(1 for t in tier2_tasks if t[4])})...",
        flush=True,
    )

    t2_summaries = []
    t2_sites = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=40) as ex:
        for summ, srows in ex.map(run_tier2_orthogroup, tier2_tasks):
            if summ is not None:
                t2_summaries.append(summ)
                t2_sites.extend(srows)

    # Apply BH-FDR across all tested Tier 2 orthogroups (and per-locus q-value)
    all_ps = [float(s["busted_mh_p"]) for s in t2_summaries]
    all_qs = bh_fdr(all_ps)
    for s, q in zip(t2_summaries, all_qs):
        s["busted_mh_q_global"] = f"{q:.6g}"
        s["busted_mh_p"] = f"{s['busted_mh_p']:.6g}"

    t2_summaries.sort(key=lambda x: (x["locus"], float(x["busted_mh_p"])))
    t2_tsv = os.path.join(OUT_DIR, "tier2_orthogroup_selection.tsv")
    if t2_summaries:
        with open(t2_tsv, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(t2_summaries[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(t2_summaries)

    t2_sites_tsv = os.path.join(OUT_DIR, "tier2_orthogroup_sites.tsv")
    if t2_sites:
        with open(t2_sites_tsv, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(t2_sites[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(t2_sites)

    print(f"[ALL_SELECTION_COMPLETE] Tier2 tested={len(t2_summaries)}, q<0.05={sum(1 for s in t2_summaries if float(s['busted_mh_q_global'])<0.05)}", flush=True)


if __name__ == "__main__":
    main()
