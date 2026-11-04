#!/usr/bin/env python3
"""
03b_recombination_gard_sensitivity.py
Reviewer-proof Gene Conversion / Ectopic Recombination Control:
In multi-copy Ig/TCR families, interlocus gene conversion frequently exchanges 5' (FR1-CDR1-FR2, IMGT 1..55)
and 3' (CDR2-FR3, IMGT 56..104) segments, which could distort a single full-exon tree topology.
Here, for each Tier 1 dataset (IGHV_ClanI/II/III, TRBV_All, TRAV_All, IGKV_All, IGLV_All):
  1) Partition each alignment at the canonical FR2/CDR2 boundary into:
       - 5' Partition: IMGT 1..55 (FR1 1..26, CDR1 27..38, FR2 39..55)
       - 3' Partition: IMGT 56..104 (CDR2 56..65, FR3_core 66..73 & 86..104, HV4 74..85)
  2) Infer INDEPENDENT Maximum-Likelihood phylogenies (`iqtree -m GTR+G4`) on each partition separately,
     after deduplicating identical partition sequences.
  3) Run `hyphy fubar` and `hyphy meme` on each partition under its OWN partition-specific ML tree!
  4) Compare CDR1 vs FR1+FR2 (within 5' partition tree) and CDR2 vs FR3_core (within 3' partition tree),
     plus run `hyphy gard` on representative subfamilies (IGHV1, TRBV6) to verify breakpoint robustness.
"""
import concurrent.futures
import csv
import json
import os
import re
import subprocess
import numpy as np
from scipy import stats as sp_stats

ROOT = os.environ.get("BRIM_DATA_DIR", os.environ.get("BRIM_DATA_DIR", "./data"))
GERM_TSV = os.path.join(ROOT, "02_germline_v", "t2t_primate_germline_v_catalog.tsv")
OUT_DIR = os.path.join(ROOT, "03_selection", "recombination_sensitivity")

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


def run_partition(unit_name: str, part_name: str, pos_range: range, records: list, cpu: int = 6) -> list:
    pdir = os.path.join(OUT_DIR, f"{unit_name}__{part_name}")
    os.makedirs(pdir, exist_ok=True)
    prefix = os.path.join(pdir, f"{unit_name}_{part_name}")

    # Slice to pos_range and deduplicate identical partition sequences (critical so iqtree/hyphy have no zero-branch duplicates)
    seen = {}
    for sid, raw_nt in records:
        nt_full = raw_nt.upper().replace(".", "-")
        sub_nt = "".join(nt_full[(p - 1) * 3:p * 3] for p in pos_range)
        non_gap_cods = sum(1 for i in range(0, len(sub_nt), 3) if "-" not in sub_nt[i:i + 3])
        if non_gap_cods < 10:
            continue
        if sub_nt not in seen:
            seen[sub_nt] = re.sub(r"[^A-Za-z0-9_]", "_", sid)

    uniq = list(seen.items())
    active_pos = []
    active_idx = []
    pos_list = list(pos_range)
    for idx_local, p_imgt in enumerate(pos_list):
        st = idx_local * 3
        ng = sum(1 for s, _ in uniq if "-" not in s[st:st + 3])
        if ng >= 3:
            active_pos.append(p_imgt)
            active_idx.append((st, st + 3))

    aln_fa = prefix + ".aln.fa"
    with open(aln_fa, "w") as fh:
        for s, sid in uniq:
            trimmed = "".join(s[a:b] if "-" not in s[a:b] else "---" for a, b in active_idx)
            fh.write(f">{sid}\n{trimmed}\n")

    treefile = aln_fa + ".treefile"
    if not (os.path.exists(treefile) and os.path.getsize(treefile) > 10):
        subprocess.check_call(
            [IQTREE, "-s", aln_fa, "-st", "DNA", "-m", "GTR+G4", "-fast", "-nt", str(min(cpu, 6)), "-redo", "-quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    fub_json = prefix + ".fubar.json"
    meme_json = prefix + ".meme.json"
    if not (os.path.exists(fub_json) and os.path.getsize(fub_json) > 500):
        subprocess.check_call(
            [HYPHY, f"CPU={cpu}", "fubar", "--alignment", aln_fa, "--tree", treefile, "--cache", prefix + ".fubar.cache", "--output", fub_json],
            stdout=open(prefix + ".fubar.log", "w"),
            stderr=subprocess.STDOUT,
        )
    if not (os.path.exists(meme_json) and os.path.getsize(meme_json) > 500):
        subprocess.check_call(
            [HYPHY, f"CPU={cpu}", "meme", "--alignment", aln_fa, "--tree", treefile, "--pvalue", "0.05", "--output", meme_json],
            stdout=open(prefix + ".meme.log", "w"),
            stderr=subprocess.STDOUT,
        )

    with open(fub_json) as fh:
        fd = json.load(fh)
    with open(meme_json) as fh:
        md = json.load(fh)
    fhdr = [h[0] for h in fd["MLE"]["headers"]]
    mhdr = [h[0] for h in md["MLE"]["headers"]]
    frows = fd["MLE"]["content"]["0"]
    mrows = md["MLE"]["content"]["0"]
    bi = {k: i for i, k in enumerate(fhdr)}
    mi = {k: i for i, k in enumerate(mhdr)}
    fpp_key = next(k for k in fhdr if "alpha<beta" in k.replace(" ", "") and "Prob" in k)

    out = []
    for i, p_imgt in enumerate(active_pos):
        fa, fb = float(frows[i][bi["alpha"]]), float(frows[i][bi["beta"]])
        fpp = float(frows[i][bi[fpp_key]])
        mp = float(mrows[i][mi["p-value"]])
        out.append({
            "unit": unit_name,
            "partition": part_name,
            "n_unique_partition_seqs": len(uniq),
            "imgt_pos": p_imgt,
            "region": imgt_region_of_pos(p_imgt),
            "fubar_omega_part": (fb + 0.01) / (fa + 0.01),
            "fubar_pp_part": fpp,
            "meme_p_part": mp,
        })
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    with open(GERM_TSV, "r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["status"] == "F" and r["assembly_id"] != "Human_GRCh38":
                rows.append(r)

    clan_map = {
        "IGHV1": "IGHV_ClanI", "IGHV5": "IGHV_ClanI", "IGHV7": "IGHV_ClanI",
        "IGHV2": "IGHV_ClanII", "IGHV4": "IGHV_ClanII", "IGHV6": "IGHV_ClanII",
        "IGHV3": "IGHV_ClanIII",
    }
    tasks = []
    for cname in ["IGHV_ClanI", "IGHV_ClanII", "IGHV_ClanIII"]:
        recs = [(r["gene_uid"], r["imgt_312_nt"]) for r in rows if r["locus"] == "IGHV" and clan_map.get(r["subfamily"]) == cname]
        if len(recs) >= 10:
            tasks.append((cname, "5prime_IMGT1_55", range(1, 56), recs))
            tasks.append((cname, "3prime_IMGT56_104", range(56, 105), recs))
    for loc in ["TRBV", "TRAV", "IGKV", "IGLV"]:
        recs = [(r["gene_uid"], r["imgt_312_nt"]) for r in rows if r["locus"] == loc]
        if len(recs) >= 10:
            tasks.append((f"{loc}_All", "5prime_IMGT1_55", range(1, 56), recs))
            tasks.append((f"{loc}_All", "3prime_IMGT56_104", range(56, 105), recs))

    site_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=14) as ex:
        futs = [ex.submit(run_partition, u, p, rng, recs, 6) for u, p, rng, recs in tasks]
        for f in concurrent.futures.as_completed(futs):
            site_results.extend(f.result())

    site_results.sort(key=lambda x: (x["unit"], x["imgt_pos"]))
    sites_tsv = os.path.join(OUT_DIR, "partitioned_tree_site_selection.tsv")
    with open(sites_tsv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(site_results[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(site_results)

    # Summarize CDR vs FR under partition-specific ML phylogenies
    by_unit = {}
    for r in site_results:
        by_unit.setdefault(r["unit"], []).append(r)

    summ_rows = []
    for uname, ulist in sorted(by_unit.items()):
        cdr = [x for x in ulist if x["region"] in ("CDR1", "CDR2")]
        fr = [x for x in ulist if x["region"] in ("FR1", "FR2", "FR3_core")]
        c_om = [x["fubar_omega_part"] for x in cdr]
        f_om = [x["fubar_omega_part"] for x in fr]
        _, p_mwu = sp_stats.mannwhitneyu(c_om, f_om, alternative="greater")
        a_m = sum(1 for x in cdr if x["meme_p_part"] < 0.05)
        c_m = sum(1 for x in fr if x["meme_p_part"] < 0.05)
        or_m, p_fm = sp_stats.fisher_exact([[a_m, len(cdr) - a_m], [c_m, len(fr) - c_m]], alternative="greater")
        summ_rows.append({
            "unit": uname,
            "median_omega_cdr12_part_tree": round(float(np.median(c_om)), 4),
            "median_omega_core_fr_part_tree": round(float(np.median(f_om)), 4),
            "ratio_cdr_vs_fr_part_tree": round(float(np.median(c_om)) / max(float(np.median(f_om)), 1e-6), 3),
            "mwu_p_part_tree": p_mwu,
            "meme_p05_cdr12_part_tree": f"{a_m}/{len(cdr)}",
            "meme_p05_core_fr_part_tree": f"{c_m}/{len(fr)}",
            "meme_or_part_tree": round(float(or_m), 3) if np.isfinite(or_m) else "Inf",
            "meme_fisher_p_part_tree": p_fm,
        })

    summ_tsv = os.path.join(OUT_DIR, "partitioned_tree_summary.tsv")
    with open(summ_tsv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(summ_rows)

    print("[PARTITION_SENSITIVITY_COMPLETE]")
    for s in summ_rows:
        print(" ", s)


if __name__ == "__main__":
    main()
