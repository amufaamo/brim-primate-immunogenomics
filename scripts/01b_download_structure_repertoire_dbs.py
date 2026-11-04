#!/usr/bin/env python3
"""
Reproducible downloader & baseline aggregator for Stage 0 / Full Plan A:
1. SAbDab2 complete summary CSV (~22,692 antibody structures)
2. CoV-AbDab CSV + ANARCI numberings JSON + PDBStructures.tar.gz (IMGT-numbered Ab-Ag complexes)
3. STCRDab complete summary TSV + TCR3d metadata TSVs + TCR_complexes.tar.gz (solved TCR-pMHC complexes)
4. VDJdb 2026-06-03 release zip (curated antigen-specific human TCR alpha/beta records)
5. OLGA default human models (IGH, IGK, IGL, TRB, TRA) -> parsed P_gen(V) table
6. Read-only aggregation of in-house FACS-sorted human Naive & Memory BCR repertoires (~/vane/germ_presto/)
"""
import csv
import glob
import io
import json
import os
import tarfile
import time
import urllib.request
import zipfile
from collections import defaultdict

ROOT = os.environ.get("BRIM_DATA_DIR", os.environ.get("BRIM_DATA_DIR", "./data"))
assert ROOT.endswith("analysis/t2t_germline_20261030"), f"Unexpected ROOT: {ROOT}"
OUT = os.path.join(ROOT, "04_structure_repertoire")
os.makedirs(OUT, exist_ok=True)


import subprocess

def download(url: str, dest: str, min_bytes: int = 100) -> None:
    if os.path.exists(dest) and os.path.getsize(dest) >= min_bytes:
        if not dest.endswith(".tar.gz") or subprocess.call(["gzip", "-t", dest]) == 0:
            print(f"[SKIP] {os.path.basename(dest)} ({os.path.getsize(dest):,} bytes)")
            return
    print(f"[GET ] {url} -> {dest}", flush=True)
    tmp = dest + ".part"
    subprocess.check_call(["wget", "-q", "-c", "--tries=10", "--timeout=60", "-O", tmp, url])
    sz = os.path.getsize(tmp)
    if sz < min_bytes:
        raise RuntimeError(f"Downloaded file too small ({sz} bytes): {dest}")
    if dest.endswith(".tar.gz") and subprocess.call(["gzip", "-t", tmp]) != 0:
        raise RuntimeError(f"gzip integrity check failed for {tmp}")
    os.replace(tmp, dest)
    print(f"[DONE] {os.path.basename(dest)} ({sz:,} bytes)", flush=True)


def parse_olga_pgen(olga_dir: str) -> str:
    """Parse exact marginal V-gene choice probabilities P_gen(V) from OLGA default models."""
    loci = [
        ("IGHV", "human_B_heavy"),
        ("IGKV", "human_B_kappa"),
        ("IGLV", "human_B_lambda"),
        ("TRBV", "human_T_beta"),
        ("TRAV", "human_T_alpha"),
    ]
    rows = []
    for locus_tag, model_name in loci:
        ldir = os.path.join(olga_dir, model_name)
        os.makedirs(ldir, exist_ok=True)
        base = f"https://raw.githubusercontent.com/statbiophys/OLGA/master/olga/default_models/{model_name}"
        marg_f = os.path.join(ldir, "model_marginals.txt")
        anch_f = os.path.join(ldir, "V_gene_CDR3_anchors.csv")
        download(f"{base}/model_marginals.txt", marg_f, min_bytes=50)
        download(f"{base}/V_gene_CDR3_anchors.csv", anch_f, min_bytes=50)

        with open(marg_f, "r", encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        probs = None
        for i, ln in enumerate(lines):
            if ln.startswith("@v_choice"):
                for j in range(i + 1, min(i + 6, len(lines))):
                    if lines[j].startswith("%"):
                        probs = [float(x) for x in lines[j][1:].split(",") if x.strip()]
                        break
                break
        if probs is None:
            raise RuntimeError(f"Could not parse @v_choice in {marg_f}")

        with open(anch_f, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            anchors = list(reader)

        if len(probs) != len(anchors):
            print(f"[WARN] {model_name}: probs={len(probs)} vs anchors={len(anchors)}")

        gene_prob = defaultdict(float)
        gene_func = {}
        for idx, a in enumerate(anchors):
            p = probs[idx] if idx < len(probs) else 0.0
            allele = a["gene"].strip()
            gene = allele.split("*")[0]
            func = a.get("function", "").strip()
            gene_prob[gene] += p
            gene_func[gene] = func
            rows.append({
                "locus": locus_tag,
                "model": model_name,
                "level": "allele",
                "name": allele,
                "gene": gene,
                "function": func,
                "pgen_v": p,
            })
        for g, gp in sorted(gene_prob.items()):
            rows.append({
                "locus": locus_tag,
                "model": model_name,
                "level": "gene",
                "name": g,
                "gene": g,
                "function": gene_func.get(g, ""),
                "pgen_v": gp,
            })

    out_tsv = os.path.join(OUT, "olga_human_v_pgen.tsv")
    with open(out_tsv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["locus", "model", "level", "name", "gene", "function", "pgen_v"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"[DONE] Wrote {len(rows)} OLGA P_gen(V) records -> {out_tsv}", flush=True)
    return out_tsv


def summarize_vane_repertoire() -> str:
    """Read-only summary of FACS-sorted Naive (N) and Memory (M) BCR TSVs from ~/vane/germ_presto/."""
    vane_dir = "./data/vane_presto"
    out_tsv = os.path.join(OUT, "vane_bcr_v_usage_summary.tsv")
    if os.path.exists(out_tsv) and os.path.getsize(out_tsv) > 1000:
        print(f"[SKIP] {out_tsv} already exists")
        return out_tsv

    tsv_files = sorted(glob.glob(os.path.join(vane_dir, "*_IGH.tsv")) + glob.glob(os.path.join(vane_dir, "*_light.tsv")))
    print(f"[INFO] Summarizing {len(tsv_files)} read-only TSV files from {vane_dir}...", flush=True)

    summary_rows = []
    for path in tsv_files:
        fname = os.path.basename(path)
        parts = fname.replace(".tsv", "").split("_")
        if len(parts) < 3:
            continue
        donor, subset, chain_group = parts[0], parts[1], parts[2]
        cohort = "HD" if donor.startswith("HD") else ("AChR_MG" if donor.startswith("AR") else "MuSK_MG")

        counts_seq = defaultdict(int)
        counts_umi = defaultdict(int)
        shm_sum = defaultdict(float)
        locus_total_seq = defaultdict(int)
        locus_total_umi = defaultdict(int)

        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                if row.get("productive", "").strip() not in ("T", "TRUE", "True", "true"):
                    continue
                v_call = row.get("v_call", "").strip()
                if not v_call:
                    continue
                # Primary V gene (strip allele, take first call if ambiguous within same gene; record ambiguity)
                first_gene = v_call.split(",")[0].split("*")[0].strip()
                loc = row.get("locus", "").strip() or first_gene[:3]
                try:
                    umi = int(float(row.get("consensus_count", "1") or 1))
                except ValueError:
                    umi = 1
                try:
                    vid = float(row.get("v_identity", "1.0") or 1.0)
                    if vid > 1.5:
                        vid = vid / 100.0
                except ValueError:
                    vid = 1.0

                key = (loc, first_gene)
                counts_seq[key] += 1
                counts_umi[key] += umi
                shm_sum[key] += (1.0 - vid)
                locus_total_seq[loc] += 1
                locus_total_umi[loc] += umi

        for (loc, gene), n_seq in sorted(counts_seq.items()):
            tot_s = max(locus_total_seq[loc], 1)
            tot_u = max(locus_total_umi[loc], 1)
            summary_rows.append({
                "donor": donor,
                "cohort": cohort,
                "subset": subset,
                "locus": loc,
                "v_gene": gene,
                "n_productive_seq": n_seq,
                "n_umi": counts_umi[(loc, gene)],
                "freq_seq": f"{n_seq / tot_s:.6f}",
                "freq_umi": f"{counts_umi[(loc, gene)] / tot_u:.6f}",
                "mean_shm_freq": f"{shm_sum[(loc, gene)] / n_seq:.5f}",
            })

    with open(out_tsv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["donor", "cohort", "subset", "locus", "v_gene", "n_productive_seq", "n_umi", "freq_seq", "freq_umi", "mean_shm_freq"],
            delimiter="\t",
        )
        w.writeheader()
        w.writerows(summary_rows)
    print(f"[DONE] Wrote {len(summary_rows)} vane V-usage rows -> {out_tsv}", flush=True)
    return out_tsv


def main():
    # 1. SAbDab2 summary
    download(
        "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/api/download/all-summary",
        os.path.join(OUT, "sabdab2_all_summary.csv"),
        min_bytes=5_000_000,
    )
    # 2. CoV-AbDab CSV + ANARCI JSON + IMGT-numbered PDB structures
    download(
        "https://opig.stats.ox.ac.uk/webapps/covabdab/static/downloads/CoV-AbDab_080224.csv",
        os.path.join(OUT, "CoV-AbDab_080224.csv"),
        min_bytes=1_000_000,
    )
    download(
        "https://opig.stats.ox.ac.uk/webapps/covabdab/static/downloads/CoV-AbDab_ANARCINumberings_080224.json",
        os.path.join(OUT, "CoV-AbDab_ANARCINumberings_080224.json"),
        min_bytes=1_000_000,
    )
    cov_pdb_tar = os.path.join(OUT, "CoV-AbDab_PDBStructures_080224.tar.gz")
    download(
        "https://opig.stats.ox.ac.uk/webapps/covabdab/static/downloads/CoV-AbDab_PDBStructures_080224.tar.gz",
        cov_pdb_tar,
        min_bytes=1_000_000,
    )

    # 3. STCRDab summary + TCR3d metadata & solved TCR-pMHC structure archive
    download(
        "https://opig.stats.ox.ac.uk/webapps/stcrdab/summary/all/",
        os.path.join(OUT, "stcrdab_summary_all.tsv"),
        min_bytes=50_000,
    )
    download(
        "https://tcr3d.ibbr.umd.edu/static/download/tcr_chain_data.tsv",
        os.path.join(OUT, "tcr3d_chain_data.tsv"),
        min_bytes=20_000,
    )
    download(
        "https://tcr3d.ibbr.umd.edu/static/download/tcr_complexes_data.tsv",
        os.path.join(OUT, "tcr3d_complexes_data.tsv"),
        min_bytes=20_000,
    )
    tcr_tar = os.path.join(OUT, "TCR_complexes.tar.gz")
    download(
        "https://tcr3d.ibbr.umd.edu/static/download/TCR_complexes.tar.gz",
        tcr_tar,
        min_bytes=1_000_000,
    )

    # 4. VDJdb 2026-06-03 release
    vdj_zip = os.path.join(OUT, "vdjdb-2026-06-03.zip")
    download(
        "https://github.com/antigenomics/vdjdb-db/releases/download/2026-06-03-ZENODO/vdjdb-2026-06-03.zip",
        vdj_zip,
        min_bytes=10_000_000,
    )
    vdj_extract = os.path.join(OUT, "vdjdb_2026_06_03")
    if not os.path.exists(os.path.join(vdj_extract, "vdjdb.txt")):
        os.makedirs(vdj_extract, exist_ok=True)
        with zipfile.ZipFile(vdj_zip, "r") as zf:
            zf.extractall(vdj_extract)
        print(f"[DONE] Extracted {vdj_zip} -> {vdj_extract}", flush=True)

    # Extract structure tarballs into clean subdirs
    cov_pdb_dir = os.path.join(OUT, "covabdab_pdbs")
    if not os.path.exists(cov_pdb_dir):
        os.makedirs(cov_pdb_dir, exist_ok=True)
        with tarfile.open(cov_pdb_tar, "r:gz") as tf:
            tf.extractall(cov_pdb_dir)
        print(f"[DONE] Extracted {cov_pdb_tar} -> {cov_pdb_dir}", flush=True)

    tcr_pdb_dir = os.path.join(OUT, "tcr3d_complexes")
    if not os.path.exists(tcr_pdb_dir):
        os.makedirs(tcr_pdb_dir, exist_ok=True)
        with tarfile.open(tcr_tar, "r:gz") as tf:
            tf.extractall(tcr_pdb_dir)
        print(f"[DONE] Extracted {tcr_tar} -> {tcr_pdb_dir}", flush=True)

    # 5. OLGA P_gen(V)
    parse_olga_pgen(os.path.join(OUT, "olga_models"))

    # 6. vane FACS Naive/Memory BCR summary
    summarize_vane_repertoire()

    with open(os.path.join(OUT, ".dbs_ready"), "w") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    print("ALL_DBS_AND_REPERTOIRES_READY", flush=True)


if __name__ == "__main__":
    main()
