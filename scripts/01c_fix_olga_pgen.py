#!/usr/bin/env python3
"""
Fix OLGA P_gen(V) table by reading exact @v_choice allele-to-index mappings from model_params.txt
instead of assuming row parity with V_gene_CDR3_anchors.csv (fixes human_T_beta 89 vs 134).
"""
import csv
import os
import re
import subprocess
from collections import defaultdict

ROOT = os.environ.get("BRIM_DATA_DIR", os.environ.get("BRIM_DATA_DIR", "./data"))
OUT = os.path.join(ROOT, "04_structure_repertoire")
OLGA_DIR = os.path.join(OUT, "olga_models")

LOCI = [
    ("IGHV", "human_B_heavy"),
    ("IGKV", "human_B_kappa"),
    ("IGLV", "human_B_lambda"),
    ("TRBV", "human_T_beta"),
    ("TRAV", "human_T_alpha"),
]

rows = []
for locus_tag, model_name in LOCI:
    ldir = os.path.join(OLGA_DIR, model_name)
    os.makedirs(ldir, exist_ok=True)
    base = f"https://raw.githubusercontent.com/statbiophys/OLGA/master/olga/default_models/{model_name}"
    param_f = os.path.join(ldir, "model_params.txt")
    marg_f = os.path.join(ldir, "model_marginals.txt")
    anch_f = os.path.join(ldir, "V_gene_CDR3_anchors.csv")
    if not os.path.exists(param_f) or os.path.getsize(param_f) < 100:
        subprocess.check_call(["wget", "-q", "-O", param_f, f"{base}/model_params.txt"])

    # Parse anchor functionality map
    func_map = {}
    with open(anch_f, "r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            g = r["gene"].strip()
            func_map[g] = r.get("function", "").strip()
            func_map[g.split("*")[0]] = r.get("function", "").strip()

    # Parse exact V_gene index -> allele name from model_params.txt
    idx_to_allele = {}
    in_v_event = False
    with open(param_f, "r", encoding="utf-8") as fh:
        for raw in fh:
            ln = raw.strip()
            if ln.startswith("#GeneChoice;V_gene;"):
                in_v_event = True
                continue
            if in_v_event:
                if ln.startswith("#") or ln.startswith("@"):
                    break
                if ln.startswith("%"):
                    # Format: %allele_name;SEQUENCE;INDEX
                    parts = ln[1:].split(";")
                    if len(parts) >= 3:
                        allele_raw = parts[0].strip()
                        # Clean IMGT pipe headers if present (e.g., L36092|TRBV1*01|Homo_sapiens|...)
                        if "|" in allele_raw:
                            p2 = [x for x in allele_raw.split("|") if "*" in x]
                            allele = p2[0] if p2 else allele_raw.split("|")[1]
                        else:
                            allele = allele_raw
                        idx = int(parts[-1])
                        idx_to_allele[idx] = allele

    # Parse @v_choice marginal probabilities
    probs = None
    with open(marg_f, "r", encoding="utf-8") as fh:
        lines = [x.strip() for x in fh if x.strip()]
    for i, ln in enumerate(lines):
        if ln.startswith("@v_choice"):
            for j in range(i + 1, min(i + 6, len(lines))):
                if lines[j].startswith("%"):
                    probs = [float(x) for x in lines[j][1:].split(",") if x.strip()]
                    break
            break

    assert probs is not None and len(probs) == len(idx_to_allele), (
        f"{model_name}: probs={len(probs) if probs else None} vs params={len(idx_to_allele)}"
    )
    print(f"[OK] {model_name} ({locus_tag}): exact match {len(probs)} alleles, sum(P)={sum(probs):.6f}")

    gene_prob = defaultdict(float)
    for idx in sorted(idx_to_allele):
        allele = idx_to_allele[idx]
        # Clean IMGT prefix if any remains
        m = re.search(r"((?:IGHV|IGKV|IGLV|TRAV|TRBV|TRDV)[A-Za-z0-9/.\-]+\*\d+)", allele)
        if m:
            allele = m.group(1)
        gene = allele.split("*")[0]
        p = probs[idx]
        gene_prob[gene] += p
        rows.append({
            "locus": locus_tag,
            "model": model_name,
            "level": "allele",
            "name": allele,
            "gene": gene,
            "function": func_map.get(allele, func_map.get(gene, "")),
            "pgen_v": f"{p:.8e}",
        })
    for gene, gp in sorted(gene_prob.items()):
        rows.append({
            "locus": locus_tag,
            "model": model_name,
            "level": "gene",
            "name": gene,
            "gene": gene,
            "function": func_map.get(gene, ""),
            "pgen_v": f"{gp:.8e}",
        })

out_tsv = os.path.join(OUT, "olga_human_v_pgen.tsv")
with open(out_tsv, "w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["locus", "model", "level", "name", "gene", "function", "pgen_v"], delimiter="\t")
    w.writeheader()
    w.writerows(rows)
print(f"[DONE] Wrote verified {len(rows)} rows -> {out_tsv}")
