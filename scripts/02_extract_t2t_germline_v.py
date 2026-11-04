#!/usr/bin/env python3
"""
02_extract_t2t_germline_v.py
Uniform, annotation-bias-free germline V-segment extractor across all 9 T2T/RefSeq primate assemblies
and all 5 adaptive immune receptor loci (IGHV, IGKV, IGLV, TRAV, TRBV).

Method:
1. Restrict canonical locus coordinates to assembled primary chromosomes (RefSeq NC_*) using sequence_report.jsonl
   so alt-loci/patches (e.g. GRCh38 NT_/NW_) never inflate copy counts.
2. Search whole primary-chromosome FASTA with combined IMGT Human + Rhesus V-REGION and J-REGION seeds (blastn).
3. Localize the canonical J-linked rearranging chromosome(s) per locus; flag dispersed (>4.5 Mb or non-J-chrom)
   hits as `is_orphon=True` (preserved in catalog, excluded from rearranging functional evolution set).
4. Merge overlapping same-strand V-hits (>=180 bp) into candidate V-exon windows ([hit_min-90, hit_max+120]).
5. Run NCBI IgBLAST (`immcantation_4.8.0.sif`, `-domain_system imgt -outfmt 19`) to segment each candidate into
   IMGT `fwr1`, `cdr1`, `fwr2`, `cdr2`, `fwr3`, extending 5' bases in-frame from the genomic window when
   `v_germline_start > 1`.
6. Map `(fwr1, cdr1, fwr2, cdr2, fwr3)` onto the exact 104-codon (312-nt) IMGT coordinate grid (FR1 1..26,
   CDR1 27..38, FR2 39..55, CDR2 56..65, FR3 66..104) using official IMGT human gap masks.
7. Scan +3..+45 bp 3' of Cys104 in the genomic window for the canonical 3' Recombination Signal Sequence (RSS):
   7-mer (`CACAGTG`) + 12-bp (IGKV) or 23-bp (IGHV/IGLV/TRAV/TRBV) spacer + 9-mer (`ACAAAAACC`).
8. Output complete TSV catalog, functional IMGT-104 codon FASTAs, and benchmark audit vs IMGT Human/Rhesus.
"""
import concurrent.futures
import csv
import glob
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from Bio import SeqIO
from Bio.Seq import Seq

ROOT = os.environ.get("BRIM_DATA_DIR", os.environ.get("BRIM_DATA_DIR", "./data"))
assert ROOT.endswith("analysis/t2t_germline_20261030")
GENOME_DIR = os.path.join(ROOT, "01_genomes")
OUT_DIR = os.path.join(ROOT, "02_germline_v")
IMGT_REF = os.path.join(OUT_DIR, "imgt_ref")

BLASTN = "blastn"
MAKEBLASTDB = "makeblastdb"
APPTAINER = "/usr/bin/apptainer"
SIF = "./containers/immcantation_4.8.0.sif"

LOCI = ["IGHV", "IGKV", "IGLV", "TRAV", "TRBV"]
J_MAP = {
    "IGHV": ["IGHJ"],
    "IGKV": ["IGKJ"],
    "IGLV": ["IGLJ"],
    "TRAV": ["TRAJ", "TRDJ"],
    "TRBV": ["TRBJ"],
}
RSS_SPACER = {
    "IGHV": (22, 24),
    "IGKV": (11, 13),
    "IGLV": (22, 24),
    "TRAV": (22, 24),
    "TRBV": (22, 24),
}
DOMAIN_SLOTS = [("fwr1", 26), ("cdr1", 12), ("fwr2", 17), ("cdr2", 10), ("fwr3", 39)]
DOMAIN_SLICE = {
    "fwr1": (0, 78),
    "cdr1": (78, 114),
    "fwr2": (114, 165),
    "cdr2": (165, 195),
    "fwr3": (195, 312),
}


def load_human_imgt_templates():
    """
    Build:
    1) allele_templates[locus][allele] = dict(domain -> list of bool [True=codon, False=gap])
    2) mask_by_len[locus][domain][n_codons] = most frequent official IMGT mask of that length
    """
    allele_templates = {loc: {} for loc in LOCI}
    mask_counts = {loc: {d: defaultdict(Counter) for d, _ in DOMAIN_SLOTS} for loc in LOCI}
    human_primary_f = {loc: {} for loc in LOCI}

    for loc in LOCI:
        fa = os.path.join(IMGT_REF, "human", f"imgt_human_{loc}.fasta")
        for rec in SeqIO.parse(fa, "fasta"):
            parts = rec.description.split("|")
            allele = parts[1].strip()
            func = parts[3].strip()
            s = str(rec.seq).upper()
            if len(s) < 312:
                continue
            s312 = s[:312]
            if func == "F" and allele.endswith("*01"):
                human_primary_f[loc][allele.split("*")[0]] = s312
            dom_masks = {}
            ok = True
            for dname, n_slots in DOMAIN_SLOTS:
                st, en = DOMAIN_SLICE[dname]
                sub = s312[st:en]
                cods = [sub[i:i + 3] for i in range(0, len(sub), 3)]
                if any(("." in c and c != "...") for c in cods):
                    ok = False
                    break
                mask = tuple(c != "..." for c in cods)
                if len(mask) != n_slots:
                    ok = False
                    break
                dom_masks[dname] = mask
                if func in ("F", "ORF"):
                    mask_counts[loc][dname][sum(mask)][mask] += 1
            if ok:
                allele_templates[loc][allele] = dom_masks

    mask_by_len = {loc: {d: {} for d, _ in DOMAIN_SLOTS} for loc in LOCI}
    for loc in LOCI:
        for d, _ in DOMAIN_SLOTS:
            for n_cod, ctr in mask_counts[loc][d].items():
                mask_by_len[loc][d][n_cod] = ctr.most_common(1)[0][0]
    return allele_templates, mask_by_len, human_primary_f


def imgt_default_mask(domain: str, n_cod: int, n_slots: int):
    """Official Lefranc IMGT fallback mask if an unseen loop/framework codon count occurs."""
    if n_cod <= 0:
        return tuple([False] * n_slots)
    if n_cod == n_slots:
        return tuple([True] * n_slots)
    if n_cod > n_slots:
        return None
    gaps = n_slots - n_cod
    if domain in ("cdr1", "cdr2"):
        left = (n_cod + 1) // 2
        right = n_cod - left
        return tuple([True] * left + [False] * gaps + [True] * right)
    if domain == "fwr1":
        # IMGT FR1 gap is around pos 10 (index 9)
        left = min(9, n_cod)
        right = n_cod - left
        return tuple([True] * left + [False] * gaps + [True] * right)
    if domain == "fwr2":
        # IMGT FR2 gaps (rare, e.g. IGLV) are after Trp41 around pos 47..48
        left = min(8, n_cod)
        right = n_cod - left
        return tuple([True] * left + [False] * gaps + [True] * right)
    # fwr3: IMGT gaps at pos 73 (index 7 of FR3) and pos 81 (index 15 of FR3), plus safe fallback for truncated pseudogenes
    mask = [True] * n_slots
    priority_gaps = [7, 15, 16, 8, 14] + [i for i in range(n_slots - 2, -1, -1) if i not in {7, 15, 16, 8, 14}]
    for g_idx in priority_gaps[:gaps]:
        mask[g_idx] = False
    return tuple(mask)


def apply_mask(ungapped_nt: str, mask: tuple) -> str:
    cods = [ungapped_nt[i:i + 3] for i in range(0, len(ungapped_nt), 3)]
    assert len(cods) == sum(mask)
    it = iter(cods)
    return "".join(next(it) if m else "..." for m in mask)


def build_seed_fastas(seed_dir: str):
    os.makedirs(seed_dir, exist_ok=True)
    for loc in LOCI:
        v_out = os.path.join(seed_dir, f"seeds_{loc}_V.fasta")
        if not os.path.exists(v_out):
            with open(v_out, "w") as fh:
                for sp in ["human", "rhesus_monkey"]:
                    fa = os.path.join(IMGT_REF, sp, f"imgt_{sp}_{loc}.fasta")
                    # For TRAV, also include TRDV1/2/3 shared locus V genes? Keep TRAV pure so TRDV doesn't pollute TRAV
                    for rec in SeqIO.parse(fa, "fasta"):
                        parts = rec.description.split("|")
                        allele, func = parts[1].strip(), parts[3].strip()
                        if func in ("F", "ORF") and allele.endswith("*01"):
                            seq = str(rec.seq).replace(".", "").upper()
                            fh.write(f">{sp}_{allele}\n{seq}\n")

        j_out = os.path.join(seed_dir, f"seeds_{loc}_J.fasta")
        if not os.path.exists(j_out):
            with open(j_out, "w") as fh:
                for sp in ["human", "rhesus_monkey"]:
                    for j_tag in J_MAP[loc]:
                        fa = os.path.join(IMGT_REF, sp, f"imgt_{sp}_{j_tag}.fasta")
                        if not os.path.exists(fa):
                            continue
                        for rec in SeqIO.parse(fa, "fasta"):
                            parts = rec.description.split("|")
                            allele = parts[1].strip()
                            seq = str(rec.seq).replace(".", "").upper()
                            fh.write(f">{sp}_{allele}\n{seq}\n")


def score_3prime_rss(downstream_nt: str, spacer_range: tuple):
    """
    Scan offsets 2..46 nt downstream of Cys104 end for the best 3' RSS:
    Heptamer consensus: CACAGTG (7 nt)
    Spacer: spacer_range (e.g. 22..24 or 11..13)
    Nonamer consensus: ACAAAAACC (9 nt)
    Returns (best_score, ca_canonical, offset, heptamer, spacer_len, nonamer)
    """
    hept_cons = "CACAGTG"
    nona_cons = "ACAAAAACC"
    # Weight first 3 bases CAC higher as they are required for RAG cleavage
    hept_w = [3.0, 3.0, 2.0, 1.0, 1.5, 1.0, 1.5]  # max = 13.0
    nona_w = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.0, 1.5]  # max = 13.0 -> total max = 26.0

    best = (-1.0, False, -1, "", 0, "")
    s = downstream_nt.upper()
    for off in range(2, min(48, max(3, len(s) - 25))):
        hept = s[off:off + 7]
        if len(hept) < 7 or "N" in hept:
            continue
        ca_ok = hept.startswith("CA")
        h_sc = sum(w for b, c, w in zip(hept, hept_cons, hept_w) if b == c)
        for sp_len in range(spacer_range[0], spacer_range[1] + 1):
            n_start = off + 7 + sp_len
            nona = s[n_start:n_start + 9]
            if len(nona) < 9 or "N" in nona:
                continue
            n_sc = sum(w for b, c, w in zip(nona, nona_cons, nona_w) if b == c)
            # Bonus for exact canonical spacer (12 or 23) and CA start
            sp_mid = (spacer_range[0] + spacer_range[1]) // 2
            sp_bonus = 1.0 if sp_len == sp_mid else 0.0
            ca_bonus = 4.0 if ca_ok else 0.0
            tot = h_sc + n_sc + sp_bonus + ca_bonus
            # Raw 16-bp unweighted matches as easy intuitive count
            raw_m = sum(1 for a, b in zip(hept, hept_cons) if a == b) + sum(1 for a, b in zip(nona, nona_cons) if a == b)
            cand = (tot, ca_ok, off, hept, sp_len, nona, raw_m)
            if cand[0] > best[0]:
                best = cand
    if best[0] < 0:
        return (0.0, False, -1, "", 0, "", 0)
    return best


def get_primary_chrom_fasta(aid: str, acc: str, work_dir: str) -> str:
    """Extract only assembled primary chromosomes (NC_* / assembled-molecule) so patches/unplaced scaffolds don't duplicate."""
    out_fa = os.path.join(work_dir, f"{aid}_primary_chroms.fna")
    if os.path.exists(out_fa) and os.path.getsize(out_fa) > 100_000_000:
        return out_fa

    data_dir = os.path.join(GENOME_DIR, aid, "ncbi_dataset", "data", acc)
    seq_rep = os.path.join(data_dir, "sequence_report.jsonl")
    allowed = set()
    if os.path.exists(seq_rep):
        with open(seq_rep, "r", encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                if d.get("role") == "assembled-molecule":
                    if d.get("refseqAccession"):
                        allowed.add(d["refseqAccession"])
                    if d.get("genbankAccession"):
                        allowed.add(d["genbankAccession"])

    raw_fnas = glob.glob(os.path.join(data_dir, "*_genomic.fna"))
    assert raw_fnas, f"Missing genomic.fna for {aid}"
    n_kept = 0
    with open(out_fa, "w") as out_fh:
        for rec in SeqIO.parse(raw_fnas[0], "fasta"):
            rid = rec.id.split()[0]
            if (allowed and rid in allowed) or (not allowed and rid.startswith("NC_")):
                SeqIO.write(rec, out_fh, "fasta")
                n_kept += 1
    assert n_kept >= 15, f"Too few primary chromosomes ({n_kept}) for {aid}"
    subprocess.check_call([MAKEBLASTDB, "-in", out_fa, "-dbtype", "nucl", "-out", out_fa + ".db"], stdout=subprocess.DEVNULL)
    return out_fa


def process_assembly(aid: str, species: str, acc: str, seed_dir: str, allele_templates: dict, mask_by_len: dict):
    asm_dir = os.path.join(OUT_DIR, "per_assembly", aid)
    os.makedirs(asm_dir, exist_ok=True)
    primary_fa = get_primary_chrom_fasta(aid, acc, asm_dir)
    chrom_seqs = {rec.id.split()[0]: str(rec.seq).upper() for rec in SeqIO.parse(primary_fa, "fasta")}

    all_records = []
    for loc in LOCI:
        j_seed = os.path.join(seed_dir, f"seeds_{loc}_J.fasta")
        j_tsv = os.path.join(asm_dir, f"{loc}_J_blast.tsv")
        v_seed = os.path.join(seed_dir, f"seeds_{loc}_V.fasta")
        v_tsv = os.path.join(asm_dir, f"{loc}_V_blast.tsv")
        igb_tsv = os.path.join(asm_dir, f"{loc}_igblast.tsv")
        already_completed = os.path.exists(igb_tsv) and os.path.getsize(igb_tsv) > 200

        # 1. BLAST J-seeds to find canonical rearranging locus chromosome & J-cluster center
        if not (already_completed and os.path.exists(j_tsv)):
            tmp_j = j_tsv + ".tmp"
            subprocess.check_call([
                BLASTN, "-task", "blastn", "-word_size", "9", "-evalue", "1e-4",
                "-query", j_seed, "-db", primary_fa + ".db",
                "-outfmt", "6 qseqid sseqid pident length qlen sstart send bitscore evalue",
                "-num_threads", "8", "-out", tmp_j
            ])
            os.replace(tmp_j, j_tsv)
        j_hits_by_chrom = defaultdict(list)
        with open(j_tsv) as fh:
            for ln in fh:
                p = ln.strip().split("\t")
                if int(p[3]) >= 30 and float(p[2]) >= 78.0:
                    s1, s2 = int(p[5]), int(p[6])
                    j_hits_by_chrom[p[1]].append((min(s1, s2), max(s1, s2), float(p[7])))

        # 2. BLAST V-seeds
        if not (already_completed and os.path.exists(v_tsv) and os.path.getsize(v_tsv) > 100):
            tmp_v = v_tsv + ".tmp"
            subprocess.check_call([
                BLASTN, "-task", "blastn", "-word_size", "11", "-evalue", "1e-8",
                "-max_target_seqs", "10000",
                "-query", v_seed, "-db", primary_fa + ".db",
                "-outfmt", "6 qseqid sseqid pident length qlen qstart qend sstart send bitscore evalue",
                "-num_threads", "8", "-out", tmp_v
            ])
            os.replace(tmp_v, v_tsv)

        raw_intervals = []
        with open(v_tsv) as fh:
            for ln in fh:
                p = ln.strip().split("\t")
                chrom = p[1]
                length = int(p[3])
                if length < 120:
                    continue
                qlen, qst, qen = int(p[4]), int(p[5]), int(p[6])
                sst, sen = int(p[7]), int(p[8])
                bits = float(p[9])
                strand = "+" if sst < sen else "-"
                if strand == "+":
                    est_start = max(1, sst - (qst - 1))
                    est_end = sen + (qlen - qen)
                else:
                    est_start = max(1, sen - (qlen - qen))
                    est_end = sst + (qst - 1)
                raw_intervals.append((chrom, strand, min(sst, sen), max(sst, sen), est_start, est_end, bits, p[0]))

        # Sort & merge overlapping intervals on same (chrom, strand)
        raw_intervals.sort(key=lambda x: (x[0], x[1], x[4], -x[6]))
        merged = []
        for ch, st, hmin, hmax, emin, emax, bits, qid in raw_intervals:
            if merged and merged[-1][0] == ch and merged[-1][1] == st and emin <= merged[-1][5] + 40:
                prev = merged[-1]
                best_b, best_q = (bits, qid) if bits > prev[6] else (prev[6], prev[7])
                merged[-1] = (ch, st, min(prev[2], hmin), max(prev[3], hmax), min(prev[4], emin), max(prev[5], emax), best_b, best_q)
            else:
                merged.append((ch, st, hmin, hmax, emin, emax, bits, qid))

        # Keep candidates with span >= 180 bp
        candidates = [m for m in merged if (m[5] - m[4] + 1) >= 180]

        # Jointly identify canonical J-linked rearranging locus: chromosome + J-cluster within 4.5 Mb of V-cluster
        canonical_j_intervals = {}
        best_locus_score = (-1, -1.0)
        for ch, jlist in j_hits_by_chrom.items():
            v_on_ch = [m[4] for m in candidates if m[0] == ch]
            if not v_on_ch:
                continue
            for ja, jb, jbits in jlist:
                near_v = sum(1 for vx in v_on_ch if abs(vx - ja) <= 4_500_000)
                near_j_bits = sum(x[2] for x in jlist if abs(x[0] - ja) <= 150_000)
                sc = (near_v, near_j_bits)
                if sc > best_locus_score and near_v >= 5:
                    best_locus_score = sc
                    j_cluster_pts = sorted(x[0] for x in jlist if abs(x[0] - ja) <= 150_000)
                    med_j = j_cluster_pts[len(j_cluster_pts) // 2]
                    canonical_j_intervals = {ch: (med_j - 4_500_000, med_j + 4_500_000, med_j)}

        win_fa = os.path.join(asm_dir, f"{loc}_candidates_win.fasta")
        win_meta = {}
        with open(win_fa, "w") as fh:
            for idx, (ch, st, hmin, hmax, emin, emax, bits, qid) in enumerate(candidates, 1):
                cid = f"{aid}|{loc}|cand{idx:04d}"
                cseq = chrom_seqs[ch]
                w_start = max(0, emin - 1 - 90)
                w_end = min(len(cseq), emax + 150)
                sub = cseq[w_start:w_end]
                if st == "-":
                    sub = str(Seq(sub).reverse_complement())
                fh.write(f">{cid}\n{sub}\n")
                win_meta[cid] = {
                    "chrom": ch,
                    "strand": st,
                    "win_start_1b": w_start + 1,
                    "win_end_1b": w_end,
                    "win_seq": sub,
                    "blast_bits": bits,
                    "blast_seed": qid,
                }

        if not win_meta:
            continue

        # 3. Run IgBLAST (-outfmt 19) inside container
        ig_type = "Ig" if loc.startswith("IG") else "TCR"
        db_prefix = "ig" if loc.startswith("IG") else "tr"
        rel_in = os.path.relpath(win_fa, ROOT)
        rel_out = os.path.relpath(igb_tsv, ROOT)

        if not already_completed:
            tmp_rel_out = rel_out + ".tmp"
            cmd = (
                f"export IGDATA=/usr/local/share/igblast && "
                f"igblastn -germline_db_V /usr/local/share/igblast/database/imgt_human_{db_prefix}_v "
                f"-germline_db_D /usr/local/share/igblast/database/imgt_human_{db_prefix}_d "
                f"-germline_db_J /usr/local/share/igblast/database/imgt_human_{db_prefix}_j "
                f"-auxiliary_data /usr/local/share/igblast/optional_file/human_gl.aux "
                f"-domain_system imgt -ig_seqtype {ig_type} -organism human "
                f"-outfmt 19 -num_threads 8 -query /work/{rel_in} -out /work/{tmp_rel_out}"
            )
            subprocess.check_call([APPTAINER, "exec", "--bind", f"{ROOT}:/work", SIF, "bash", "-c", cmd])
            os.replace(igb_tsv + ".tmp", igb_tsv)

        with open(igb_tsv, "r", encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                cid = r["sequence_id"]
                if cid not in win_meta:
                    continue
                meta = win_meta[cid]
                v_call_raw = r.get("v_call", "").strip()
                if not v_call_raw:
                    continue
                top_human_allele = v_call_raw.split(",")[0].strip()
                # Check locus match! (e.g. prevent IGKV window hitting IGLV db entry or TRAV hitting TRBV/TRDV)
                if not top_human_allele.startswith(loc[:4]):
                    # Allow TRAV29/DV5 etc. which start with TRAV; exclude pure TRDV1/2/3 when scanning TRAV
                    continue

                f1 = (r.get("fwr1") or "").upper()
                c1 = (r.get("cdr1") or "").upper()
                f2 = (r.get("fwr2") or "").upper()
                c2 = (r.get("cdr2") or "").upper()
                f3 = (r.get("fwr3") or "").upper()
                if not (f1 and c1 and f2 and c2 and f3):
                    continue

                vss = int(r["v_sequence_start"])
                vse = int(r["v_sequence_end"])
                vgs = int(r["v_germline_start"])
                wseq = meta["win_seq"]

                # Extend 5' end of fwr1 from genomic window if v_germline_start > 1 (in-frame recovery)
                if 1 < vgs <= 24 and (vss - 1) >= (vgs - 1):
                    prefix_nt = wseq[(vss - 1) - (vgs - 1):(vss - 1)]
                    f1 = prefix_nt + f1
                    vss = vss - (vgs - 1)
                    vgs = 1

                # Note: in IgBLAST AIRR, `fwr3` ends RIGHT AT Cys104 (`TGT`/`TGC`), while `v_sequence_end`
                # includes the 6..30 bp germline CDR3 tail! Locate exact end of `fwr3` in `wseq`:
                ungapped_104 = f1 + c1 + f2 + c2 + f3
                c104_end_0b = (vss - 1) + len(ungapped_104)  # 0-based slice end of Cys104 in wseq

                # Genomic coordinates of Codon 1..Cys104
                if meta["strand"] == "+":
                    g_start = meta["win_start_1b"] + (vss - 1)
                    g_end = meta["win_start_1b"] + c104_end_0b - 1
                else:
                    g_end = meta["win_end_1b"] - (vss - 1)
                    g_start = meta["win_end_1b"] - c104_end_0b + 1

                # Check canonical J-linked locus vs dispersed orphon
                ch = meta["chrom"]
                if ch in canonical_j_intervals:
                    j_lo, j_hi, j_med = canonical_j_intervals[ch]
                    is_orphon = not (j_lo <= g_start <= j_hi)
                    dist_to_j = abs(g_start - j_med)
                else:
                    is_orphon = True
                    dist_to_j = -1

                # Score 3' RSS immediately downstream of Cys104
                downstream = wseq[c104_end_0b:c104_end_0b + 90]
                rss_sc, rss_ca, rss_off, rss_hept, rss_splen, rss_nona, rss_raw16 = score_3prime_rss(
                    downstream, RSS_SPACER[loc]
                )

                # Check frameshifts & map to IMGT 1..104 (312 nt)
                dom_vals = {"fwr1": f1, "cdr1": c1, "fwr2": f2, "cdr2": c2, "fwr3": f3}
                frameshift = any(len(v) % 3 != 0 for v in dom_vals.values())
                insertion_gt_104 = False
                gapped_domains = {}

                ref_masks = allele_templates[loc].get(top_human_allele)
                if not frameshift:
                    for dname, n_slots in DOMAIN_SLOTS:
                        val = dom_vals[dname].replace("-", "").replace(".", "")
                        nc = len(val) // 3
                        if nc > n_slots:
                            insertion_gt_104 = True
                            extra = nc - n_slots
                            # Preserve conserved anchors (Cys23 at 3' of FR1, Trp41 at 5' of FR2, Cys104 at 3' of FR3)
                            if dname == "fwr1":
                                val = val[-n_slots * 3:]
                            elif dname in ("cdr1", "cdr2"):
                                mid = (n_slots // 2) * 3
                                val = val[:mid] + val[mid + extra * 3:]
                            elif dname == "fwr2":
                                mid = min(8, n_slots - 1) * 3
                                val = val[:mid] + val[mid + extra * 3:]
                            else:  # fwr3: trim HV4 loop insertion around index 15 so Cys104 at 3' end is preserved
                                mid = min(15, n_slots - 3) * 3
                                val = val[:mid] + val[mid + extra * 3:]
                            nc = n_slots
                            assert len(val) == n_slots * 3
                        if ref_masks and sum(ref_masks[dname]) == nc:
                            mask = ref_masks[dname]
                        elif nc in mask_by_len[loc][dname]:
                            mask = mask_by_len[loc][dname][nc]
                        else:
                            mask = imgt_default_mask(dname, nc, n_slots)
                        gapped_domains[dname] = apply_mask(val, mask)

                if frameshift or "N" in ungapped_104:
                    imgt_312 = ""
                    aa_104 = ""
                    has_stop = True
                    c23, w41, c104 = "", "", ""
                    n_codons = len(ungapped_104) // 3
                else:
                    imgt_312 = "".join(gapped_domains[d] for d, _ in DOMAIN_SLOTS)
                    assert len(imgt_312) == 312
                    cods = [imgt_312[i:i + 3] for i in range(0, 312, 3)]
                    aa_list = [str(Seq(c).translate()) if c != "..." else "-" for c in cods]
                    aa_104 = "".join(aa_list)
                    has_stop = "*" in aa_104
                    c23, w41, c104 = aa_list[22], aa_list[40], aa_list[103]
                    n_codons = sum(1 for a in aa_list if a != "-")

                triad_ok = (c23 == "C" and w41 == "W" and c104 == "C")
                rss_ok = (rss_ca and (rss_hept.startswith("CAC") or rss_raw16 >= 8) and rss_raw16 >= 6)

                if frameshift or has_stop or n_codons < 88:
                    status = "P"
                elif (not is_orphon) and triad_ok and rss_ok:
                    status = "F"
                else:
                    status = "ORF"

                top_gene = top_human_allele.split("*")[0]
                subfam_m = re.match(r"^([A-Z]+\d+)", top_gene)
                subfam = subfam_m.group(1) if subfam_m else top_gene

                all_records.append({
                    "assembly_id": aid,
                    "species": species,
                    "accession": acc,
                    "locus": loc,
                    "gene_uid": f"{aid}__{loc}_{ch}_{g_start}",
                    "chrom": ch,
                    "start_1b": g_start,
                    "end_1b": g_end,
                    "strand": meta["strand"],
                    "dist_to_j_bp": dist_to_j,
                    "is_orphon": is_orphon,
                    "status": status,
                    "closest_human_allele": top_human_allele,
                    "closest_human_gene": top_gene,
                    "subfamily": subfam,
                    "v_identity_to_human": r.get("v_identity", ""),
                    "n_codons": n_codons,
                    "frameshift": frameshift,
                    "has_stop": has_stop,
                    "c23_w41_c104": f"{c23}/{w41}/{c104}",
                    "insertion_gt_104": insertion_gt_104,
                    "rss_ca": rss_ca,
                    "rss_matches_16": rss_raw16,
                    "rss_offset_bp": rss_off,
                    "rss_heptamer": rss_hept,
                    "rss_spacer_bp": rss_splen,
                    "rss_nonamer": rss_nona,
                    "imgt_312_nt": imgt_312,
                    "imgt_104_aa": aa_104,
                })

    # Deduplicate any identical coordinate calls within assembly
    dedup = {}
    for rec in all_records:
        key = (rec["locus"], rec["chrom"], rec["strand"], rec["start_1b"] // 50)
        if key not in dedup or (rec["status"] == "F" and dedup[key]["status"] != "F"):
            dedup[key] = rec
    final_list = sorted(dedup.values(), key=lambda x: (x["locus"], x["chrom"], x["start_1b"]))
    print(
        f"[DONE] {aid:25s}: "
        + ", ".join(
            f"{loc} F={sum(1 for r in final_list if r['locus']==loc and r['status']=='F')}"
            f"/tot={sum(1 for r in final_list if r['locus']==loc)}"
            for loc in LOCI
        ),
        flush=True,
    )
    return final_list


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    seed_dir = os.path.join(OUT_DIR, "seeds")
    build_seed_fastas(seed_dir)
    allele_templates, mask_by_len, human_primary_f = load_human_imgt_templates()

    manifest = []
    with open(os.path.join(GENOME_DIR, "t2t_manifest.tsv")) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            manifest.append((r["assembly_id"], r["species"], r["accession"]))

    all_rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=9) as ex:
        futs = [
            ex.submit(process_assembly, aid, sp, acc, seed_dir, allele_templates, mask_by_len)
            for aid, sp, acc in manifest
        ]
        for f in concurrent.futures.as_completed(futs):
            all_rows.extend(f.result())

    all_rows.sort(key=lambda x: (x["locus"], x["assembly_id"], x["chrom"], x["start_1b"]))

    # Save full catalog TSV
    catalog_tsv = os.path.join(OUT_DIR, "t2t_primate_germline_v_catalog.tsv")
    fields = list(all_rows[0].keys())
    with open(catalog_tsv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(all_rows)

    # Save summary counts table per assembly x locus
    summary_tsv = os.path.join(OUT_DIR, "t2t_primate_germline_v_counts_summary.tsv")
    with open(summary_tsv, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["assembly_id", "species", "locus", "n_functional_F", "n_ORF", "n_pseudogene_P", "n_orphon", "n_total"])
        for aid, sp, _ in manifest:
            for loc in LOCI:
                sub = [r for r in all_rows if r["assembly_id"] == aid and r["locus"] == loc]
                n_f = sum(1 for r in sub if r["status"] == "F")
                n_o = sum(1 for r in sub if r["status"] == "ORF")
                n_p = sum(1 for r in sub if r["status"] == "P")
                n_orph = sum(1 for r in sub if r["is_orphon"])
                w.writerow([aid, sp, loc, n_f, n_o, n_p, n_orph, len(sub)])

    # Write per-locus functional IMGT-104 codon FASTAs (converting '...' gaps to '---' for IQ-TREE / HyPhy)
    # Note: Keep BOTH Human_CHM13 (primary haploid T2T human in 8-species phylogeny) and a full 9-assembly set!
    fasta_dir = os.path.join(OUT_DIR, "functional_codon_fastas")
    os.makedirs(fasta_dir, exist_ok=True)
    for loc in LOCI:
        # 8-species unique haploid T2T set (using Human_CHM13 for Homo sapiens, plus Human_GRCh38 sensitivity set)
        for panel_name, drop_aid in [("t2t8_chm13", "Human_GRCh38"), ("t2t8_grch38", "Human_CHM13"), ("all9", None)]:
            fa_out = os.path.join(fasta_dir, f"{loc}_{panel_name}_functional_imgt104.fasta")
            with open(fa_out, "w") as fh:
                for r in all_rows:
                    if r["locus"] == loc and r["status"] == "F" and r["assembly_id"] != drop_aid:
                        seq_hyphy = r["imgt_312_nt"].replace(".", "-")
                        hdr = f"{r['assembly_id']}_{r['subfamily']}_{r['closest_human_gene'].replace('/','_')}_{r['chrom']}_{r['start_1b']}"
                        hdr = re.sub(r"[^A-Za-z0-9_]", "_", hdr)
                        fh.write(f">{hdr}\n{seq_hyphy}\n")

    # Benchmark Human_GRCh38 & Human_CHM13 extracted functional genes against IMGT Human reference
    print("\n=== BENCHMARK vs OFFICIAL IMGT HUMAN (*01 Functional Subfamilies/Genes) ===")
    for aid in ["Human_GRCh38", "Human_CHM13"]:
        for loc in LOCI:
            ext_f = [r for r in all_rows if r["assembly_id"] == aid and r["locus"] == loc and r["status"] == "F"]
            ext_subfams = set(r["subfamily"] for r in ext_f)
            ref_subfams = set(re.match(r"^([A-Z]+\d+)", g).group(1) for g in human_primary_f[loc])
            exact_seq_matches = sum(1 for r in ext_f if r["imgt_312_nt"] in set(human_primary_f[loc].values()))
            print(
                f"  {aid:14s} {loc}: extracted_F={len(ext_f):2d}, "
                f"IMGT_ref_*01_genes={len(human_primary_f[loc]):2d} (includes CNV duplicates), "
                f"subfamily_recall={len(ext_subfams & ref_subfams)}/{len(ref_subfams)}, "
                f"100pct_exact_IMGT_*01_nt={exact_seq_matches}"
            )

    print("\nALL_EXTRACTION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
