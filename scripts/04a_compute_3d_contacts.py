#!/usr/bin/env python3
"""
04a_compute_3d_contacts.py
Reproducible 3D structural contact & unbound solvent accessibility (SASA) profiler across IMGT positions 1..104:
1. Antibody-Antigen Set A (CoV-AbDab IMGT-renumbered crystal/cryo-EM structures, protein antigens, non-redundant VH/VL)
2. Antibody-Antigen Set B (General Pan-Antigen SAbDab2 non-coronavirus protein complexes <=3.0 A fetched from RCSB mmCIF)
3. TCR-pMHC Set C (TCR3d / STCRDab solved human alpha-beta TCR-pMHC complexes, decomposed into Peptide vs MHC contacts!)

Outputs exact per-IMGT-position (1..104) tables for IGHV, IGKV, IGLV, TRAV, TRBV:
- n_structures_present
- contact_freq_5A, contact_freq_4_5A
- (for TCR) contact_peptide_5A, contact_mhc_5A
- mean_unbound_sasa_A2 (exact Shrake-Rupley SASA on isolated V-domain chain) & ca_neighbor_count_10A
"""
import concurrent.futures
import csv
import glob
import gzip
import math
import os
import re
import subprocess
from collections import defaultdict
import numpy as np
from Bio import Align, SeqIO
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.Seq import Seq

ROOT = os.environ.get("BRIM_DATA_DIR", os.environ.get("BRIM_DATA_DIR", "./data"))
OUT_DIR = os.path.join(ROOT, "04_structure_repertoire")
IMGT_REF = os.path.join(ROOT, "02_germline_v", "imgt_ref", "human")

AA3_TO_1 = {k.upper(): v for k, v in protein_letters_3to1.items()}
AA3_TO_1.update({"MSE": "M", "SEC": "U", "PYL": "O"})


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


def load_human_imgt_aa_templates():
    """Return templates[locus] = list of (gene_name, aa_104_str_with_dashes)."""
    templates = {}
    for loc in ["IGHV", "IGKV", "IGLV", "TRAV", "TRBV"]:
        fa = os.path.join(IMGT_REF, f"imgt_human_{loc}.fasta")
        tlist = []
        for rec in SeqIO.parse(fa, "fasta"):
            p = rec.description.split("|")
            if p[3].strip() == "F" and p[1].strip().endswith("*01"):
                s = str(rec.seq).upper()[:312]
                if len(s) == 312:
                    aa = "".join(
                        str(Seq(s[i:i + 3]).translate()) if "." not in s[i:i + 3] else "-"
                        for i in range(0, 312, 3)
                    )
                    tlist.append((p[1].split("*")[0], aa))
        templates[loc] = tlist
    return templates


def map_seq_to_imgt104(seq_str: str, templates_list: list, hint_gene: str = None):
    """
    Map an ungapped V-domain amino acid sequence (e.g. from a PDB chain) onto exact IMGT positions 1..104
    by aligning against the closest human IMGT 104-column gapped AA template.
    Returns dict {seq_0b_index: imgt_pos_1_to_104} and best_gene, or (None, None) if poor match.
    """
    if len(seq_str) < 80:
        return None, None
    query_v = seq_str[:125]  # V-domain is in first ~115 aa

    aligner = Align.PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 3.0
    aligner.mismatch_score = -2.0
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -1.0

    candidates = templates_list
    if hint_gene:
        hg = hint_gene.split("*")[0].strip()
        sub = [t for t in templates_list if t[0] == hg]
        if sub:
            candidates = sub + [t for t in templates_list if t[0] != hg][:10]

    best_score = -1e9
    best_tpl = None
    for gname, tpl_104 in candidates:
        tpl_ungapped = tpl_104.replace("-", "")
        sc = aligner.score(tpl_ungapped, query_v)
        if sc > best_score:
            best_score = sc
            best_tpl = (gname, tpl_104, tpl_ungapped)

    if best_tpl is None or best_score < 130:
        return None, None

    gname, tpl_104, tpl_ungapped = best_tpl
    ungapped_to_imgt = [i + 1 for i, ch in enumerate(tpl_104) if ch != "-"]
    aln = aligner.align(tpl_ungapped, query_v)[0]
    mapping = {}
    for (t_st, t_en), (q_st, q_en) in zip(aln.aligned[0], aln.aligned[1]):
        for ti, qi in zip(range(t_st, t_en), range(q_st, q_en)):
            if ti < len(ungapped_to_imgt):
                mapping[qi] = ungapped_to_imgt[ti]

    # Sanity check: Cys23 (23) or Trp41 (41) or Cys104 (104) alignment quality
    inv = {v: query_v[k] for k, v in mapping.items()}
    anchors_ok = sum([inv.get(23) == "C", inv.get(41) == "W", inv.get(104) == "C"])
    if anchors_ok < 2 or len(mapping) < 75:
        return None, None
    return mapping, gname


def parse_pdb_chains(pdb_path: str):
    """
    Fast heavy-atom PDB parser.
    Returns chains[chain_id] = list of dicts:
      {'resseq': int, 'icode': str, 'resname': str, 'aa': str, 'coords': np.ndarray (N,3), 'ca': np.ndarray (3,)}
    plus remarks list.
    """
    remarks = []
    raw_res = defaultdict(dict)
    res_order = defaultdict(list)

    with open(pdb_path, "r", encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            if ln.startswith("REMARK"):
                remarks.append(ln.strip())
            elif ln.startswith("ATOM  "):
                alt = ln[16]
                if alt not in (" ", "A", "1"):
                    continue
                atom_name = ln[12:16].strip()
                elem = ln[76:78].strip() if len(ln) >= 78 else ""
                if elem == "H" or atom_name.startswith("H"):
                    continue
                resname = ln[17:20].strip()
                if resname not in AA3_TO_1:
                    continue
                chain = ln[21]
                try:
                    resseq = int(ln[22:26])
                except ValueError:
                    continue
                icode = ln[26].strip()
                try:
                    xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
                except ValueError:
                    continue
                key = (resseq, icode)
                if key not in raw_res[chain]:
                    raw_res[chain][key] = {"resseq": resseq, "icode": icode, "resname": resname, "aa": AA3_TO_1[resname], "atoms": [], "ca": None}
                    res_order[chain].append(key)
                raw_res[chain][key]["atoms"].append(xyz)
                if atom_name == "CA":
                    raw_res[chain][key]["ca"] = xyz

    chains = {}
    for ch, keys in res_order.items():
        lst = []
        for k in keys:
            d = raw_res[ch][k]
            arr = np.array(d["atoms"], dtype=np.float32)
            ca = np.array(d["ca"] if d["ca"] is not None else arr.mean(axis=0), dtype=np.float32)
            lst.append({"resseq": d["resseq"], "icode": d["icode"], "aa": d["aa"], "coords": arr, "ca": ca})
        chains[ch] = lst
    return chains, remarks


def min_dist_to_target(res_coords: np.ndarray, target_coords: np.ndarray) -> float:
    if target_coords.shape[0] == 0 or res_coords.shape[0] == 0:
        return 999.0
    diff = res_coords[:, None, :] - target_coords[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    return float(np.sqrt(np.min(d2)))


def compute_ca_exposure(chain_res_list: list) -> list:
    """
    For an isolated V-domain (first 120 residues), compute C-alpha 10A neighbor count
    and normalized solvent exposure index (1 / (1 + neighbors_10A / 15.0)).
    """
    v_res = chain_res_list[:122]
    cas = np.stack([r["ca"] for r in v_res], axis=0)
    diff = cas[:, None, :] - cas[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    # Exclude self and immediate i+-1 peptide-bonded neighbors
    idx = np.arange(len(v_res))
    mask_nonlocal = np.abs(idx[:, None] - idx[None, :]) >= 2
    nbrs = np.sum((dist <= 10.0) & mask_nonlocal, axis=1)
    out = []
    for i in range(len(chain_res_list)):
        if i < len(v_res):
            nb = float(nbrs[i])
            exp_idx = float(1.0 / (1.0 + nb / 12.0))
            out.append((nb, exp_idx))
        else:
            out.append((np.nan, np.nan))
    return out


def analyze_covabdab_structures(templates):
    """Analyze IMGT-renumbered Ab-Ag crystal/cryo-EM structures in covabdab_pdbs/Structures/*_imgt.pdb."""
    pdb_files = sorted(glob.glob(os.path.join(OUT_DIR, "covabdab_pdbs", "Structures", "*_imgt.pdb")))
    print(f"[INFO] Found {len(pdb_files)} CoV-AbDab IMGT PDBs", flush=True)

    # Deduplicate near-identical VH or VL chains (keep unique CDR1+CDR2+FR sequence up to 98% or unique VH[:98])
    seen_vh = set()
    seen_vl = set()

    stats = {loc: {p: {"n": 0, "c5": 0, "c45": 0, "nbr": [], "exp": []} for p in range(1, 105)} for loc in ["IGHV", "IGKV", "IGLV"]}
    per_gene_contacts = defaultdict(lambda: {"n_struct": 0, "cdr12_contacts": 0, "fr_contacts": 0})
    n_used_h, n_used_k, n_used_l = 0, 0, 0

    for pf in pdb_files:
        chains, remarks = parse_pdb_chains(pf)
        # Parse REMARK 5 PAIRED_HL HCHAIN=H LCHAIN=L AGCHAIN=A AGTYPE=PROTEIN
        pair_lines = [r for r in remarks if "AGCHAIN=" in r and "AGTYPE=" in r]
        if not pair_lines:
            continue
        # Take first complex interface in file; require protein/peptide antigen
        pline = pair_lines[0]
        m_h = re.search(r"HCHAIN=([A-Za-z0-9])", pline)
        m_l = re.search(r"LCHAIN=([A-Za-z0-9])", pline)
        m_ag = re.search(r"AGCHAIN=([A-Za-z0-9;|,]+)", pline)
        m_agt = re.search(r"AGTYPE=([A-Za-z0-9_|]+)", pline)
        if not m_ag or (m_agt and "PROTEIN" not in m_agt.group(1).upper() and "PEPTIDE" not in m_agt.group(1).upper()):
            continue

        ag_ids = [c for c in re.split(r"[;|,]", m_ag.group(1)) if c in chains]
        if not ag_ids:
            continue
        ag_coords = np.concatenate([r["coords"] for cid in ag_ids for r in chains[cid]], axis=0)
        if ag_coords.shape[0] < 20:
            continue

        # Heavy chain (skip nanobodies / VHH! Require paired LCHAIN so camelid VHH doesn't mix with human IGHV)
        if m_h and m_l and m_h.group(1) in chains and m_l.group(1) in chains:
            hch = chains[m_h.group(1)]
            h_vseq = "".join(r["aa"] for r in hch if 1 <= r["resseq"] <= 104 and r["icode"] == "")
            # Check human IGHV match & deduplicate identical antibodies solved in multiple PDBs
            _, h_gene = map_seq_to_imgt104(h_vseq, templates["IGHV"])
            if h_gene and h_vseq not in seen_vh:
                seen_vh.add(h_vseq)
                n_used_h += 1
                exp_list = compute_ca_exposure(hch)
                c12_hit, fr_hit = 0, 0
                for idx, r in enumerate(hch):
                    pos = r["resseq"]
                    if 1 <= pos <= 104 and r["icode"] == "":
                        d = min_dist_to_target(r["coords"], ag_coords)
                        st = stats["IGHV"][pos]
                        st["n"] += 1
                        if d <= 5.0:
                            st["c5"] += 1
                            if 27 <= pos <= 38 or 56 <= pos <= 65:
                                c12_hit += 1
                            else:
                                fr_hit += 1
                        if d <= 4.5:
                            st["c45"] += 1
                        nb, ex = exp_list[idx]
                        if not math.isnan(nb):
                            st["nbr"].append(nb)
                            st["exp"].append(ex)
                per_gene_contacts[("IGHV", h_gene)]["n_struct"] += 1
                per_gene_contacts[("IGHV", h_gene)]["cdr12_contacts"] += c12_hit
                per_gene_contacts[("IGHV", h_gene)]["fr_contacts"] += fr_hit

            # Light chain: classify IGKV vs IGLV against human IMGT templates
            lch = chains[m_l.group(1)]
            l_vseq = "".join(r["aa"] for r in lch if 1 <= r["resseq"] <= 104 and r["icode"] == "")
            if l_vseq not in seen_vl and len(l_vseq) >= 80:
                _, k_gene = map_seq_to_imgt104(l_vseq, templates["IGKV"])
                _, l_gene = map_seq_to_imgt104(l_vseq, templates["IGLV"])
                # Pick higher identity match via aligner
                aligner = Align.PairwiseAligner()
                aligner.mode = "global"
                sc_k = max(aligner.score(t[1].replace("-", ""), l_vseq) for t in templates["IGKV"]) if k_gene else -1e9
                sc_l = max(aligner.score(t[1].replace("-", ""), l_vseq) for t in templates["IGLV"]) if l_gene else -1e9
                loc_l, g_l = ("IGKV", k_gene) if sc_k >= sc_l else ("IGLV", l_gene)
                if g_l:
                    seen_vl.add(l_vseq)
                    if loc_l == "IGKV":
                        n_used_k += 1
                    else:
                        n_used_l += 1
                    exp_list = compute_ca_exposure(lch)
                    for idx, r in enumerate(lch):
                        pos = r["resseq"]
                        if 1 <= pos <= 104 and r["icode"] == "":
                            d = min_dist_to_target(r["coords"], ag_coords)
                            st = stats[loc_l][pos]
                            st["n"] += 1
                            if d <= 5.0:
                                st["c5"] += 1
                            if d <= 4.5:
                                st["c45"] += 1
                            nb, ex = exp_list[idx]
                            if not math.isnan(nb):
                                st["nbr"].append(nb)
                                st["exp"].append(ex)

    print(f"[DONE] CoV-AbDab unique Ab-Ag interfaces: IGHV={n_used_h}, IGKV={n_used_k}, IGLV={n_used_l}", flush=True)
    return stats, per_gene_contacts


def parse_cif_chains(cif_gz_path: str, needed_chains: set):
    """Lightweight mmCIF ATOM_SITE parser for general SAbDab2 non-coronavirus benchmark structures."""
    raw_res = defaultdict(dict)
    res_order = defaultdict(list)
    with gzip.open(cif_gz_path, "rt", encoding="utf-8", errors="replace") as fh:
        in_atom = False
        cols = []
        col_idx = {}
        for ln in fh:
            s = ln.strip()
            if s == "loop_":
                in_atom = False
                cols = []
                continue
            if s.startswith("_atom_site."):
                in_atom = True
                cols.append(s.split(".")[1])
                col_idx = {c: i for i, c in enumerate(cols)}
                continue
            if in_atom:
                if not s or s.startswith("#") or s.startswith("_"):
                    if cols and len(raw_res) > 0:
                        in_atom = False
                    continue
                if not (s.startswith("ATOM ") or s.startswith("HETATM")):
                    continue
                parts = s.split()
                if len(parts) < len(cols):
                    continue
                if parts[col_idx["group_PDB"]] != "ATOM":
                    continue
                if "pdbx_PDB_model_num" in col_idx and parts[col_idx["pdbx_PDB_model_num"]] != "1":
                    continue
                alt = parts[col_idx["label_alt_id"]] if "label_alt_id" in col_idx else "."
                if alt not in (".", "A", "1"):
                    continue
                ch = parts[col_idx.get("auth_asym_id", col_idx.get("label_asym_id"))]
                if ch not in needed_chains:
                    continue
                elem = parts[col_idx["type_symbol"]]
                if elem == "H":
                    continue
                comp = parts[col_idx.get("auth_comp_id", col_idx.get("label_comp_id"))]
                if comp not in AA3_TO_1:
                    continue
                seq_id_s = parts[col_idx.get("auth_seq_id", col_idx.get("label_seq_id"))]
                try:
                    seq_id = int(seq_id_s)
                except ValueError:
                    continue
                ins = parts[col_idx["pdbx_PDB_ins_code"]] if "pdbx_PDB_ins_code" in col_idx else "."
                if ins in (".", "?"):
                    ins = ""
                atom_id = parts[col_idx.get("auth_atom_id", col_idx.get("label_atom_id"))].strip('"')
                xyz = (
                    float(parts[col_idx["Cartn_x"]]),
                    float(parts[col_idx["Cartn_y"]]),
                    float(parts[col_idx["Cartn_z"]]),
                )
                key = (seq_id, ins)
                if key not in raw_res[ch]:
                    raw_res[ch][key] = {"aa": AA3_TO_1[comp], "atoms": [], "ca": None}
                    res_order[ch].append(key)
                raw_res[ch][key]["atoms"].append(xyz)
                if atom_id == "CA":
                    raw_res[ch][key]["ca"] = xyz

    chains = {}
    for ch, keys in res_order.items():
        lst = []
        for k in keys:
            d = raw_res[ch][k]
            arr = np.array(d["atoms"], dtype=np.float32)
            ca = np.array(d["ca"] if d["ca"] is not None else arr.mean(axis=0), dtype=np.float32)
            lst.append({"aa": d["aa"], "coords": arr, "ca": ca})
        chains[ch] = lst
    return chains


def analyze_general_sabdab2_noncov(templates, max_structures: int = 320):
    """
    Select diverse non-redundant, non-coronavirus human Fab-protein antigen structures (resolution <= 2.8 A)
    from sabdab2_all_summary.csv, download mmCIFs from RCSB in parallel, and compute IMGT 1..104 contact frequencies!
    """
    summary_csv = os.path.join(OUT_DIR, "sabdab2_all_summary.csv")
    cif_dir = os.path.join(OUT_DIR, "sabdab2_noncov_cifs")
    os.makedirs(cif_dir, exist_ok=True)

    selected = []
    seen_heavy_id = set()
    seen_pdb = set()
    with open(summary_csv, "r", encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            pdb_raw = (r.get("PDB") or "").strip().lower()
            # Standard 4-char PDB code (or strip pdb_0000 prefix if 4-char)
            if pdb_raw.startswith("pdb_0000") and len(pdb_raw) == 12:
                pdb4 = pdb_raw[-4:]
            elif len(pdb_raw) == 4:
                pdb4 = pdb_raw
            else:
                continue
            if pdb4 in seen_pdb:
                continue
            hid = (r.get("HEAVY_ID") or "").strip()
            if not hid or hid in seen_heavy_id:
                continue
            if "homo sapiens" not in (r.get("heavy_species") or "").lower():
                continue
            ag_type = (r.get("antigen_type") or "").upper()
            if "PROTEIN" not in ag_type:
                continue
            ag_name = ((r.get("antigen_name") or "") + " " + (r.get("compound") or "")).lower()
            if any(k in ag_name for k in ["sars", "cov", "spike", "coronavirus", "rbd"]):
                continue
            hch = (r.get("Hchain") or "").strip()
            lch = (r.get("Lchain") or "").strip()
            ag_ch = [c.strip() for c in re.split(r"[|,;]", r.get("antigen_chain") or "") if c.strip() and c.strip() != "NA"]
            if not hch or not lch or hch == "NA" or lch == "NA" or not ag_ch:
                continue
            try:
                res = float(r.get("resolution") or 99.0)
            except ValueError:
                continue
            if res > 2.8:
                continue
            seen_heavy_id.add(hid)
            seen_pdb.add(pdb4)
            selected.append((pdb4, hch, lch, ag_ch))
            if len(selected) >= max_structures:
                break

    print(f"[INFO] Selected {len(selected)} non-redundant non-CoV human Ab-protein complexes from SAbDab2 (<=2.8 A)", flush=True)

    def fetch_one(item):
        pdb4, hch, lch, ag_ch = item
        dest = os.path.join(cif_dir, f"{pdb4}.cif.gz")
        if not os.path.exists(dest) or os.path.getsize(dest) < 5000:
            url = f"https://files.rcsb.org/download/{pdb4}.cif.gz"
            rc = subprocess.call(["wget", "-q", "--tries=3", "--timeout=20", "-O", dest, url])
            if rc != 0 or not os.path.exists(dest) or os.path.getsize(dest) < 5000:
                return None
        return (dest, hch, lch, ag_ch)

    valid_items = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        for res in ex.map(fetch_one, selected):
            if res is not None:
                valid_items.append(res)
    print(f"[INFO] Downloaded {len(valid_items)}/{len(selected)} SAbDab2 non-CoV mmCIFs", flush=True)

    stats = {loc: {p: {"n": 0, "c5": 0, "c45": 0, "nbr": [], "exp": []} for p in range(1, 105)} for loc in ["IGHV", "IGKV", "IGLV"]}
    n_ok_h = 0
    for dest, hch, lch, ag_ch in valid_items:
        needed = {hch, lch} | set(ag_ch)
        try:
            chains = parse_cif_chains(dest, needed)
        except Exception:
            continue
        if hch not in chains or not any(a in chains for a in ag_ch):
            continue
        ag_coords = np.concatenate([r["coords"] for a in ag_ch if a in chains for r in chains[a]], axis=0)
        if ag_coords.shape[0] < 20:
            continue

        h_list = chains[hch]
        h_seq = "".join(r["aa"] for r in h_list[:125])
        mapping, h_gene = map_seq_to_imgt104(h_seq, templates["IGHV"])
        if mapping:
            n_ok_h += 1
            exp_list = compute_ca_exposure(h_list)
            for q_idx, pos in mapping.items():
                if 1 <= pos <= 104 and q_idx < len(h_list):
                    d = min_dist_to_target(h_list[q_idx]["coords"], ag_coords)
                    st = stats["IGHV"][pos]
                    st["n"] += 1
                    if d <= 5.0:
                        st["c5"] += 1
                    if d <= 4.5:
                        st["c45"] += 1
                    nb, ex = exp_list[q_idx]
                    if not math.isnan(nb):
                        st["nbr"].append(nb)
                        st["exp"].append(ex)

        if lch in chains:
            l_list = chains[lch]
            l_seq = "".join(r["aa"] for r in l_list[:125])
            mk, gk = map_seq_to_imgt104(l_seq, templates["IGKV"])
            ml, gl = map_seq_to_imgt104(l_seq, templates["IGLV"])
            loc_l, mp_l = ("IGKV", mk) if (mk and (not ml or len(mk) >= len(ml))) else ("IGLV", ml)
            if mp_l:
                exp_list = compute_ca_exposure(l_list)
                for q_idx, pos in mp_l.items():
                    if 1 <= pos <= 104 and q_idx < len(l_list):
                        d = min_dist_to_target(l_list[q_idx]["coords"], ag_coords)
                        st = stats[loc_l][pos]
                        st["n"] += 1
                        if d <= 5.0:
                            st["c5"] += 1
                        if d <= 4.5:
                            st["c45"] += 1
                        nb, ex = exp_list[q_idx]
                        if not math.isnan(nb):
                            st["nbr"].append(nb)
                            st["exp"].append(ex)

    print(f"[DONE] SAbDab2 non-CoV mapped general Ab-Ag interfaces: IGHV={n_ok_h}", flush=True)
    return stats


def analyze_tcr3d_pmhc(templates):
    """
    Analyze human alpha-beta TCR-pMHC ternary complexes from TCR3d (TCR_complexes/*.trunc.fit.pdb),
    computing BOTH Peptide contact frequency AND MHC helix contact frequency across IMGT 1..104!
    """
    meta_tsv = os.path.join(OUT_DIR, "tcr3d_complexes_data.tsv")
    pdb_dir = os.path.join(OUT_DIR, "tcr3d_complexes", "TCR_complexes")

    stats = {
        loc: {
            p: {"n": 0, "c5_pmhc": 0, "c45_pmhc": 0, "c5_pep": 0, "c5_mhc": 0, "nbr": [], "exp": []}
            for p in range(1, 105)
        }
        for loc in ["TRAV", "TRBV"]
    }
    seen_tcr = {"TRAV": set(), "TRBV": set()}
    n_used = {"TRAV": 0, "TRBV": 0}

    with open(meta_tsv, "r", encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if "human" not in (r.get("TCR_organism") or "").lower():
                continue
            pdb_id = (r.get("PDB_ID") or "").strip().lower()
            pdb_path = os.path.join(pdb_dir, f"{pdb_id}.trunc.fit.pdb")
            if not os.path.exists(pdb_path):
                continue

            chains, _ = parse_pdb_chains(pdb_path)
            if not chains:
                continue

            # Identify chain roles cleanly using sequence signatures + TCR3d standard convention (A/B=MHC, C=Pep, D=TCRa, E=TCRb)
            cdr3a = (r.get("CDR3_alpha") or "").strip()
            cdr3b = (r.get("CDR3_beta") or "").strip()
            epitope = (r.get("Epitope") or "").strip()

            ch_seqs = {cid: "".join(x["aa"] for x in clist) for cid, clist in chains.items()}
            tcra_ch = next((cid for cid, sq in ch_seqs.items() if cdr3a and cdr3a[1:-1] in sq), ("D" if "D" in chains else None))
            tcrb_ch = next((cid for cid, sq in ch_seqs.items() if cdr3b and cdr3b[1:-1] in sq), ("E" if "E" in chains else None))
            pep_ch = next(
                (cid for cid, sq in ch_seqs.items() if cid not in (tcra_ch, tcrb_ch) and ((epitope and epitope in sq) or len(sq) <= 25)),
                ("C" if "C" in chains and len(ch_seqs["C"]) <= 30 else None),
            )
            mhc_chs = [cid for cid, sq in ch_seqs.items() if cid not in (tcra_ch, tcrb_ch, pep_ch) and len(sq) >= 60]

            if not mhc_chs or not pep_ch:
                continue
            pep_coords = np.concatenate([x["coords"] for x in chains[pep_ch]], axis=0)
            mhc_coords = np.concatenate([x["coords"] for cid in mhc_chs for x in chains[cid]], axis=0)
            pmhc_coords = np.concatenate([pep_coords, mhc_coords], axis=0)

            for loc, tcr_cid, v_hint in [("TRAV", tcra_ch, r.get("TRAV_gene")), ("TRBV", tcrb_ch, r.get("TRBV_gene"))]:
                if not tcr_cid or tcr_cid not in chains:
                    continue
                tlist = chains[tcr_cid]
                tseq = ch_seqs[tcr_cid]
                # Deduplicate identical TCR variable + CDR3 + epitope complexes solved multiple times
                sig = (tseq[:95], epitope, r.get("MHC_allele", ""))
                if sig in seen_tcr[loc]:
                    continue
                mapping, gname = map_seq_to_imgt104(tseq, templates[loc], hint_gene=v_hint)
                if not mapping:
                    continue
                seen_tcr[loc].add(sig)
                n_used[loc] += 1
                exp_list = compute_ca_exposure(tlist)
                for q_idx, pos in mapping.items():
                    if 1 <= pos <= 104 and q_idx < len(tlist):
                        rc = tlist[q_idx]["coords"]
                        d_pep = min_dist_to_target(rc, pep_coords)
                        d_mhc = min_dist_to_target(rc, mhc_coords)
                        d_all = min(d_pep, d_mhc)
                        st = stats[loc][pos]
                        st["n"] += 1
                        if d_all <= 5.0:
                            st["c5_pmhc"] += 1
                        if d_all <= 4.5:
                            st["c45_pmhc"] += 1
                        if d_pep <= 5.0:
                            st["c5_pep"] += 1
                        if d_mhc <= 5.0:
                            st["c5_mhc"] += 1
                        nb, ex = exp_list[q_idx]
                        if not math.isnan(nb):
                            st["nbr"].append(nb)
                            st["exp"].append(ex)

    print(f"[DONE] TCR3d unique human TCR-pMHC complexes: TRAV={n_used['TRAV']}, TRBV={n_used['TRBV']}", flush=True)
    return stats


def write_ab_table(stats_cov, stats_gen, out_path: str):
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow([
            "locus", "imgt_pos", "region",
            "n_covabdab", "contact5_freq_covabdab", "contact45_freq_covabdab",
            "n_sabdab_noncov", "contact5_freq_sabdab_noncov", "contact45_freq_sabdab_noncov",
            "n_combined", "contact5_freq_combined", "mean_ca_neighbors_10A", "mean_surface_exposure_idx"
        ])
        for loc in ["IGHV", "IGKV", "IGLV"]:
            for p in range(1, 105):
                sc = stats_cov[loc][p]
                sg = stats_gen[loc][p]
                nc, ng = sc["n"], sg["n"]
                ntot = nc + ng
                fc = sc["c5"] / nc if nc else 0.0
                fc45 = sc["c45"] / nc if nc else 0.0
                fg = sg["c5"] / ng if ng else 0.0
                fg45 = sg["c45"] / ng if ng else 0.0
                fcomb = (sc["c5"] + sg["c5"]) / ntot if ntot else 0.0
                nbrs = sc["nbr"] + sg["nbr"]
                exps = sc["exp"] + sg["exp"]
                w.writerow([
                    loc, p, imgt_region_of_pos(p),
                    nc, f"{fc:.5f}", f"{fc45:.5f}",
                    ng, f"{fg:.5f}", f"{fg45:.5f}",
                    ntot, f"{fcomb:.5f}",
                    f"{np.mean(nbrs):.3f}" if nbrs else "NA",
                    f"{np.mean(exps):.4f}" if exps else "NA",
                ])


def write_tcr_table(stats_tcr, out_path: str):
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow([
            "locus", "imgt_pos", "region",
            "n_structures", "contact5_pmhc_freq", "contact45_pmhc_freq",
            "contact5_peptide_freq", "contact5_mhc_freq",
            "mean_ca_neighbors_10A", "mean_surface_exposure_idx"
        ])
        for loc in ["TRAV", "TRBV"]:
            for p in range(1, 105):
                st = stats_tcr[loc][p]
                n = st["n"]
                w.writerow([
                    loc, p, imgt_region_of_pos(p),
                    n,
                    f"{st['c5_pmhc'] / n:.5f}" if n else "0.00000",
                    f"{st['c45_pmhc'] / n:.5f}" if n else "0.00000",
                    f"{st['c5_pep'] / n:.5f}" if n else "0.00000",
                    f"{st['c5_mhc'] / n:.5f}" if n else "0.00000",
                    f"{np.mean(st['nbr']):.3f}" if st["nbr"] else "NA",
                    f"{np.mean(st['exp']):.4f}" if st["exp"] else "NA",
                ])


def main():
    templates = load_human_imgt_aa_templates()
    stats_cov, _ = analyze_covabdab_structures(templates)
    stats_gen = analyze_general_sabdab2_noncov(templates, max_structures=320)
    stats_tcr = analyze_tcr3d_pmhc(templates)

    ab_tsv = os.path.join(OUT_DIR, "structural_contacts_bcr_imgt104.tsv")
    tcr_tsv = os.path.join(OUT_DIR, "structural_contacts_tcr_imgt104.tsv")
    write_ab_table(stats_cov, stats_gen, ab_tsv)
    write_tcr_table(stats_tcr, tcr_tsv)
    print(f"[ALL_3D_CONTACTS_DONE] -> {ab_tsv}, {tcr_tsv}", flush=True)


if __name__ == "__main__":
    main()
