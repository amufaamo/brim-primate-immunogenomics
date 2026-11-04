#!/usr/bin/env python3
"""
07_plot_supplementary_figures.py (v2 Revision)
Publication-Quality Supplementary Figures S1–S4 for Primate T2T Immunogenomics
Reconciled against 3AI Peer Review Critiques:
- Fig S1: Reconciled counts (1,588 F / 3,671 total), 75 strict 1:1 orthologs gene-level ratio (1.22x, p=1.26e-4)
- Fig S2: Gene conversion partitioned trees audit (all 7 units p < 0.03)
- Fig S3: TCR diagonal docking structural decomposition (peptide vs MHC helix footprints)
- Fig S4: Archetypes: IGHV4-34 tolerance dynamics, IGHV1-69 Kabat 54 F/L/V diversification, public TCRs
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy import stats as sp_stats

DRIVE_DIR = os.environ.get("BRIM_DATA_DIR", os.environ.get("BRIM_DATA_DIR", "./data"))
COUNTS_TSV = os.path.join(DRIVE_DIR, "brim_t2t_02_germline_counts.tsv")
K1_TSV = os.path.join(DRIVE_DIR, "brim_t2t_04_k1_summary.tsv")
PART_TSV = os.path.join(DRIVE_DIR, "brim_t2t_03b_partition_sensitivity.tsv")
MASTER_TSV = os.path.join(DRIVE_DIR, "brim_t2t_04_master_imgt104_3d_selection.tsv")
ATLAS_TSV = os.path.join(DRIVE_DIR, "brim_t2t_04_gene_integrated_atlas.tsv")

FIG_S1_PNG = os.path.join(DRIVE_DIR, "brim_t2t_supp_fig_s1.png")
FIG_S1_PDF = os.path.join(DRIVE_DIR, "brim_t2t_supp_fig_s1.pdf")
FIG_S2_PNG = os.path.join(DRIVE_DIR, "brim_t2t_supp_fig_s2.png")
FIG_S2_PDF = os.path.join(DRIVE_DIR, "brim_t2t_supp_fig_s2.pdf")
FIG_S3_PNG = os.path.join(DRIVE_DIR, "brim_t2t_supp_fig_s3.png")
FIG_S3_PDF = os.path.join(DRIVE_DIR, "brim_t2t_supp_fig_s3.pdf")
FIG_S4_PNG = os.path.join(DRIVE_DIR, "brim_t2t_supp_fig_s4.png")
FIG_S4_PDF = os.path.join(DRIVE_DIR, "brim_t2t_supp_fig_s4.pdf")

# Publication style
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["xtick.major.width"] = 0.8
plt.rcParams["ytick.major.width"] = 0.8
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"

LOCUS_COLORS = {
    "IGHV": "#1F618D",  # Dark Blue
    "IGKV": "#E67E22",  # Orange
    "IGLV": "#16A085",  # Teal Green
    "TRAV": "#8E44AD",  # Purple
    "TRBV": "#27AE60"   # Forest Green
}

# =============================================================================
# FIGURE S1
# =============================================================================
def plot_figure_s1():
    print("[RUN] Plotting Supplementary Figure S1 (v2)...", flush=True)
    df_counts = pd.read_csv(COUNTS_TSV, sep="\t")
    df_atlas = pd.read_csv(ATLAS_TSV, sep="\t")

    species_order = [
        ("Human_CHM13", "Human (CHM13)"),
        ("Pan_paniscus", "Bonobo"),
        ("Pan_troglodytes", "Chimpanzee"),
        ("Gorilla_gorilla", "Gorilla"),
        ("Pongo_pygmaeus", "Bornean orangutan"),
        ("Pongo_abelii", "Sumatran orangutan"),
        ("Symphalangus_syndactylus", "Siamang (gibbon)"),
        ("Macaca_mulatta", "Rhesus macaque")
    ]

    fig = plt.figure(figsize=(14, 6), dpi=300)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.5, 0.8, 1.2], wspace=0.35)

    # Panel A: Stacked bar chart of functional genes
    ax_a = fig.add_subplot(gs[0])
    loci = ["IGHV", "IGKV", "IGLV", "TRAV", "TRBV"]
    x = np.arange(len(species_order))
    bottoms = np.zeros(len(species_order))

    for locus in loci:
        counts = []
        for asm_id, _ in species_order:
            row = df_counts[(df_counts["assembly_id"] == asm_id) & (df_counts["locus"] == locus)]
            c = row["n_functional_F"].values[0] if len(row) > 0 else 0
            counts.append(c)
        counts = np.array(counts)
        ax_a.bar(x, counts, bottom=bottoms, label=locus, color=LOCUS_COLORS[locus], edgecolor="white", lw=0.6, alpha=0.92)
        bottoms += counts

    ax_a.set_xticks(x)
    ax_a.set_xticklabels([name for _, name in species_order], rotation=40, ha="right", fontsize=9.5)
    ax_a.set_ylabel("Number of Functional V Genes (Haploid)", fontsize=10.5, fontweight="bold")
    ax_a.set_title("A   Functional Germline V Repertoires Across 8 T2T Primates\n(Reconciled: 1,588 Functional / 3,671 Total Segments)", loc="left", fontsize=11, fontweight="bold", pad=10)
    ax_a.grid(True, axis="y", ls=":", alpha=0.5)
    ax_a.legend(loc="upper right", frameon=True, fontsize=8.5, ncol=2)

    # Annotate Siamang IGK deletion
    siamang_idx = 6
    total_siamang = bottoms[siamang_idx]
    ax_a.annotate("IGK deleted (0)\nIGHV contracted (8)",
                  xy=(siamang_idx, total_siamang), xytext=(siamang_idx - 1.2, total_siamang + 45),
                  arrowprops=dict(facecolor="#C0392B", shrink=0.08, width=1.2, headwidth=5),
                  fontsize=8.5, fontweight="bold", color="#C0392B", ha="center")
    ax_a.set_ylim(0, 310)

    # Panel B: Strict 1:1 Orthologs by Locus
    ax_b = fig.add_subplot(gs[1])
    strict_df = df_atlas[df_atlas["is_strict_1to1"] == 1]
    locus_counts = strict_df["locus"].value_counts().reindex(loci).fillna(0)

    bars_b = ax_b.bar(range(len(loci)), locus_counts.values, color=[LOCUS_COLORS[l] for l in loci],
                      edgecolor="#333333", lw=0.8, alpha=0.9)
    for bar, val in zip(bars_b, locus_counts.values):
        ax_b.text(bar.get_x() + bar.get_width()/2.0, val + 0.8, f"{int(val)}", ha="center", va="bottom",
                  fontsize=9.5, fontweight="bold")
    ax_b.set_xticks(range(len(loci)))
    ax_b.set_xticklabels(loci, fontsize=10, fontweight="bold")
    ax_b.set_ylabel("Number of Strict 1:1 Orthologs", fontsize=10.5, fontweight="bold")
    ax_b.set_title("B   Strict 1:1 Syntenic Orthologs\n(n = 75 paralog-free)", loc="left", fontsize=11, fontweight="bold", pad=10)
    ax_b.set_ylim(0, 38)
    ax_b.grid(True, axis="y", ls=":", alpha=0.5)

    # Panel C: Strict 1:1 Orthologs CDR vs FR Omega Ratio
    ax_c = fig.add_subplot(gs[2])
    data_by_locus = [strict_df[strict_df["locus"] == l]["cdr12_vs_fr_omega_ratio"].dropna().values for l in ["IGHV", "TRAV", "TRBV"]]
    data_all = strict_df["cdr12_vs_fr_omega_ratio"].dropna().values

    all_data = [data_all] + data_by_locus
    labels = ["All 1:1\n(n=75)", "IGHV\n(n=5)", "TRAV\n(n=31)", "TRBV\n(n=31)"]
    colors = ["#34495E", LOCUS_COLORS["IGHV"], LOCUS_COLORS["TRAV"], LOCUS_COLORS["TRBV"]]

    bp = ax_c.boxplot(all_data, tick_labels=labels, patch_artist=True, widths=0.55,
                     medianprops=dict(color="#C0392B", lw=2.0), showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.3)
        patch.set_edgecolor(c)
        patch.set_linewidth(1.2)

    np.random.seed(42)
    for i, (vals, c) in enumerate(zip(all_data, colors)):
        jitter = np.random.normal(0, 0.06, size=len(vals))
        ax_c.scatter(np.full_like(vals, i + 1) + jitter, vals, color=c, alpha=0.75, s=24, edgecolor="white", lw=0.4)

    ax_c.axhline(1.0, color="#7F8C8D", ls="--", lw=1.2, zorder=0)

    stat_w, p_w = sp_stats.wilcoxon(data_all - 1.0)
    ax_c.text(0.96, 0.94, f"All 1:1 Orthologs:\nGene Median Ratio = {np.median(data_all):.2f}x\nWilcoxon p = {p_w:.2e}",
              transform=ax_c.transAxes, ha="right", va="top", fontsize=9, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.4", fc="#FFFFFF", ec="#BDC3C7", lw=1.0, alpha=0.9))

    ax_c.set_ylabel(r"Paratope Diversification Ratio ($\omega_{\mathrm{CDR}} / \omega_{\mathrm{FR}}$)", fontsize=10.5, fontweight="bold")
    ax_c.set_title("C   Paratope Acceleration in 1:1 Orthologs", loc="left", fontsize=11, fontweight="bold", pad=10)
    ax_c.grid(True, axis="y", ls=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIG_S1_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(FIG_S1_PDF, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved Figure S1 to {FIG_S1_PNG}", flush=True)

# =============================================================================
# FIGURE S2
# =============================================================================
def plot_figure_s2():
    print("[RUN] Plotting Supplementary Figure S2 (v2)...", flush=True)
    df_k1 = pd.read_csv(K1_TSV, sep="\t")
    df_part = pd.read_csv(PART_TSV, sep="\t")

    merged = pd.merge(df_k1, df_part, left_on="dataset", right_on="unit")
    units = merged["dataset"].tolist()
    labels = [u.replace("_All", "").replace("_MetaClans", " (Meta)").replace("_Clan", " Clan ") for u in units]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    x = np.arange(len(units))
    w = 0.2

    ax1.bar(x - 1.5*w, merged["median_omega_cdr12"], width=w, label="Standard Tree CDR1+2",
            color="#C0392B", alpha=0.85, edgecolor="#7B241C", lw=0.8)
    ax1.bar(x - 0.5*w, merged["median_omega_core_fr"], width=w, label="Standard Tree Core FR",
            color="#2980B9", alpha=0.85, edgecolor="#1B4F72", lw=0.8)

    ax1.bar(x + 0.5*w, merged["median_omega_cdr12_part_tree"], width=w, label="5'/3' Partitioned CDR1+2",
            color="#E74C3C", alpha=0.55, edgecolor="#7B241C", lw=0.8, hatch="//")
    ax1.bar(x + 1.5*w, merged["median_omega_core_fr_part_tree"], width=w, label="5'/3' Partitioned Core FR",
            color="#3498DB", alpha=0.55, edgecolor="#1B4F72", lw=0.8, hatch="//")

    ax1.axhline(1.0, color="#7F8C8D", ls="--", lw=1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=9.5, fontweight="bold")
    ax1.set_ylabel(r"Median Selection Parameter ($\omega = d_N/d_S$)", fontsize=10.5, fontweight="bold")
    ax1.set_title("A   Codon Selection Rates Across Phylogenetic Partition Regimes", loc="left", fontsize=11, fontweight="bold", pad=10)
    ax1.grid(True, axis="y", ls=":", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True, fontsize=8.5, ncol=2)
    ax1.set_ylim(0, 1.35)

    r_std = merged["omega_median_ratio_cdr_vs_fr"]
    r_part = merged["ratio_cdr_vs_fr_part_tree"]

    ax2.plot(x, r_std, marker="o", markersize=8, color="#8E44AD", lw=2.0, label="Standard Single Tree")
    ax2.plot(x, r_part, marker="s", markersize=8, color="#16A085", lw=2.0, ls="--", label="5'/3' Partitioned Trees")

    for i in range(len(units)):
        ax2.annotate(f"{r_part[i]:.2f}x", (x[i], r_part[i]), xytext=(x[i], r_part[i] + 0.12),
                     ha="center", fontsize=8.5, fontweight="bold", color="#16A085")

    ax2.axhline(1.0, color="#C0392B", ls=":", lw=1.2, label="Paratope Acceleration Threshold (1.0x)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=9.5, fontweight="bold")
    ax2.set_ylabel(r"Acceleration Ratio ($\omega_{\mathrm{CDR}} / \omega_{\mathrm{FR}}$)", fontsize=10.5, fontweight="bold")
    ax2.set_title("B   Robustness of Paratope Acceleration Against Interlocus Gene Conversion\n(All 7 units retain significant acceleration under partitioned trees, p < 0.03)",
                  loc="left", fontsize=10.5, fontweight="bold", pad=10)
    ax2.grid(True, ls=":", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True, fontsize=9)
    ax2.set_ylim(0.8, 3.4)

    plt.tight_layout()
    plt.savefig(FIG_S2_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(FIG_S2_PDF, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved Figure S2 to {FIG_S2_PNG}", flush=True)

# =============================================================================
# FIGURE S3
# =============================================================================
def plot_figure_s3():
    print("[RUN] Plotting Supplementary Figure S3 (v2)...", flush=True)
    df_m = pd.read_csv(MASTER_TSV, sep="\t")

    trbv = df_m[df_m["eval_unit"] == "TRBV_All"].sort_values("imgt_pos").copy()
    trav = df_m[df_m["eval_unit"] == "TRAV_All"].sort_values("imgt_pos").copy()

    trbv["diff_beta_alpha"] = trbv["fubar_beta"] - trbv["fubar_alpha"]
    trav["diff_beta_alpha"] = trav["fubar_beta"] - trav["fubar_alpha"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), dpi=300, sharex=True)

    # Panel A: TRBV
    color_bar = "#3498DB"
    color_line = "#C0392B"

    ax1.bar(trbv["imgt_pos"], trbv["contact5_peptide_freq"], width=0.85, color=color_bar, alpha=0.6,
            edgecolor="#2980B9", lw=0.5, label="Empirical Peptide Contact Frequency (<5.0 Å)")
    ax1.set_ylabel("Peptide Contact Frequency\n(n = 260 TRBV chains)", color="#2980B9", fontsize=10, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#2980B9")
    ax1.set_ylim(0, 0.8)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(trbv["imgt_pos"], trbv["diff_beta_alpha"], color=color_line, lw=1.8, marker="o", markersize=3.5,
                  label=r"Evolutionary Diversification (FUBAR $\beta - \alpha$)")
    ax1_twin.axhline(0, color="#7F8C8D", ls="--", lw=0.8)
    ax1_twin.set_ylabel(r"Diversification Parameter ($\beta - \alpha$)", color=color_line, fontsize=10, fontweight="bold")
    ax1_twin.tick_params(axis="y", labelcolor=color_line)
    ax1_twin.set_ylim(-2.5, 3.5)

    ax1.axvspan(27, 38, color="#FADBD8", alpha=0.3, label="CDR1 (IMGT 27–38)")
    ax1.axvspan(56, 65, color="#F5B7B1", alpha=0.4, label="CDR2 (IMGT 56–65)")
    ax1.annotate(r"TRBV CDR2 Peptide Contact Footprint" "\n" r"(Partial $\rho = 0.320$, $p = 0.0019$)",
                 xy=(58, 0.35), xytext=(66, 0.60),
                 arrowprops=dict(facecolor="#C0392B", shrink=0.08, width=1.2, headwidth=5),
                 fontsize=9.5, fontweight="bold", color="#922B21")

    ax1.set_title("A   TRBV (TCRβ): Positive Selection Tracks the Antigenic Peptide Contact Interface",
                  loc="left", fontsize=11, fontweight="bold", pad=8)
    ax1.grid(True, axis="x", ls=":", alpha=0.5)

    # Panel B: TRAV
    color_bar_trav = "#E67E22"
    ax2.bar(trav["imgt_pos"], trav["contact5_mhc_freq"], width=0.85, color=color_bar_trav, alpha=0.6,
            edgecolor="#D35400", lw=0.5, label="Empirical MHC Contact Frequency (<5.0 Å)")
    ax2.set_ylabel("MHC Contact Frequency\n(n = 256 TRAV chains)", color="#D35400", fontsize=10, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#D35400")
    ax2.set_ylim(0, 1.05)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(trav["imgt_pos"], trav["diff_beta_alpha"], color=color_line, lw=1.8, marker="s", markersize=3.5,
                  label=r"Evolutionary Diversification (FUBAR $\beta - \alpha$)")
    ax2_twin.axhline(0, color="#7F8C8D", ls="--", lw=0.8)
    ax2_twin.set_ylabel(r"Diversification Parameter ($\beta - \alpha$)", color=color_line, fontsize=10, fontweight="bold")
    ax2_twin.tick_params(axis="y", labelcolor=color_line)
    ax2_twin.set_ylim(-2.5, 3.5)

    ax2.axvspan(27, 38, color="#FCF3CF", alpha=0.4, label="CDR1 (IMGT 27–38)")
    ax2.axvspan(56, 65, color="#FDEBD0", alpha=0.4, label="CDR2 (IMGT 56–65)")
    ax2.annotate(r"TRAV CDR1 MHC Contact Footprint (IMGT 30–36)" "\n" r"(Spearman $\rho = 0.285$, $p = 0.0059$)",
                 xy=(30, 0.90), xytext=(40, 0.85),
                 arrowprops=dict(facecolor="#D35400", shrink=0.08, width=1.2, headwidth=5),
                 fontsize=9.5, fontweight="bold", color="#A04000")

    ax2.set_title("B   TRAV (TCRα): Evolutionary Diversification Tracks the Conserved MHC Interaction Surface",
                  loc="left", fontsize=11, fontweight="bold", pad=8)
    ax2.set_xlabel("IMGT Unique Codon Position (1 .. 104)", fontsize=11, fontweight="bold")
    ax2.set_xlim(0, 105)
    ax2.grid(True, axis="x", ls=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIG_S3_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(FIG_S3_PDF, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved Figure S3 to {FIG_S3_PNG}", flush=True)

# =============================================================================
# FIGURE S4
# =============================================================================
def plot_figure_s4():
    print("[RUN] Plotting Supplementary Figure S4 (v2)...", flush=True)
    df_atlas = pd.read_csv(ATLAS_TSV, sep="\t")
    df_m = pd.read_csv(MASTER_TSV, sep="\t")

    fig = plt.figure(figsize=(15, 5.5), dpi=300)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.2, 1.2], wspace=0.35)

    # Panel A: IGHV4-34 Peripheral Regulation
    ax_a = fig.add_subplot(gs[0])
    row_434 = df_atlas[df_atlas["human_gene"] == "IGHV4-34"].iloc[0]

    stages = ["Healthy\nNaive B", "Healthy\nMemory B", "Myasthenia\nGravis Memory"]
    freqs = [row_434["vane_hd_naive_freq"] * 100,
             row_434["vane_hd_memory_freq"] * 100,
             row_434["vane_mg_memory_freq"] * 100]
    colors = ["#2980B9", "#27AE60", "#C0392B"]

    bars_a = ax_a.bar(stages, freqs, color=colors, edgecolor="#333333", lw=0.8, alpha=0.88, width=0.55)
    for bar, f in zip(bars_a, freqs):
        ax_a.text(bar.get_x() + bar.get_width()/2.0, f + 0.12, f"{f:.2f}%", ha="center", va="bottom",
                  fontsize=9.5, fontweight="bold")

    ax_a.annotate("Peripheral Purge\n" + r"$\log_2 = -0.74$", xy=(0.5, 3.5), xytext=(0.5, 4.3),
                  ha="center", fontsize=8.5, fontweight="bold", color="#7F8C8D",
                  arrowprops=dict(arrowstyle="->", lw=1.2, color="#7F8C8D"))
    ax_a.annotate("Autoimmune Expansion\n" + r"$\log_2 = +0.36$", xy=(1.5, 3.2), xytext=(1.5, 4.3),
                  ha="center", fontsize=8.5, fontweight="bold", color="#C0392B",
                  arrowprops=dict(arrowstyle="->", lw=1.2, color="#C0392B"))

    ax_a.set_ylabel("IGHV4-34 Repertoire Frequency (%)", fontsize=10.5, fontweight="bold")
    ax_a.set_title("A   IGHV4-34 Repertoire Dynamics\n(vane Cohort, n = 62 Libraries)", loc="left", fontsize=11, fontweight="bold", pad=8)
    ax_a.set_ylim(0, 5.2)
    ax_a.grid(True, axis="y", ls=":", alpha=0.5)

    # Panel B: IGHV1-69 Kabat 54 Hydrophobic Diversification
    ax_b = fig.add_subplot(gs[1])
    sub_c1 = df_m[df_m["eval_unit"] == "IGHV_ClanI"].sort_values("imgt_pos").copy()
    sub_c1["diff_beta_alpha"] = sub_c1["fubar_beta"] - sub_c1["fubar_alpha"]
    focus = sub_c1[(sub_c1["imgt_pos"] >= 50) & (sub_c1["imgt_pos"] <= 70)]

    ax_b.bar(focus["imgt_pos"], focus["contact5_freq_combined"], width=0.75, color="#1F618D", alpha=0.5,
             edgecolor="#154360", lw=0.6, label="Viral Antigen Contact Freq")
    ax_b.set_ylabel("Antigen Contact Frequency\n(SAbDab + CoV-AbDab)", color="#154360", fontsize=10, fontweight="bold")
    ax_b.tick_params(axis="y", labelcolor="#154360")
    ax_b.set_ylim(0, 0.95)

    ax_b_twin = ax_b.twinx()
    ax_b_twin.plot(focus["imgt_pos"], focus["diff_beta_alpha"], color="#C0392B", lw=1.8, marker="o", markersize=4.5,
                   label=r"Diversification ($\beta - \alpha$)")
    ax_b_twin.axhline(0, color="#7F8C8D", ls="--", lw=0.8)
    ax_b_twin.set_ylabel(r"Diversification Parameter ($\beta - \alpha$)", color="#C0392B", fontsize=10, fontweight="bold")
    ax_b_twin.tick_params(axis="y", labelcolor="#C0392B")
    ax_b_twin.set_ylim(-1.5, 3.5)

    # Highlight Kabat 54 diversification
    ax_b.annotate("Kabat 54 (IMGT 62)\nHydrophobic Diversification\n(Human: Phe; Apes: Leu/Val/Ala)",
                  xy=(62, 0.60), xytext=(58, 0.82),
                  arrowprops=dict(facecolor="#C0392B", shrink=0.08, width=1.2, headwidth=5),
                  fontsize=8.5, fontweight="bold", color="#922B21", ha="center")

    ax_b.set_xlabel("IMGT Position in CDR2 Region (50 .. 70)", fontsize=10.5, fontweight="bold")
    ax_b.set_title("B   IGHV1-69: Diversification at Kabat 54\nSeeds Human Viral Neutralization", loc="left", fontsize=11, fontweight="bold", pad=8)
    ax_b.grid(True, axis="x", ls=":", alpha=0.5)

    # Panel C: Public T-Cell Clone Over-Mobilization
    ax_c = fig.add_subplot(gs[2])
    tcr = df_atlas[df_atlas["locus"].isin(["TRBV", "TRAV"])].dropna(subset=["public_db_count", "olga_pgen_v"]).copy()
    tcr = tcr[tcr["public_db_count"] > 0].copy()

    ax_c.scatter(tcr["olga_pgen_v"], tcr["public_db_count"], c="#7F8C8D", alpha=0.5, s=28, edgecolor="none")

    highlight_genes = [
        ("TRBV6-5", "#27AE60", (0.015, 33000), (0.003, 30000)),
        ("TRBV19", "#8E44AD", (0.031, 6874), (0.035, 12000)),
        ("TRBV5-4", "#E67E22", (0.019, 1081), (0.025, 2000))
    ]

    for g, c, (gx, gy), (tx, ty) in highlight_genes:
        row = tcr[tcr["human_gene"] == g]
        if len(row) > 0:
            rx = row["olga_pgen_v"].values[0]
            ry = row["public_db_count"].values[0]
            ax_c.scatter(rx, ry, color=c, s=65, edgecolor="#1B2631", lw=1.2, zorder=5)
            ax_c.annotate(f"{g}\n({int(ry):,} public TCRs)", xy=(rx, ry), xytext=(tx, ty),
                          arrowprops=dict(facecolor=c, shrink=0.08, width=1.0, headwidth=4),
                          fontsize=8.5, fontweight="bold", color=c)

    ax_c.set_yscale("log")
    ax_c.set_xlabel(r"Baseline Recombination Probability ($P_{\mathrm{gen}}(V)$ by OLGA)", fontsize=10, fontweight="bold")
    ax_c.set_ylabel("Public Pathogen-Specific TCR Clones (VDJdb)", fontsize=10, fontweight="bold")
    ax_c.set_title("C   Convergence of Germline Evolution\n& Human Public TCR Repertoire Mobilization", loc="left", fontsize=11, fontweight="bold", pad=8)
    ax_c.grid(True, ls=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIG_S4_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(FIG_S4_PDF, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved Figure S4 to {FIG_S4_PNG}", flush=True)

if __name__ == "__main__":
    plot_figure_s1()
    plot_figure_s2()
    plot_figure_s3()
    plot_figure_s4()
    print("[ALL DONE] Supplementary Figures S1–S4 (v2) successfully generated.", flush=True)
