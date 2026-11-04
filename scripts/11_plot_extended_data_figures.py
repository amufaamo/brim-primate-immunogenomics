#!/usr/bin/env python3
"""
11_plot_extended_data_figures.py
Publication-Grade Nature Extended Data Figures (Extended Data Figures 1-4)
T2T Primate Germline V-Segment Adaptive Evolution & Structural Repertoire Integration

Addresses Deep Review SUP-01:
- Eliminates all title collisions, label overlaps, and annotation clippings.
- Panel labels strictly unified to lowercase bold (a, b, c, d).
- Extended Data Fig 1: Repertoire Architecture & 1:1 Orthologs (reconciled counts 1,588 F / 3,671 total; n=75 1:1 ratio 1.22x)
- Extended Data Fig 2: GARD Phylogenetic Partitioning Sensitivity (all 7 units retain p < 0.03 paratope acceleration)
- Extended Data Fig 3: TCR Interface Decomposition (TRBV vs peptide rho=0.320; TRAV vs MHC-helix rho=0.285)
- Extended Data Fig 4: Antiviral & Autoimmune Archetypes (IGHV4-34 donor-level dots & boxplots; IGHV1-69 Kabat 54; public TCRs)
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
SCRIPTS_DIR = os.path.join(DRIVE_DIR, "brim_t2t_scripts")

COUNTS_TSV = os.path.join(DRIVE_DIR, "brim_t2t_02_germline_counts.tsv")
MASTER_TSV = os.path.join(DRIVE_DIR, "brim_t2t_04_master_imgt104_3d_selection.tsv")
ATLAS_TSV = os.path.join(DRIVE_DIR, "brim_t2t_04_gene_integrated_atlas.tsv")
K1_TSV = os.path.join(DRIVE_DIR, "brim_t2t_04_k1_summary.tsv")
PART_TSV = os.path.join(DRIVE_DIR, "brim_t2t_03b_partition_sensitivity.tsv")
VANE_TSV = os.path.join(SCRIPTS_DIR, "vane_bcr_v_usage_summary.tsv")

EXT_FIG1_PNG = os.path.join(DRIVE_DIR, "brim_nature_ext_fig1.png")
EXT_FIG1_PDF = os.path.join(DRIVE_DIR, "brim_nature_ext_fig1.pdf")
EXT_FIG2_PNG = os.path.join(DRIVE_DIR, "brim_nature_ext_fig2.png")
EXT_FIG2_PDF = os.path.join(DRIVE_DIR, "brim_nature_ext_fig2.pdf")
EXT_FIG3_PNG = os.path.join(DRIVE_DIR, "brim_nature_ext_fig3.png")
EXT_FIG3_PDF = os.path.join(DRIVE_DIR, "brim_nature_ext_fig3.pdf")
EXT_FIG4_PNG = os.path.join(DRIVE_DIR, "brim_nature_ext_fig4.png")
EXT_FIG4_PDF = os.path.join(DRIVE_DIR, "brim_nature_ext_fig4.pdf")

# Publication Typography
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 8.0
plt.rcParams["axes.titlesize"] = 9.0
plt.rcParams["axes.labelsize"] = 8.5
plt.rcParams["xtick.labelsize"] = 7.5
plt.rcParams["ytick.labelsize"] = 7.5
plt.rcParams["legend.fontsize"] = 7.5
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.edgecolor"] = "#222222"
plt.rcParams["xtick.major.width"] = 0.8
plt.rcParams["ytick.major.width"] = 0.8
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"

LOCUS_COLORS = {
    "IGHV": "#1F618D",  # Navy
    "IGKV": "#E67E22",  # Warm Orange
    "IGLV": "#16A085",  # Teal Green
    "TRAV": "#8E44AD",  # Royal Purple
    "TRBV": "#27AE60"   # Forest Green
}

# =============================================================================
# EXTENDED DATA FIGURE 1: Catarrhine T2T Germline V Repertoire Architecture
# =============================================================================
def plot_extended_data_fig1():
    print("[RUN] Generating Extended Data Figure 1...", flush=True)
    df_counts = pd.read_csv(COUNTS_TSV, sep="\t")
    df_atlas = pd.read_csv(ATLAS_TSV, sep="\t")

    species_order = [
        ("Human_CHM13", "Human\n(CHM13v2.0)"),
        ("Pan_paniscus", "Bonobo\n(mPanPan1)"),
        ("Pan_troglodytes", "Chimpanzee\n(mPanTro3)"),
        ("Gorilla_gorilla", "Gorilla\n(mGorGor1)"),
        ("Pongo_abelii", "Sumatran orang.\n(mPonAbe1)"),
        ("Pongo_pygmaeus", "Bornean orang.\n(mPonPyg2)"),
        ("Symphalangus_syndactylus", "Siamang\n(mSymSyn1)"),
        ("Macaca_mulatta", "Rhesus macaque\n(T2T-MMU8v2.0)")
    ]

    fig = plt.figure(figsize=(15, 6.2), dpi=300)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.35, 0.75, 1.0], wspace=0.30,
                           left=0.06, right=0.97, top=0.90, bottom=0.14)

    # Panel a: Stacked Bar Chart of Functional V-Genes
    ax_a = fig.add_subplot(gs[0])
    loci = ["IGHV", "IGKV", "IGLV", "TRAV", "TRBV"]
    x = np.arange(len(species_order))
    bottoms = np.zeros(len(species_order))

    for locus in loci:
        counts = []
        for asm_id, _ in species_order:
            sub = df_counts[(df_counts["assembly_id"] == asm_id) & (df_counts["locus"] == locus)]
            val = sub["n_functional_F"].values[0] if len(sub) > 0 else 0
            counts.append(val)
        counts = np.array(counts)
        ax_a.bar(x, counts, bottom=bottoms, label=locus, color=LOCUS_COLORS[locus],
                 edgecolor="white", lw=0.6, alpha=0.92, width=0.62)
        bottoms += counts

    for i, tot in enumerate(bottoms):
        col = "#C0392B" if i == 6 else "#222222"
        y_pos = tot + 6 if i != 6 else tot + 5
        ax_a.text(i, y_pos, f"{int(tot)}", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color=col)

    # Siamang annotation cleanly spaced
    siamang_idx = 6
    siamang_tot = bottoms[siamang_idx]
    ax_a.annotate("IGKV deleted (0)\nIGHV contracted (8)",
                  xy=(siamang_idx + 0.15, siamang_tot), xytext=(siamang_idx - 1.25, siamang_tot + 85),
                  arrowprops=dict(facecolor="#C0392B", edgecolor="#900C3F", shrink=0.08, width=1.3, headwidth=5),
                  fontsize=8.0, fontweight="bold", color="#900C3F", ha="center",
                  bbox=dict(boxstyle="round,pad=0.25", fc="#FDEDEC", ec="#E6B0AA", lw=0.8))

    ax_a.set_xticks(x)
    ax_a.set_xticklabels([name for _, name in species_order], rotation=30, ha="right", fontsize=8.0)
    ax_a.set_ylabel("Functional V-Genes (Haploid)", fontsize=9.0, fontweight="bold")
    ax_a.set_ylim(0, 345)
    ax_a.grid(True, axis="y", ls=":", alpha=0.5)
    ax_a.legend(loc="upper left", frameon=True, fontsize=7.5, ncol=2, framealpha=0.9)
    ax_a.set_title("a   Functional Germline V-Genes Across 8 T2T Primates\n"
                   "(1,588 Functional / 3,671 Total Segments)",
                   loc="left", fontsize=9.5, fontweight="bold", pad=8)

    # Panel b: Single-Copy 1:1 Orthologs by Locus
    ax_b = fig.add_subplot(gs[1])
    strict_df = df_atlas[df_atlas["is_strict_1to1"] == 1]
    locus_counts = strict_df["locus"].value_counts().reindex(loci).fillna(0)

    bars_b = ax_b.bar(range(len(loci)), locus_counts.values, color=[LOCUS_COLORS[l] for l in loci],
                      edgecolor="#222222", lw=0.8, alpha=0.9, width=0.58)
    for bar, val in zip(bars_b, locus_counts.values):
        ax_b.text(bar.get_x() + bar.get_width()/2.0, val + 0.8, f"{int(val)}", ha="center", va="bottom",
                  fontsize=8.5, fontweight="bold")

    ax_b.set_xticks(range(len(loci)))
    ax_b.set_xticklabels(loci, fontsize=8.5, fontweight="bold")
    ax_b.set_ylabel("Number of Single-Copy 1:1 Orthologs", fontsize=9.0, fontweight="bold")
    ax_b.set_ylim(0, 38)
    ax_b.grid(True, axis="y", ls=":", alpha=0.5)
    ax_b.set_title("b   Single-Copy 1:1 Orthologs\n(n = 75 Paralog-Free)",
                   loc="left", fontsize=9.5, fontweight="bold", pad=8)

    # Panel c: Strict 1:1 Orthologs Paratope Acceleration
    ax_c = fig.add_subplot(gs[2])
    data_by_locus = [strict_df[strict_df["locus"] == l]["cdr12_vs_fr_omega_ratio"].dropna().values for l in ["IGHV", "TRAV", "TRBV"]]
    data_all = strict_df["cdr12_vs_fr_omega_ratio"].dropna().values

    all_data = [data_all] + data_by_locus
    labels_c = ["All 1:1\n(n=75)", "IGHV\n(n=5)", "TRAV\n(n=31)", "TRBV\n(n=31)"]
    colors_c = ["#34495E", LOCUS_COLORS["IGHV"], LOCUS_COLORS["TRAV"], LOCUS_COLORS["TRBV"]]

    bp_c = ax_c.boxplot(all_data, tick_labels=labels_c, patch_artist=True, widths=0.52,
                        medianprops=dict(color="#C0392B", lw=1.8), showfliers=False)
    for patch, c in zip(bp_c["boxes"], colors_c):
        patch.set_facecolor(c)
        patch.set_alpha(0.32)
        patch.set_edgecolor(c)
        patch.set_linewidth(1.2)

    np.random.seed(42)
    for i, (vals, c) in enumerate(zip(all_data, colors_c)):
        jitter = np.random.normal(0, 0.05, size=len(vals))
        ax_c.scatter(np.full_like(vals, i + 1) + jitter, vals, color=c, alpha=0.80, s=28, edgecolor="white", lw=0.4)

    ax_c.axhline(1.0, color="#7F8C8D", ls="--", lw=1.0, zorder=0)

    stat_w, p_w = sp_stats.wilcoxon(data_all - 1.0)
    ax_c.text(0.96, 0.94, f"All 1:1 Orthologs:\nMedian Ratio = {np.median(data_all):.2f}×\nWilcoxon P = {p_w:.2e}",
              transform=ax_c.transAxes, ha="right", va="top", fontsize=7.5, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.3", fc="#FFFFFF", ec="#BDC3C7", lw=0.8, alpha=0.95))

    ax_c.set_ylabel(r"Paratope Diversification Ratio ($\omega_{\mathrm{CDR}} / \omega_{\mathrm{FR}}$)", fontsize=9.0, fontweight="bold")
    ax_c.set_ylim(0.4, 4.25)
    ax_c.grid(True, axis="y", ls=":", alpha=0.5)
    ax_c.set_title("c   Paratope Acceleration in 1:1 Orthologs\n(Paralog-Free Baseline)",
                   loc="left", fontsize=9.5, fontweight="bold", pad=8)

    plt.savefig(EXT_FIG1_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(EXT_FIG1_PDF, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved Extended Data Fig 1:\n  PNG: {EXT_FIG1_PNG}\n  PDF: {EXT_FIG1_PDF}", flush=True)


# =============================================================================
# EXTENDED DATA FIGURE 2: Phylogenetic Partitioning & Gene Conversion Audit
# =============================================================================
def plot_extended_data_fig2():
    print("[RUN] Generating Extended Data Figure 2...", flush=True)
    df_k1 = pd.read_csv(K1_TSV, sep="\t")
    df_part = pd.read_csv(PART_TSV, sep="\t")

    merged = pd.merge(df_k1, df_part, left_on="dataset", right_on="unit")
    units = merged["dataset"].tolist()
    labels = [u.replace("_All", "").replace("_MetaClans", " Meta").replace("IGHV_", "").replace("Clan", "Clan ") for u in units]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), dpi=300)
    plt.subplots_adjust(wspace=0.28, left=0.07, right=0.96, top=0.88, bottom=0.16)

    x = np.arange(len(units))
    w = 0.20

    # Panel a: Codon Selection Rates across Partition Regimes
    ax1.bar(x - 1.5*w, merged["median_omega_cdr12"], width=w, label="Standard Tree CDR1+2",
            color="#C0392B", alpha=0.85, edgecolor="#7B241C", lw=0.7)
    ax1.bar(x - 0.5*w, merged["median_omega_core_fr"], width=w, label="Standard Tree Core FR",
            color="#2980B9", alpha=0.85, edgecolor="#1B4F72", lw=0.7)
    ax1.bar(x + 0.5*w, merged["median_omega_cdr12_part_tree"], width=w, label="5'/3' Partitioned CDR1+2",
            color="#E74C3C", alpha=0.55, edgecolor="#7B241C", lw=0.7, hatch="//")
    ax1.bar(x + 1.5*w, merged["median_omega_core_fr_part_tree"], width=w, label="5'/3' Partitioned Core FR",
            color="#3498DB", alpha=0.55, edgecolor="#1B4F72", lw=0.7, hatch="//")

    ax1.axhline(1.0, color="#7F8C8D", ls="--", lw=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.0, fontweight="bold")
    ax1.set_ylabel(r"Median Codon Selection ($\omega = d_N/d_S$)", fontsize=9.0, fontweight="bold")
    ax1.set_title("a   Codon Selection Rates Across Phylogenetic Partition Regimes",
                  loc="left", fontsize=9.5, fontweight="bold", pad=8)
    ax1.grid(True, axis="y", ls=":", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True, fontsize=7.5, ncol=2)
    ax1.set_ylim(0, 1.35)

    # Panel b: Paratope Acceleration Ratio Robustness
    r_std = merged["omega_median_ratio_cdr_vs_fr"]
    r_part = merged["ratio_cdr_vs_fr_part_tree"]

    ax2.plot(x, r_std, marker="o", markersize=7.5, color="#8E44AD", lw=1.8, label="Standard Single Tree")
    ax2.plot(x, r_part, marker="s", markersize=7.5, color="#16A085", lw=1.8, ls="--", label="5'/3' Partitioned Trees (IMGT 1–55 / 56–104)")

    # Offsets tailored to avoid line collisions
    text_y_offsets = [-0.11, 0.10, 0.10, -0.16, 0.10, -0.16, -0.11]
    for i in range(len(units)):
        dy = text_y_offsets[i]
        va = "bottom" if dy > 0 else "top"
        ax2.annotate(f"{r_part[i]:.2f}×", (x[i], r_part[i]), xytext=(x[i], r_part[i] + dy),
                     ha="center", va=va, fontsize=7.5, fontweight="bold", color="#16A085")

    # Label TRBV Standard Tree cleanly above marker
    ax2.annotate(f"{r_std[6]:.2f}×", (x[6], r_std[6]), xytext=(x[6], r_std[6] + 0.08),
                 ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#8E44AD")

    ax2.axhline(1.0, color="#C0392B", ls=":", lw=1.2, label="Neutral Threshold (1.0×)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.0, fontweight="bold")
    ax2.set_ylabel(r"Acceleration Ratio ($\omega_{\mathrm{CDR}} / \omega_{\mathrm{FR}}$)", fontsize=9.0, fontweight="bold")
    ax2.set_title("b   Robustness of Paratope Acceleration Against Interlocus Gene Conversion\n"
                  "(All 7 units retain significant acceleration under partitioned trees, P < 0.03)",
                  loc="left", fontsize=9.5, fontweight="bold", pad=8)
    ax2.grid(True, ls=":", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True, fontsize=8.0)
    ax2.set_ylim(0.8, 3.4)

    plt.savefig(EXT_FIG2_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(EXT_FIG2_PDF, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved Extended Data Fig 2:\n  PNG: {EXT_FIG2_PNG}\n  PDF: {EXT_FIG2_PDF}", flush=True)


# =============================================================================
# EXTENDED DATA FIGURE 3: TCR Interface Decomposition (Peptide vs MHC)
# =============================================================================
def plot_extended_data_fig3():
    print("[RUN] Generating Extended Data Figure 3...", flush=True)
    df_m = pd.read_csv(MASTER_TSV, sep="\t")

    trbv = df_m[df_m["eval_unit"] == "TRBV_All"].sort_values("imgt_pos").copy()
    trav = df_m[df_m["eval_unit"] == "TRAV_All"].sort_values("imgt_pos").copy()

    trbv["diff_beta_alpha"] = trbv["fubar_beta"] - trbv["fubar_alpha"]
    trav["diff_beta_alpha"] = trav["fubar_beta"] - trav["fubar_alpha"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8.5), dpi=300, sharex=True)
    plt.subplots_adjust(hspace=0.28, left=0.07, right=0.92, top=0.92, bottom=0.10)

    # Panel a: TRBV
    ax1.bar(trbv["imgt_pos"], trbv["contact5_peptide_freq"], width=0.85, color="#3498DB", alpha=0.55,
            edgecolor="#2980B9", lw=0.5, label="Empirical Peptide Contact Freq (<5.0 Å)")
    ax1.set_ylabel("Peptide Contact Frequency\n(n = 260 TRBV chains)", color="#2980B9", fontsize=8.5, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#2980B9")
    ax1.set_ylim(0, 0.85)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(trbv["imgt_pos"], trbv["diff_beta_alpha"], color="#C0392B", lw=1.6, marker="o", markersize=3.2,
                  label=r"Evolutionary Diversification (FUBAR $\beta - \alpha$)")
    ax1_twin.axhline(0, color="#7F8C8D", ls="--", lw=0.8)
    ax1_twin.set_ylabel(r"Diversification ($\beta - \alpha$)", color="#C0392B", fontsize=8.5, fontweight="bold")
    ax1_twin.tick_params(axis="y", labelcolor="#C0392B")
    ax1_twin.set_ylim(-2.5, 4.2)

    ax1.axvspan(26.5, 38.5, color="#FADBD8", alpha=0.35, label="CDR1 (IMGT 27–38)")
    ax1.axvspan(55.5, 65.5, color="#F5B7B1", alpha=0.45, label="CDR2 (IMGT 56–65)")
    
    # Place annotation cleanly at the top right of CDR2
    ax1.annotate(r"$\bf{TRBV\ CDR2\ Peptide\ Footprint}$" "\n" r"(Partial $\rho = 0.320$, $p = 0.0019$)",
                 xy=(58, 0.45), xytext=(68, 0.68),
                 arrowprops=dict(facecolor="#C0392B", edgecolor="#7B241C", shrink=0.08, width=1.2, headwidth=5),
                 fontsize=8.0, fontweight="bold", color="#922B21",
                 bbox=dict(boxstyle="round,pad=0.25", fc="#FDEDEC", ec="#E6B0AA", lw=0.8))

    ax1.set_title("a   TRBV (TCRβ): Positive Selection Tracks the Antigenic Peptide Contact Interface",
                  loc="left", fontsize=9.5, fontweight="bold", pad=8)
    ax1.grid(True, axis="x", ls=":", alpha=0.5)

    # Panel b: TRAV
    ax2.bar(trav["imgt_pos"], trav["contact5_mhc_freq"], width=0.85, color="#E67E22", alpha=0.55,
            edgecolor="#D35400", lw=0.5, label="Empirical MHC Contact Freq (<5.0 Å)")
    ax2.set_ylabel("MHC Contact Frequency\n(n = 256 TRAV chains)", color="#D35400", fontsize=8.5, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#D35400")
    ax2.set_ylim(0, 1.15)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(trav["imgt_pos"], trav["diff_beta_alpha"], color="#C0392B", lw=1.6, marker="s", markersize=3.2,
                  label=r"Evolutionary Diversification (FUBAR $\beta - \alpha$)")
    ax2_twin.axhline(0, color="#7F8C8D", ls="--", lw=0.8)
    ax2_twin.set_ylabel(r"Diversification ($\beta - \alpha$)", color="#C0392B", fontsize=8.5, fontweight="bold")
    ax2_twin.tick_params(axis="y", labelcolor="#C0392B")
    ax2_twin.set_ylim(-2.5, 3.8)

    ax2.axvspan(26.5, 38.5, color="#FCF3CF", alpha=0.45, label="CDR1 (IMGT 27–38)")
    ax2.axvspan(55.5, 65.5, color="#FDEBD0", alpha=0.45, label="CDR2 (IMGT 56–65)")
    ax2.annotate(r"$\bf{TRAV\ CDR1\ MHC\ Helix\ Footprint}$" "\n" r"(Spearman $\rho = 0.285$, $p = 0.0059$)",
                 xy=(30, 0.90), xytext=(40, 0.92),
                 arrowprops=dict(facecolor="#D35400", edgecolor="#A04000", shrink=0.08, width=1.2, headwidth=5),
                 fontsize=8.0, fontweight="bold", color="#A04000",
                 bbox=dict(boxstyle="round,pad=0.25", fc="#FEF9E7", ec="#FAD7A0", lw=0.8))

    ax2.set_title("b   TRAV (TCRα): Evolutionary Diversification Tracks the Conserved MHC Interaction Surface",
                  loc="left", fontsize=9.5, fontweight="bold", pad=8)
    ax2.set_xlabel("IMGT Unique Codon Position (1 .. 104)", fontsize=9.0, fontweight="bold")
    ax2.set_xlim(0.5, 104.5)
    ax2.set_xticks(range(10, 105, 10))
    ax2.grid(True, axis="x", ls=":", alpha=0.5)

    plt.savefig(EXT_FIG3_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(EXT_FIG3_PDF, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved Extended Data Fig 3:\n  PNG: {EXT_FIG3_PNG}\n  PDF: {EXT_FIG3_PDF}", flush=True)


# =============================================================================
# EXTENDED DATA FIGURE 4: Antiviral & Autoimmune Archetypes
# =============================================================================
def plot_extended_data_fig4():
    print("[RUN] Generating Extended Data Figure 4...", flush=True)
    df_atlas = pd.read_csv(ATLAS_TSV, sep="\t")
    df_m = pd.read_csv(MASTER_TSV, sep="\t")
    df_vane = pd.read_csv(VANE_TSV, sep="\t")

    fig = plt.figure(figsize=(15, 6.0), dpi=300)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.1, 1.0, 1.1], wspace=0.32,
                           left=0.06, right=0.97, top=0.88, bottom=0.14)

    # Panel a: IGHV4-34 Peripheral Regulation in vane cohort (Donor Dots + Boxplot)
    ax_a = fig.add_subplot(gs[0])
    vane_434 = df_vane[(df_vane["v_gene"] == "IGHV4-34") & (df_vane["locus"] == "IGH")].copy()

    hd_n = vane_434[(vane_434["cohort"] == "HD") & (vane_434["subset"] == "N")]["freq_seq"].values * 100
    hd_m = vane_434[(vane_434["cohort"] == "HD") & (vane_434["subset"] == "M")]["freq_seq"].values * 100
    achr_m = vane_434[(vane_434["cohort"] == "AChR_MG") & (vane_434["subset"] == "M")]["freq_seq"].values * 100
    musk_m = vane_434[(vane_434["cohort"] == "MuSK_MG") & (vane_434["subset"] == "M")]["freq_seq"].values * 100
    pooled_m = np.concatenate([achr_m, musk_m])

    plot_data_a = [hd_n, hd_m, achr_m, musk_m, pooled_m]
    labels_a = ["HD\nNaive\n(n=4)", "HD\nMem\n(n=4)", "AChR\nMem\n(n=4)", "MuSK\nMem\n(n=5)", "Pooled\nMG Mem\n(n=9)"]
    colors_a = ["#2980B9", "#27AE60", "#E67E22", "#8E44AD", "#C0392B"]

    bp_a = ax_a.boxplot(plot_data_a, positions=range(5), tick_labels=labels_a, patch_artist=True, widths=0.48,
                        medianprops=dict(color="#111111", lw=1.6), showfliers=False)
    for patch, col in zip(bp_a["boxes"], colors_a):
        patch.set_facecolor(col)
        patch.set_alpha(0.35)
        patch.set_edgecolor(col)
        patch.set_linewidth(1.1)

    np.random.seed(42)
    for i, (vals, col) in enumerate(zip(plot_data_a, colors_a)):
        jitter = np.random.normal(0, 0.04, size=len(vals))
        ax_a.scatter(np.full_like(vals, i) + jitter, vals, color=col, s=45,
                     edgecolor="#1B2631", lw=0.8, zorder=4)

    # Paired HD connecting lines with clean non-overlapping donor labels strictly inside frame
    hd_df = vane_434[vane_434["cohort"] == "HD"].pivot(index="donor", columns="subset", values="freq_seq") * 100
    for donor, row in hd_df.iterrows():
        ax_a.plot([0, 1], [row["N"], row["M"]], color="#7F8C8D", ls=":", lw=1.0, alpha=0.8, zorder=2)
        ax_a.text(-0.15, row["N"], donor, ha="right", va="center", fontsize=7.5, color="#444444", fontweight="bold")

    means_a = [np.mean(v) for v in plot_data_a]
    for i, m in enumerate(means_a):
        ax_a.text(i, 0.35, f"{m:.2f}%", ha="center", va="bottom", fontsize=7.2, fontweight="bold")

    # Annotate Purge & Non-significant MG expansion
    ax_a.plot([0, 0, 1, 1], [6.6, 6.8, 6.8, 6.6], color="#2980B9", lw=1.0)
    ax_a.text(0.5, 6.9, "Purge: P = 0.057", ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#1B4F72")

    ax_a.plot([1, 1, 4, 4], [7.5, 7.7, 7.7, 7.5], color="#C0392B", lw=1.0)
    ax_a.text(2.5, 7.8, "HD Mem vs MG: P = 0.414 (n.s.)", ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#78281F")

    ax_a.set_xlim(-0.75, 4.6)
    ax_a.set_ylabel("IGHV4-34 Frequency (%)", fontsize=8.5, fontweight="bold")
    ax_a.set_ylim(0, 9.0)
    ax_a.grid(True, axis="y", ls=":", alpha=0.5)
    ax_a.set_title("a   IGHV4-34 Repertoire Dynamics\n(vane Cohort, n = 14 Donors)", loc="left", fontsize=9.5, fontweight="bold", pad=8)

    # Panel b: IGHV1-69 Kabat 54 Hydrophobic Diversification
    ax_b = fig.add_subplot(gs[1])
    sub_c1 = df_m[df_m["eval_unit"] == "IGHV_ClanI"].sort_values("imgt_pos").copy()
    sub_c1["diff_beta_alpha"] = sub_c1["fubar_beta"] - sub_c1["fubar_alpha"]
    focus = sub_c1[(sub_c1["imgt_pos"] >= 50) & (sub_c1["imgt_pos"] <= 70)]

    ax_b.bar(focus["imgt_pos"], focus["contact5_freq_combined"], width=0.75, color="#1F618D", alpha=0.45,
             edgecolor="#154360", lw=0.6, label="Viral Ag Contact Freq")
    ax_b.set_ylabel("Viral Contact Freq", color="#154360", fontsize=8.5, fontweight="bold")
    ax_b.tick_params(axis="y", labelcolor="#154360")
    ax_b.set_ylim(0, 0.95)

    ax_b_twin = ax_b.twinx()
    ax_b_twin.plot(focus["imgt_pos"], focus["diff_beta_alpha"], color="#C0392B", lw=1.6, marker="o", markersize=3.5,
                   label=r"Diversification ($\beta - \alpha$)")
    ax_b_twin.axhline(0, color="#7F8C8D", ls="--", lw=0.8)
    ax_b_twin.set_ylabel(r"Selection ($\beta - \alpha$)", color="#C0392B", fontsize=8.5, fontweight="bold")
    ax_b_twin.tick_params(axis="y", labelcolor="#C0392B")
    ax_b_twin.set_ylim(-2.0, 7.2)

    ax_b.axvspan(55.5, 65.5, color="#FADBD8", alpha=0.35, zorder=0)
    ax_b_twin.annotate("Kabat 54 (IMGT 62)\n" r"$\omega = 2.32,\ \beta-\alpha = +1.70$" "\n(Human: Phe; Apes: Leu)",
                       xy=(62, 1.70), xytext=(66.0, 5.9),
                       arrowprops=dict(facecolor="#C0392B", edgecolor="#7B241C", shrink=0.08, width=1.1, headwidth=4.5),
                       fontsize=7.5, fontweight="bold", color="#78281F", ha="center",
                       bbox=dict(boxstyle="round,pad=0.2", fc="#FDEDEC", ec="#E6B0AA", lw=0.8))

    ax_b.set_xlim(49.5, 70.5)
    ax_b.set_xticks(range(50, 71, 2))
    ax_b.set_xlabel("IMGT Position in CDR2 (50 .. 70)", fontsize=8.5, fontweight="bold")
    ax_b.set_title("b   IGHV1-69: Diversification at Kabat 54\nSeeds Human Viral Neutralization",
                   loc="left", fontsize=9.5, fontweight="bold", pad=8)
    ax_b.grid(True, axis="x", ls=":", alpha=0.5)

    # Panel c: Public T-Cell Clone Over-Mobilization (VDJdb vs OLGA)
    ax_c = fig.add_subplot(gs[2])
    tcr = df_atlas[df_atlas["locus"].isin(["TRBV", "TRAV"])].dropna(subset=["public_db_count", "olga_pgen_v"]).copy()
    tcr = tcr[tcr["public_db_count"] > 0].copy()

    ax_c.scatter(tcr["olga_pgen_v"], tcr["public_db_count"], c="#7F8C8D", alpha=0.5, s=28, edgecolor="none")

    highlight_genes = [
        ("TRBV6-5", "#27AE60", (0.015, 32968), (0.003, 30000)),
        ("TRBV19", "#8E44AD", (0.031, 6874), (0.035, 12000)),
        ("TRBV5-4", "#E67E22", (0.019, 1081), (0.025, 2000))
    ]

    for g, c, (gx, gy), (tx, ty) in highlight_genes:
        row = tcr[tcr["human_gene"] == g]
        if len(row) > 0:
            rx = row["olga_pgen_v"].values[0]
            ry = row["public_db_count"].values[0]
            ax_c.scatter(rx, ry, color=c, s=65, edgecolor="#1B2631", lw=1.1, zorder=5)
            ax_c.annotate(f"{g}\n({int(ry):,} TCRs)", xy=(rx, ry), xytext=(tx, ty),
                          arrowprops=dict(facecolor=c, edgecolor="#1B2631", shrink=0.08, width=0.9, headwidth=4),
                          fontsize=7.5, fontweight="bold", color=c)

    ax_c.set_yscale("log")
    ax_c.set_ylim(bottom=1, top=6.5e4)
    ax_c.set_xlabel(r"Baseline Recombination ($P_{\mathrm{gen}}(V)$ by OLGA)", fontsize=8.5, fontweight="bold")
    ax_c.set_ylabel("Public Pathogen-Specific TCR Clones (VDJdb)", fontsize=8.5, fontweight="bold")
    ax_c.set_title("c   Public T-Cell Repertoire Mobilization\n(VDJdb Public TCRs vs OLGA Baseline)",
                   loc="left", fontsize=9.5, fontweight="bold", pad=8)
    ax_c.grid(True, ls=":", alpha=0.5)

    plt.savefig(EXT_FIG4_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(EXT_FIG4_PDF, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved Extended Data Fig 4:\n  PNG: {EXT_FIG4_PNG}\n  PDF: {EXT_FIG4_PDF}", flush=True)


if __name__ == "__main__":
    plot_extended_data_fig1()
    plot_extended_data_fig2()
    plot_extended_data_fig3()
    plot_extended_data_fig4()
    print("[ALL DONE] Nature Extended Data Figures 1–4 generated successfully.", flush=True)
