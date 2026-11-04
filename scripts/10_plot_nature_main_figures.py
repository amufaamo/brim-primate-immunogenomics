#!/usr/bin/env python3
"""
10_plot_nature_main_figures.py
Publication-Grade Nature Main Figures (Figures 1-4)
T2T Primate Germline V-Segment Adaptive Evolution & Structural Repertoire Integration

Addresses Deep Review FIG-01:
- Fig 1: Catarrhine T2T Germline V Census (a: 8-species phylogeny + stacked counts; b: BCR vs TCR CNV; c: segmental status & orthogroups)
- Fig 2: Paratope Diversification vs Surface Controls & Recombination (a: IMGT scheme; b: CDR vs Core FR vs Exposed-NC-FR; c: 1:1 orthologs & FUBAR; d: partitioned tree sensitivity)
- Fig 3: 3D Crystallographic Interface Alignment (a: IGHV dual track; b: contact vs selection scatter & partial rho; c: IGKV vs IGLV asymmetry; d: TCR peptide vs MHC helix decomposition)
- Fig 4: Public Antiviral Archetypes & Peripheral Tolerance (a: IGHV1-69 Kabat 54; b: 147 ortholog groups scatter with exact 1:1 vs multi-copy; c: vane IGHV4-34 donor-level dots & boxplots)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from scipy import stats as sp_stats

# Directories
DRIVE_DIR = os.environ.get("BRIM_DATA_DIR", os.environ.get("BRIM_DATA_DIR", "./data"))
SCRIPTS_DIR = os.path.join(DRIVE_DIR, "brim_t2t_scripts")

COUNTS_TSV = os.path.join(DRIVE_DIR, "brim_t2t_02_germline_counts.tsv")
MASTER_TSV = os.path.join(DRIVE_DIR, "brim_t2t_04_master_imgt104_3d_selection.tsv")
ATLAS_TSV = os.path.join(DRIVE_DIR, "brim_t2t_04_gene_integrated_atlas.tsv")
K1_TSV = os.path.join(DRIVE_DIR, "brim_t2t_04_k1_summary.tsv")
K2_TSV = os.path.join(DRIVE_DIR, "brim_t2t_04_k2_3d_correlations.tsv")
K3_TSV = os.path.join(DRIVE_DIR, "brim_t2t_04_k3_repertoire_tests.tsv")
PART_TSV = os.path.join(DRIVE_DIR, "brim_t2t_03b_partition_sensitivity.tsv")
VANE_TSV = os.path.join(SCRIPTS_DIR, "vane_bcr_v_usage_summary.tsv")

FIG1_PNG = os.path.join(DRIVE_DIR, "brim_nature_fig1.png")
FIG1_PDF = os.path.join(DRIVE_DIR, "brim_nature_fig1.pdf")
FIG2_PNG = os.path.join(DRIVE_DIR, "brim_nature_fig2.png")
FIG2_PDF = os.path.join(DRIVE_DIR, "brim_nature_fig2.pdf")
FIG3_PNG = os.path.join(DRIVE_DIR, "brim_nature_fig3.png")
FIG3_PDF = os.path.join(DRIVE_DIR, "brim_nature_fig3.pdf")
FIG4_PNG = os.path.join(DRIVE_DIR, "brim_nature_fig4.png")
FIG4_PDF = os.path.join(DRIVE_DIR, "brim_nature_fig4.pdf")

# Typography & Global Aesthetics
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
    "IGHV": "#1F618D",  # Dark Blue
    "IGKV": "#E67E22",  # Warm Orange
    "IGLV": "#16A085",  # Teal Green
    "TRAV": "#8E44AD",  # Royal Purple
    "TRBV": "#27AE60"   # Forest Green
}

# =============================================================================
# FIGURE 1: Catarrhine T2T Germline V Census
# =============================================================================
def plot_figure_1():
    print("[RUN] Generating Main Figure 1...", flush=True)
    df_counts = pd.read_csv(COUNTS_TSV, sep="\t")
    df_atlas = pd.read_csv(ATLAS_TSV, sep="\t")

    # 8 T2T assemblies + GRCh38 control
    species_info = [
        ("Human_CHM13", "Human (T2T-CHM13v2.0)", 0.0, "H. sapiens"),
        ("Pan_paniscus", "Bonobo (mPanPan1)", 6.4, "P. paniscus"),
        ("Pan_troglodytes", "Chimpanzee (mPanTro3)", 6.4, "P. troglodytes"),
        ("Gorilla_gorilla", "Gorilla (mGorGor1)", 8.6, "G. gorilla"),
        ("Pongo_abelii", "Sumatran orangutan (mPonAbe1)", 15.2, "P. abelii"),
        ("Pongo_pygmaeus", "Bornean orangutan (mPonPyg2)", 15.2, "P. pygmaeus"),
        ("Symphalangus_syndactylus", "Siamang (mSymSyn1)", 19.8, "S. syndactylus"),
        ("Macaca_mulatta", "Rhesus macaque (T2T-MMU8v2.0)", 29.4, "M. mulatta"),
        ("Human_GRCh38", "Human (GRCh38.p14 ref)", 0.0, "Control ref")
    ]
    t2t_only = species_info[:8]

    fig = plt.figure(figsize=(14, 11), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.1, 0.9], width_ratios=[1.25, 0.75],
                           hspace=0.34, wspace=0.25, left=0.06, right=0.97, top=0.96, bottom=0.06)

    # -------------------------------------------------------------------------
    # Panel a: 8 T2T Primate Functional V Census Stacked Bar Chart
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, :])
    loci = ["IGHV", "IGKV", "IGLV", "TRAV", "TRBV"]
    x = np.arange(len(species_info))
    bottoms = np.zeros(len(species_info))

    for locus in loci:
        counts = []
        for asm_id, _, _, _ in species_info:
            sub = df_counts[(df_counts["assembly_id"] == asm_id) & (df_counts["locus"] == locus)]
            val = sub["n_functional_F"].values[0] if len(sub) > 0 else 0
            counts.append(val)
        counts = np.array(counts)
        ax_a.bar(x, counts, bottom=bottoms, label=locus, color=LOCUS_COLORS[locus],
                 edgecolor="white", lw=0.6, alpha=0.92, width=0.62)
        bottoms += counts

    # Label totals on top of bars
    for i, tot in enumerate(bottoms):
        color = "#C0392B" if i == 6 else "#222222"
        y_pos = tot + 6 if i != 6 else tot + 5
        ax_a.text(i, y_pos, f"{int(tot)}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=color)

    ax_a.axvline(7.5, color="#7F8C8D", ls="--", lw=1.0)
    ax_a.text(7.6, 290, "Short-read\nReference", fontsize=8.0, color="#555555", va="top")

    # Siamang annotation (pointing clearly to Siamang bar top without covering the '98' label)
    siamang_idx = 6
    siamang_tot = bottoms[siamang_idx]
    ax_a.annotate("IGKV total deletion (0 genes)\nIGHV extreme contraction (8 genes)",
                  xy=(siamang_idx + 0.15, siamang_tot), xytext=(siamang_idx - 1.25, siamang_tot + 85),
                  arrowprops=dict(facecolor="#C0392B", edgecolor="#900C3F", shrink=0.08, width=1.4, headwidth=6),
                  fontsize=8.5, fontweight="bold", color="#900C3F", ha="center",
                  bbox=dict(boxstyle="round,pad=0.3", fc="#FDEDEC", ec="#E6B0AA", lw=0.8))

    x_labels = [f"{name}\n[{myr} Myr]" if myr > 0 else f"{name}" for _, name, myr, _ in species_info]
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(x_labels, rotation=22, ha="right", fontsize=8.5)
    ax_a.set_ylabel("Functional Germline V-Genes (Haploid)", fontsize=9.5, fontweight="bold")
    ax_a.set_ylim(0, 345)
    ax_a.grid(True, axis="y", ls=":", alpha=0.5)
    ax_a.legend(loc="upper left", frameon=True, fontsize=8.5, ncol=5, framealpha=0.9)
    ax_a.set_title("a   Catarrhine T2T Germline V Census Across 8 High-Quality Primate Genomes\n"
                   "(8 T2T Catarrhines: 1,588 Functional V-Genes / 3,671 Total Segments; Human GRCh38.p14 Reference Control: 214 F)",
                   loc="left", fontsize=10.5, fontweight="bold", pad=8)

    # -------------------------------------------------------------------------
    # Panel b: Cross-Species Copy Number Dynamics (BCR vs TCR)
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[1, 0])

    t2t_counts = df_counts[df_counts["assembly_id"].isin([s[0] for s in t2t_only])].copy()
    
    locus_order = ["IGHV", "IGKV", "IGLV", "TRAV", "TRBV"]
    locus_data = [t2t_counts[t2t_counts["locus"] == l]["n_functional_F"].values for l in locus_order]
    
    bp = ax_b.boxplot(locus_data, tick_labels=locus_order, patch_artist=True, widths=0.5,
                      medianprops=dict(color="#111111", lw=1.8), showfliers=False)
    
    for patch, locus in zip(bp["boxes"], locus_order):
        patch.set_facecolor(LOCUS_COLORS[locus])
        patch.set_alpha(0.35)
        patch.set_edgecolor(LOCUS_COLORS[locus])
        patch.set_linewidth(1.3)
        
    np.random.seed(42)
    for i, (locus, vals) in enumerate(zip(locus_order, locus_data)):
        jitter = np.random.normal(0, 0.05, size=len(vals))
        ax_b.scatter(np.full_like(vals, i + 1) + jitter, vals, color=LOCUS_COLORS[locus],
                     s=45, alpha=0.9, edgecolor="#1B2631", lw=0.8, zorder=4)

    # Annotations of ranges
    ax_b.text(1, 71, "Range 8–66\n(8.3×)", ha="center", fontsize=7.5, fontweight="bold", color=LOCUS_COLORS["IGHV"])
    ax_b.text(2, 77, "Range 0–72\n(Full Loss)", ha="center", fontsize=7.5, fontweight="bold", color=LOCUS_COLORS["IGKV"])
    ax_b.text(3, 46, "Range 11–41\n(3.7×)", ha="center", fontsize=7.5, fontweight="bold", color=LOCUS_COLORS["IGLV"])
    ax_b.text(4, 54, "Mean 39.3\n(CV = 0.17)", ha="center", fontsize=7.5, fontweight="bold", color=LOCUS_COLORS["TRAV"])
    ax_b.text(5, 65, "Mean 52.8\n(CV = 0.10)", ha="center", fontsize=7.5, fontweight="bold", color=LOCUS_COLORS["TRBV"])

    # Contrast shading & non-overlapping text at bottom
    ax_b.axvspan(0.5, 3.5, color="#FADBD8", alpha=0.25, zorder=0)
    ax_b.axvspan(3.5, 5.5, color="#D4EFDF", alpha=0.25, zorder=0)
    ax_b.text(2.0, 3.0, "BCR Loci: Massive Expansion & Contraction", ha="center", fontsize=8.0, fontweight="bold", color="#78281F")
    ax_b.text(4.5, 3.0, "TCR Loci: Homeostasis (MHC-Restricted)", ha="center", fontsize=8.0, fontweight="bold", color="#145A32")

    ax_b.set_ylabel("Functional V-Gene Count per Species", fontsize=9.0, fontweight="bold")
    ax_b.set_ylim(-3, 86)
    ax_b.grid(True, axis="y", ls=":", alpha=0.5)
    ax_b.set_title("b   Cross-Species Copy Number Dynamics: BCR Volatility vs TCR Homeostasis",
                   loc="left", fontsize=10.0, fontweight="bold", pad=8)

    # -------------------------------------------------------------------------
    # Panel c: Segmental Status Breakdown & 147 Ortholog Groups Composition
    # -------------------------------------------------------------------------
    gs_c = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], hspace=0.40)
    ax_c1 = fig.add_subplot(gs_c[0, 0])
    ax_c2 = fig.add_subplot(gs_c[1, 0])

    # c1: Verified counts across 8 T2T: Functional 1,588, Pseudogene 1,738, ORF 345 (Total 3,671)
    t2t_tot_f = 1588
    t2t_tot_pse = 1738
    t2t_tot_orf = 345
    total_segments = 3671

    status_labels = ["Functional (F)", "Pseudogene (P)", "Open Reading Frame (ORF)"]
    status_counts = [t2t_tot_f, t2t_tot_pse, t2t_tot_orf]
    status_pcts = [c / total_segments * 100 for c in status_counts]
    status_colors = ["#2E86C1", "#E74C3C", "#7F8C8D"]

    bars_c1 = ax_c1.barh(range(3), status_counts, color=status_colors, edgecolor="#222222", lw=0.7, height=0.55)
    ax_c1.set_yticks(range(3))
    ax_c1.set_yticklabels(["Functional (F)", "Pseudogene (P)", "ORF"], fontsize=8.0, fontweight="bold")
    ax_c1.invert_yaxis()
    ax_c1.set_xlabel("Total Segments (8 T2T Catarrhine Assemblies)", fontsize=8.0, fontweight="bold")
    ax_c1.set_xlim(0, 2100)
    ax_c1.grid(True, axis="x", ls=":", alpha=0.5)
    for b, c, p in zip(bars_c1, status_counts, status_pcts):
        ax_c1.text(c + 30, b.get_y() + b.get_height()/2.0, f"{c:,} ({p:.1f}%)", va="center", ha="left",
                   fontsize=7.5, fontweight="bold", color="#222222")
    ax_c1.set_title("c   Segmental Status (n = 3,671 Total Segments; 1,588 F / 2,083 non-F)",
                    loc="left", fontsize=9.0, fontweight="bold", pad=4)

    # c2: Ortholog Group Architecture: Strict 1:1 vs Multi-copy by locus
    strict_by_locus = df_atlas.groupby(["locus", "is_strict_1to1"]).size().unstack(fill_value=0)
    strict_by_locus = strict_by_locus.reindex(["IGHV", "IGKV", "IGLV", "TRAV", "TRBV"])
    
    y_pos = np.arange(len(strict_by_locus))
    n_strict = strict_by_locus[1].values
    n_multi = strict_by_locus[0].values

    ax_c2.barh(y_pos, n_strict, color="#27AE60", edgecolor="#145A32", lw=0.6, height=0.52, label="Strict 1:1 (Single Copy, n=75)")
    ax_c2.barh(y_pos, n_multi, left=n_strict, color="#D35400", edgecolor="#7E5109", lw=0.6, height=0.52, label="Multi-copy Dynamic (n=72)")

    for i in range(len(strict_by_locus)):
        tot = n_strict[i] + n_multi[i]
        pct1 = n_strict[i] / tot * 100
        ax_c2.text(tot + 0.8, i, f"{tot} (1:1: {pct1:.0f}%)", va="center", ha="left", fontsize=7.5, fontweight="bold")

    ax_c2.set_yticks(y_pos)
    ax_c2.set_yticklabels(strict_by_locus.index, fontsize=8.0, fontweight="bold")
    ax_c2.invert_yaxis()
    ax_c2.set_xlabel("Ortholog Groups Count (147 Atlas Total)", fontsize=8.0, fontweight="bold")
    ax_c2.set_xlim(0, 72)
    ax_c2.grid(True, axis="x", ls=":", alpha=0.5)
    ax_c2.legend(loc="upper right", frameon=True, fontsize=6.8, framealpha=0.9)
    ax_c2.set_title("Ortholog Group Architecture (75 Strict 1:1 vs 72 Multi-Copy)", loc="left", fontsize=9.0, fontweight="bold", pad=4)

    plt.savefig(FIG1_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(FIG1_PDF, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved Figure 1:\n  PNG: {FIG1_PNG}\n  PDF: {FIG1_PDF}", flush=True)


# =============================================================================
# FIGURE 2: Paratope Diversification vs Surface Controls & Recombination
# =============================================================================
def plot_figure_2():
    print("[RUN] Generating Main Figure 2...", flush=True)
    df_m = pd.read_csv(MASTER_TSV, sep="\t")
    df_k1 = pd.read_csv(K1_TSV, sep="\t")
    df_atlas = pd.read_csv(ATLAS_TSV, sep="\t")
    df_part = pd.read_csv(PART_TSV, sep="\t")

    fig = plt.figure(figsize=(14, 11), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[0.95, 1.05], width_ratios=[1.1, 0.9],
                           hspace=0.34, wspace=0.26, left=0.06, right=0.97, top=0.96, bottom=0.06)

    # -------------------------------------------------------------------------
    # Panel a: IMGT 1..104 Structural Classification Scheme Schematic
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])

    regions = [
        ("FR1", 1, 26, "#2980B9", "Core FR (Scaffold)"),
        ("CDR1", 27, 38, "#C0392B", "Paratope Loop"),
        ("FR2", 39, 55, "#2980B9", "Core FR (Scaffold)"),
        ("CDR2", 56, 65, "#C0392B", "Paratope Loop"),
        ("FR3a", 66, 73, "#2980B9", "Core FR"),
        ("HV4", 74, 85, "#8E44AD", "Hypervariable 4"),
        ("FR3b", 86, 104, "#2980B9", "Core FR")
    ]

    ax_a.set_xlim(0, 105)
    ax_a.set_ylim(-2.2, 4.4)
    ax_a.axis("off")

    # Draw backbone track
    ax_a.plot([1, 104], [2.0, 2.0], color="#555555", lw=3.0, zorder=1)

    for name, start, end, color, desc in regions:
        width = end - start + 1
        rect = Rectangle((start - 0.5, 1.4), width, 1.2, facecolor=color, edgecolor="#222222", lw=0.9, alpha=0.88, zorder=2)
        ax_a.add_patch(rect)
        label_text = "CDR1" if name == "CDR1" else ("CDR2" if name == "CDR2" else ("HV4" if name == "HV4" else name))
        ax_a.text((start + end) / 2.0, 2.0, label_text, ha="center", va="center",
                  fontsize=8.0, fontweight="bold", color="white", zorder=3)

    # Triad callouts: Cys23, Trp41, Cys104
    triad = [(23, "Cys23"), (41, "Trp41"), (104, "Cys104")]
    for pos, label in triad:
        ax_a.plot(pos, 2.0, marker="D", color="#F1C40F", markersize=6.5, markeredgecolor="#222222", markeredgewidth=1.0, zorder=4)
        ax_a.annotate(f"Triad\n{label}", xy=(pos, 2.6), xytext=(pos, 3.5),
                      arrowprops=dict(arrowstyle="->", color="#B7950B", lw=1.0),
                      fontsize=7.5, fontweight="bold", color="#7D6608", ha="center")

    # Ticks along IMGT axis
    for p in range(10, 105, 10):
        ax_a.plot([p, p], [1.25, 1.35], color="#333333", lw=0.9)
        ax_a.text(p, 1.05, str(p), ha="center", va="top", fontsize=7.5, color="#333333")

    ax_a.text(52, 0.65, "IMGT Unique Codon Position (1 .. 104)", ha="center", va="top", fontsize=8.0, fontweight="bold", color="#333333")

    # Exposed-NC-FR definition callout box cleanly spaced below
    ax_a.annotate("Exposed Non-Contact Framework Control (Exposed-NC-FR)\nCriteria: Solvent Exposure ≥ 0.50 & Antigen Contact Freq < 0.05\n(Controls for mutational bias and solvent accessibility)",
                  xy=(52, 0.5), xytext=(52, -1.1),
                  arrowprops=dict(arrowstyle="->", color="#D35400", lw=1.2),
                  fontsize=8.0, fontweight="bold", color="#A04000", ha="center",
                  bbox=dict(boxstyle="round,pad=0.35", fc="#FEF9E7", ec="#F39C12", lw=0.9))

    ax_a.text(0, 4.3, "a   Structural Partitioning Scheme of Germline V-Segment (IMGT 1..104)",
              fontsize=10.0, fontweight="bold", va="top", ha="left")

    # -------------------------------------------------------------------------
    # Panel b: Codon Selection Comparison: CDR vs Core FR vs Exposed-NC-FR
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])

    units_data = [
        {"name": "IGHV MetaClans", "cdr": 1.132, "core": 0.458, "exp_nc": 0.509, "ratio": 2.23, "p": 0.0029},
        {"name": "IGHV Clan III", "cdr": 1.135, "core": 0.375, "exp_nc": 0.372, "ratio": 3.05, "p": 0.0032},
        {"name": "IGKV Light", "cdr": 0.993, "core": 0.490, "exp_nc": 0.490, "ratio": 2.03, "p": 0.0051},
        {"name": "TRAV Alpha", "cdr": 0.902, "core": 0.556, "exp_nc": 0.635, "ratio": 1.42, "p": 0.0221},
    ]

    x_b = np.arange(len(units_data))
    w_b = 0.25

    vals_cdr = [u["cdr"] for u in units_data]
    vals_core = [u["core"] for u in units_data]
    vals_exp = [u["exp_nc"] for u in units_data]

    ax_b.bar(x_b - w_b, vals_cdr, width=w_b, color="#C0392B", edgecolor="#7B241C", lw=0.7, label="CDR1 + CDR2 (Paratope)", zorder=3)
    ax_b.bar(x_b, vals_core, width=w_b, color="#2980B9", edgecolor="#1B4F72", lw=0.7, label="Core FR (Buried Scaffold)", zorder=3)
    ax_b.bar(x_b + w_b, vals_exp, width=w_b, color="#F39C12", edgecolor="#B9770E", lw=0.7, label="Exposed-NC-FR Control", zorder=3)

    ax_b.axhline(1.0, color="#7F8C8D", ls="--", lw=0.9, zorder=1)

    for i, u in enumerate(units_data):
        ax_b.text(i, max(u["cdr"], u["exp_nc"]) + 0.06, f"{u['ratio']:.2f}×\n(p={u['p']:.4f})",
                  ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#78281F")

    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels([u["name"] for u in units_data], fontsize=8.0, fontweight="bold")
    ax_b.set_ylabel(r"Median Codon Selection ($\omega = d_N/d_S$)", fontsize=8.5, fontweight="bold")
    ax_b.set_ylim(0, 1.48)
    ax_b.grid(True, axis="y", ls=":", alpha=0.5)
    ax_b.legend(loc="upper right", frameon=True, fontsize=7.5, framealpha=0.9)
    ax_b.set_title("b   Paratope vs Scaffold vs Exposed Non-Contact FR\n"
                   r"(Paratope acceleration robust against surface exposure controls)",
                   loc="left", fontsize=9.5, fontweight="bold", pad=8)

    # -------------------------------------------------------------------------
    # Panel c: Strict 1:1 Orthologs Gene-Level Distribution & FUBAR Enrichment
    # -------------------------------------------------------------------------
    gs_c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 0], width_ratios=[1.1, 0.9], wspace=0.32)
    ax_c1 = fig.add_subplot(gs_c[0, 0])
    ax_c2 = fig.add_subplot(gs_c[0, 1])

    # c1: Gene-level omega_CDR / omega_FR distribution across 75 strict 1:1 orthologs
    strict = df_atlas[df_atlas["is_strict_1to1"] == 1]
    ratios_1to1 = strict["cdr12_vs_fr_omega_ratio"].dropna().values

    bp_c = ax_c1.boxplot([ratios_1to1], patch_artist=True, widths=0.45,
                         medianprops=dict(color="#C0392B", lw=2.0), showfliers=False)
    bp_c["boxes"][0].set_facecolor("#D4E6F1")
    bp_c["boxes"][0].set_edgecolor("#2980B9")
    bp_c["boxes"][0].set_linewidth(1.3)

    np.random.seed(42)
    jitter = np.random.normal(0, 0.05, size=len(ratios_1to1))
    ax_c1.scatter(np.ones_like(ratios_1to1) + jitter, ratios_1to1, color="#2471A3", alpha=0.8,
                  s=32, edgecolor="white", lw=0.4, zorder=4)
    ax_c1.axhline(1.0, color="#7F8C8D", ls="--", lw=1.0)

    med_r = np.median(ratios_1to1)  # 1.22
    stat_w, p_w = sp_stats.wilcoxon(ratios_1to1 - 1.0)

    # Place text on top-left to avoid colliding with top points
    ax_c1.text(0.05, 0.94, f"Median = {med_r:.2f}×\n48/75 > 1.0 (64%)\nWilcoxon p = {p_w:.2e}",
               transform=ax_c1.transAxes, ha="left", va="top", fontsize=7.5, fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.3", fc="#FFFFFF", ec="#BDC3C7", lw=0.8, alpha=0.95))

    ax_c1.set_xticks([1])
    ax_c1.set_xticklabels(["Strict 1:1\n(n=75 genes)"], fontsize=8.0, fontweight="bold")
    ax_c1.set_ylabel(r"Gene Paratope Ratio ($\omega_{\mathrm{CDR}} / \omega_{\mathrm{FR}}$)", fontsize=8.5, fontweight="bold")
    ax_c1.set_ylim(0.4, 3.8)
    ax_c1.grid(True, axis="y", ls=":", alpha=0.5)
    ax_c1.set_title("c   Strict 1:1 Orthologs\nGene Ratio Distribution", loc="left", fontsize=9.0, fontweight="bold", pad=6)

    # c2: FUBAR Positively Selected Codons Enrichment (PP >= 0.90)
    pct_fubar_cdr = 24 / 925 * 100
    pct_fubar_fr = 61 / 5048 * 100

    bars_fubar = ax_c2.bar([0, 1], [pct_fubar_cdr, pct_fubar_fr], color=["#C0392B", "#2980B9"],
                           edgecolor="#222222", lw=0.8, width=0.55)
    ax_c2.set_xticks([0, 1])
    ax_c2.set_xticklabels(["CDR1+2\n(n=925)", "Core FR\n(n=5,048)"], fontsize=8.0, fontweight="bold")
    ax_c2.set_ylabel("FUBAR Sites (PP ≥ 0.90, %)", fontsize=8.5, fontweight="bold")
    ax_c2.set_ylim(0, 3.8)
    ax_c2.grid(True, axis="y", ls=":", alpha=0.5)

    ax_c2.text(0, pct_fubar_cdr + 0.10, "2.59%\n(24/925)", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax_c2.text(1, pct_fubar_fr + 0.10, "1.21%\n(61/5,048)", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # Place enrichment callout clearly above bars
    ax_c2.annotate("2.14× Enrichment\nFisher p = 0.0018", xy=(0.5, 3.3), xytext=(0.5, 3.3),
                   ha="center", va="center", fontsize=7.5, fontweight="bold", color="#78281F",
                   bbox=dict(boxstyle="round,pad=0.25", fc="#FDEDEC", ec="#E6B0AA", lw=0.8))
    ax_c2.set_title("FUBAR Positive Selection\nCodon Enrichment", loc="left", fontsize=9.0, fontweight="bold", pad=6)

    # -------------------------------------------------------------------------
    # Panel d: 5'/3' Partitioned Trees Recombination Sensitivity
    # -------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])

    merged_p = pd.merge(df_k1, df_part, left_on="dataset", right_on="unit")
    units = merged_p["dataset"].tolist()
    labels = [u.replace("_All", "").replace("_MetaClans", " Meta").replace("IGHV_", "").replace("Clan", "Clan ") for u in units]

    x_d = np.arange(len(units))
    r_std = merged_p["omega_median_ratio_cdr_vs_fr"].values
    r_part = merged_p["ratio_cdr_vs_fr_part_tree"].values

    ax_d.plot(x_d, r_std, marker="o", markersize=6.5, color="#8E44AD", lw=1.8, label="Standard Single Tree")
    ax_d.plot(x_d, r_part, marker="s", markersize=6.5, color="#16A085", lw=1.8, ls="--", label="5'/3' Partitioned Trees")

    # Offsets tailored to avoid line collisions
    text_y_offsets = [-0.11, 0.10, 0.10, -0.16, 0.10, -0.16, -0.11]
    for i in range(len(units)):
        dy = text_y_offsets[i]
        va = "bottom" if dy > 0 else "top"
        ax_d.text(x_d[i], r_part[i] + dy, f"{r_part[i]:.2f}×", ha="center", va=va, fontsize=7.2, fontweight="bold", color="#16A085")

    # Label TRBV Standard Tree cleanly above marker
    ax_d.text(x_d[6], r_std[6] + 0.08, f"{r_std[6]:.2f}×", ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#8E44AD")

    ax_d.axhline(1.0, color="#C0392B", ls=":", lw=1.2, label="Neutral Baseline (1.0×)")
    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.0, fontweight="bold")
    ax_d.set_ylabel(r"Paratope Diversification Ratio ($\omega_{\mathrm{CDR}} / \omega_{\mathrm{FR}}$)", fontsize=8.5, fontweight="bold")
    ax_d.grid(True, ls=":", alpha=0.5)
    ax_d.legend(loc="upper right", frameon=True, fontsize=7.5, framealpha=0.9)
    ax_d.set_ylim(0.8, 3.4)
    ax_d.set_title("d   5'/3' Structural Partitioning (IMGT 1–55 / 56–104)\n"
                   "(All 7 lineages retain significant paratope acceleration, P < 0.03)",
                   loc="left", fontsize=9.5, fontweight="bold", pad=8)

    plt.savefig(FIG2_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(FIG2_PDF, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved Figure 2:\n  PNG: {FIG2_PNG}\n  PDF: {FIG2_PDF}", flush=True)


# =============================================================================
# FIGURE 3: 3D Crystallographic Interface Alignment
# =============================================================================
def plot_figure_3():
    print("[RUN] Generating Main Figure 3...", flush=True)
    df_m = pd.read_csv(MASTER_TSV, sep="\t")
    df_k2 = pd.read_csv(K2_TSV, sep="\t")

    fig = plt.figure(figsize=(14, 11), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.1, 0.9],
                           hspace=0.34, wspace=0.26, left=0.06, right=0.97, top=0.96, bottom=0.06)

    # -------------------------------------------------------------------------
    # Panel a: IGHV_MetaClans Dual Track: Selection Pressure vs 3D Contact Freq
    # -------------------------------------------------------------------------
    gs_a = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 0], hspace=0.25)
    ax_a1 = fig.add_subplot(gs_a[0, 0])
    ax_a2 = fig.add_subplot(gs_a[1, 0])

    ighv = df_m[df_m["eval_unit"] == "IGHV_MetaClans"].sort_values("imgt_pos").copy()
    ighv["diff_beta_alpha"] = ighv["fubar_beta"] - ighv["fubar_alpha"]

    def shade_regions(ax):
        ax.axvspan(26.5, 38.5, color="#FADBD8", alpha=0.55, zorder=0)  # CDR1
        ax.axvspan(55.5, 65.5, color="#FADBD8", alpha=0.55, zorder=0)  # CDR2
        ax.axvspan(73.5, 85.5, color="#E8DAEF", alpha=0.50, zorder=0)  # HV4

    # Top track: 3D Antigen Contact Frequency
    shade_regions(ax_a1)
    ax_a1.plot(ighv["imgt_pos"], ighv["contact5_freq_combined"], color="#C0392B", lw=1.6,
               marker="s", markersize=3, label="Antigen Contact Freq (<5.0 Å, 719 PDBs)", zorder=3)
    ax_a1.set_ylabel("3D Antigen Contact Freq", color="#C0392B", fontsize=8.5, fontweight="bold")
    ax_a1.set_xlim(0.5, 104.5)
    ax_a1.set_ylim(-0.02, 1.02)
    ax_a1.set_xticks(range(10, 105, 10))
    ax_a1.set_xticklabels([])
    ax_a1.grid(True, axis="x", ls=":", alpha=0.5)
    ax_a1.legend(loc="upper left", frameon=True, fontsize=7.2, framealpha=0.9)

    # Region labels placed at y = 0.80, well above peak with clear headspace below axis top
    ax_a1.text(32.5, 0.80, "CDR1", ha="center", fontsize=8.0, fontweight="bold", color="#922B21")
    ax_a1.text(60.5, 0.80, "CDR2", ha="center", fontsize=8.0, fontweight="bold", color="#922B21")
    ax_a1.text(79.5, 0.80, "HV4", ha="center", fontsize=8.0, fontweight="bold", color="#6C3483")
    ax_a1.set_title("a   Evolutionary Selection Tracks 3D Antigen-Binding Interface (IGHV MetaClans)\n"
                    "Top: 3D Antigen Contact Freq (719 PDBs); Bottom: Primate Diversification (FUBAR β−α)",
                    loc="left", fontsize=9.5, fontweight="bold", pad=6)

    # Bottom track: Selection Parameter (FUBAR beta - alpha)
    shade_regions(ax_a2)
    ax_a2.plot(ighv["imgt_pos"], ighv["diff_beta_alpha"], color="#1F618D", lw=1.6,
               marker="o", markersize=3, label=r"Primate Diversification (FUBAR $\beta - \alpha$)", zorder=3)
    ax_a2.axhline(0, color="#7F8C8D", ls="--", lw=0.9, zorder=1)
    ax_a2.set_ylabel(r"Selection ($\beta - \alpha$)", color="#1F618D", fontsize=8.5, fontweight="bold")
    ax_a2.set_xlim(0.5, 104.5)
    ax_a2.set_ylim(-2.0, 7.6)
    ax_a2.set_xticks(range(10, 105, 10))
    ax_a2.set_xlabel("IMGT Unique Codon Position (1 .. 104)", fontsize=8.5, fontweight="bold")
    ax_a2.grid(True, axis="x", ls=":", alpha=0.5)
    ax_a2.legend(loc="upper right", frameon=True, fontsize=7.5)

    # -------------------------------------------------------------------------
    # Panel b: IGHV_MetaClans Contact vs Selection Scatter Plot & Partial Rho
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])

    cdr_mask = ighv["region"].isin(["CDR1", "CDR2"])
    exp_nc_mask = (~cdr_mask) & (ighv["contact5_freq_combined"] < 0.05) & (ighv["mean_surface_exposure_idx"] >= 0.50)
    core_mask = (~cdr_mask) & (~exp_nc_mask)

    ax_b.scatter(ighv.loc[core_mask, "contact5_freq_combined"], ighv.loc[core_mask, "diff_beta_alpha"],
                 color="#2980B9", alpha=0.75, s=40, edgecolors="#1B4F72", lw=0.6, label="Core FR (n=60)")
    ax_b.scatter(ighv.loc[exp_nc_mask, "contact5_freq_combined"], ighv.loc[exp_nc_mask, "diff_beta_alpha"],
                 color="#F39C12", alpha=0.85, s=45, edgecolors="#B9770E", lw=0.8, label="Exposed-NC-FR (n=20)")
    ax_b.scatter(ighv.loc[cdr_mask, "contact5_freq_combined"], ighv.loc[cdr_mask, "diff_beta_alpha"],
                 color="#C0392B", alpha=0.90, s=55, edgecolors="#7B241C", lw=1.0, label="CDR1+2 (n=18)")

    x_vals = ighv["contact5_freq_combined"].values
    y_vals = ighv["diff_beta_alpha"].values
    slope, intercept, r_val, p_val, _ = sp_stats.linregress(x_vals, y_vals)
    x_line = np.linspace(0, max(x_vals), 50)
    ax_b.plot(x_line, slope * x_line + intercept, color="#2C3E50", ls="--", lw=1.4, zorder=2)

    stat_box = (
        r"$\bf{Correlation\ with\ 3D\ Contacts:}$" "\n"
        r"Raw Spearman $\rho$ = 0.355 ($p$ = 3.35×10⁻⁴)" "\n"
        r"Partial $\rho$ = 0.365 ($p$ = 2.43×10⁻⁴)" "\n"
        r"$\it{(Controlling\ 3D\ Surface\ Exposure)}$"
    )
    ax_b.text(0.04, 0.94, stat_box, transform=ax_b.transAxes, va="top", ha="left",
              fontsize=7.8, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.35", fc="#FFFFFF", ec="#BDC3C7", lw=0.9, alpha=0.95))

    ax_b.set_xlabel("Antigen Contact Frequency (<5.0 Å)", fontsize=8.5, fontweight="bold")
    ax_b.set_ylabel(r"Codon Diversification (FUBAR $\beta - \alpha$)", fontsize=8.5, fontweight="bold")
    ax_b.set_ylim(-11, 8.6)
    ax_b.grid(True, ls=":", alpha=0.5)
    ax_b.legend(loc="lower right", frameon=True, fontsize=7.5)
    ax_b.set_title("b   Crystallographic Interface Correlation (IGHV MetaClans)\n"
                   "(Contact correlation strengthened after controlling surface exposure)",
                   loc="left", fontsize=9.5, fontweight="bold", pad=8)

    # -------------------------------------------------------------------------
    # Panel c: Light Chain Asymmetry: IGKV (CDR1-focused) vs IGLV (CDR2/HV4-dispersed)
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])

    igkv = df_m[df_m["eval_unit"] == "IGKV_All"].sort_values("imgt_pos").copy()
    iglv = df_m[df_m["eval_unit"] == "IGLV_All"].sort_values("imgt_pos").copy()

    regions_lc = ["CDR1\n(27–38)", "CDR2\n(56–65)", "HV4\n(74–85)", "Core FR\n(Scaffold)"]
    
    def get_reg_means(df_sub):
        c1 = df_sub[df_sub["region"] == "CDR1"]["contact5_freq_combined"].mean()
        c2 = df_sub[df_sub["region"] == "CDR2"]["contact5_freq_combined"].mean()
        hv4 = df_sub[(df_sub["imgt_pos"] >= 74) & (df_sub["imgt_pos"] <= 85)]["contact5_freq_combined"].mean()
        fr = df_sub[(~df_sub["region"].isin(["CDR1", "CDR2"])) & (~((df_sub["imgt_pos"] >= 74) & (df_sub["imgt_pos"] <= 85)))]["contact5_freq_combined"].mean()
        return [c1, c2, hv4, fr]

    means_k = get_reg_means(igkv)
    means_l = get_reg_means(iglv)

    x_c = np.arange(len(regions_lc))
    w_c = 0.35

    bars_k = ax_c.bar(x_c - w_c/2, means_k, width=w_c, color="#E67E22", edgecolor="#A04000", lw=0.8,
                      label="IGK (Kappa Light Chain, n=244)")
    bars_l = ax_c.bar(x_c + w_c/2, means_l, width=w_c, color="#16A085", edgecolor="#0E6251", lw=0.8,
                      label="IGL (Lambda Light Chain, n=176)")

    for b, v in zip(bars_k, means_k):
        ax_c.text(b.get_x() + b.get_width()/2, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=7.2, fontweight="bold")
    for b, v in zip(bars_l, means_l):
        ax_c.text(b.get_x() + b.get_width()/2, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=7.2, fontweight="bold")

    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(regions_lc, fontsize=8.0, fontweight="bold")
    ax_c.set_ylabel("Mean Antigen Contact Frequency (<5.0 Å)", fontsize=8.5, fontweight="bold")
    ax_c.set_ylim(0, 0.55)
    ax_c.grid(True, axis="y", ls=":", alpha=0.5)
    ax_c.legend(loc="upper right", frameon=True, fontsize=7.5)

    ax_c.annotate("IGK: CDR1-concentrated\n(0.440 vs 0.138; 3.2×)", xy=(0 - w_c/2, 0.44), xytext=(0.3, 0.48),
                  arrowprops=dict(arrowstyle="->", color="#A04000", lw=1.0),
                  fontsize=7.5, fontweight="bold", color="#A04000")
    ax_c.annotate("IGL: CDR2/HV4-dispersed\n(CDR1: 0.346, CDR2: 0.280)", xy=(1 + w_c/2, 0.28), xytext=(1.4, 0.38),
                  arrowprops=dict(arrowstyle="->", color="#0E6251", lw=1.0),
                  fontsize=7.5, fontweight="bold", color="#0E6251")

    ax_c.set_title("c   Light Chain Structural Asymmetry: IGK Specialization vs IGL Dispersion",
                   loc="left", fontsize=9.5, fontweight="bold", pad=8)

    # -------------------------------------------------------------------------
    # Panel d: TCR Interface Decomposition: Peptide vs MHC Footprints
    # -------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])

    categories = [
        "TRBV vs Peptide\n(CDR2 Footprint)",
        "TRBV vs MHC-Helix\n(Conserved)",
        "TRAV vs MHC-Helix\n(CDR1 Footprint)",
        "TRAV vs Peptide\n(Variable)"
    ]
    corrs = [0.320, 0.158, 0.285, 0.104]
    p_values = [0.0019, 0.1200, 0.0059, 0.3200]
    bar_colors = ["#27AE60", "#A9DFBF", "#8E44AD", "#D7BDE2"]

    x_d = np.arange(len(categories))
    bars_d = ax_d.bar(x_d, corrs, color=bar_colors, edgecolor="#222222", lw=0.8, width=0.55)

    for b, c, p in zip(bars_d, corrs, p_values):
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        ax_d.text(b.get_x() + b.get_width()/2, c + 0.015, f"{c:.3f}\n({sig})",
                  ha="center", va="bottom", fontsize=7.2, fontweight="bold")

    ax_d.axhline(0, color="#222222", lw=0.8)
    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels(categories, fontsize=7.5, fontweight="bold")
    ax_d.set_ylabel(r"Correlation ($\rho$) with TCR 3D Contacts", fontsize=8.5, fontweight="bold")
    ax_d.set_ylim(0, 0.42)
    ax_d.grid(True, axis="y", ls=":", alpha=0.5)

    ax_d.text(0.04, 0.94, "516 TCR 3D Chains\n(260 TRBV, 256 TRAV)", transform=ax_d.transAxes,
              va="top", ha="left", fontsize=7.5, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.3", fc="#FFFFFF", ec="#BDC3C7", lw=0.8))

    ax_d.set_title("d   TCR Diagonal Docking: Genetic Separation of Binding Footprints\n"
                   "(TRBV focuses on foreign peptide; TRAV tracks conserved MHC α-helices)",
                   loc="left", fontsize=9.5, fontweight="bold", pad=8)

    plt.savefig(FIG3_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(FIG3_PDF, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved Figure 3:\n  PNG: {FIG3_PNG}\n  PDF: {FIG3_PDF}", flush=True)


# =============================================================================
# FIGURE 4: Public Antiviral Archetypes & Peripheral Tolerance
# =============================================================================
def plot_figure_4():
    print("[RUN] Generating Main Figure 4...", flush=True)
    df_atlas = pd.read_csv(ATLAS_TSV, sep="\t")
    df_m = pd.read_csv(MASTER_TSV, sep="\t")
    df_vane = pd.read_csv(VANE_TSV, sep="\t")

    fig = plt.figure(figsize=(14, 11), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.05, 0.95],
                           hspace=0.34, wspace=0.26, left=0.06, right=0.97, top=0.96, bottom=0.06)

    # -------------------------------------------------------------------------
    # Panel a: IGHV1-69 Kabat 54 (IMGT 62) Primate Sequence Alignment & Selection
    # -------------------------------------------------------------------------
    gs_a = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 0], height_ratios=[1.1, 0.9], hspace=0.30)
    ax_a1 = fig.add_subplot(gs_a[0, 0])
    ax_a2 = fig.add_subplot(gs_a[1, 0])

    sub_c1 = df_m[df_m["eval_unit"] == "IGHV_ClanI"].sort_values("imgt_pos").copy()
    sub_c1["diff_beta_alpha"] = sub_c1["fubar_beta"] - sub_c1["fubar_alpha"]
    focus = sub_c1[(sub_c1["imgt_pos"] >= 50) & (sub_c1["imgt_pos"] <= 70)]

    ax_a1.bar(focus["imgt_pos"], focus["contact5_freq_combined"], width=0.75, color="#1F618D", alpha=0.45,
              edgecolor="#154360", lw=0.6, label="Viral Antigen Contact Freq")
    ax_a1.set_ylabel("Antigen Contact Freq", color="#154360", fontsize=8.0, fontweight="bold")
    ax_a1.tick_params(axis="y", labelcolor="#154360")
    ax_a1.set_ylim(0, 0.95)

    ax_a1_twin = ax_a1.twinx()
    ax_a1_twin.plot(focus["imgt_pos"], focus["diff_beta_alpha"], color="#C0392B", lw=1.8, marker="o", markersize=4,
                    label=r"Diversification ($\beta - \alpha$)")
    ax_a1_twin.axhline(0, color="#7F8C8D", ls="--", lw=0.8)
    ax_a1_twin.set_ylabel(r"Selection ($\beta - \alpha$)", color="#C0392B", fontsize=8.0, fontweight="bold")
    ax_a1_twin.tick_params(axis="y", labelcolor="#C0392B")
    ax_a1_twin.set_ylim(-2.0, 7.2)

    ax_a1.axvspan(55.5, 65.5, color="#FADBD8", alpha=0.35, zorder=0)  # CDR2
    
    # Explicit Kabat 54 (IMGT 62) callout
    ax_a1.annotate(r"$\bf{Kabat\ 54\ (IMGT\ 62)}$" "\n" r"$\omega = 2.32,\ \beta-\alpha = +1.70$",
                   xy=(62, 0.53), xytext=(66.2, 0.81),
                   arrowprops=dict(facecolor="#C0392B", edgecolor="#7B241C", shrink=0.08, width=1.2, headwidth=5),
                   fontsize=7.8, fontweight="bold", color="#78281F", ha="center",
                   bbox=dict(boxstyle="round,pad=0.25", fc="#FDEDEC", ec="#E6B0AA", lw=0.8))

    ax_a1.set_xlim(49.5, 70.5)
    ax_a1.set_xticks(range(50, 71, 2))
    ax_a1.set_xlabel("IMGT Unique Codon Position in CDR2 Region (50 .. 70)", fontsize=8.0, fontweight="bold")
    ax_a1.set_title("a   Public Antiviral Archetype: IGHV1-69 Hydrophobic Tip (Kabat 54 / IMGT 62)\n"
                    "Viral Paratope Contact Tracks Primate Codon Diversification",
                    loc="left", fontsize=9.5, fontweight="bold", pad=6)

    # a2: Alignment snippet of Kabat 54 across primates
    ax_a2.set_xlim(0, 10)
    ax_a2.set_ylim(0, 7)
    ax_a2.axis("off")

    align_data = [
        ("Human (T2T-CHM13)", "I  I  P  I", "F", "G  T  A  N", "#C0392B", "Hydrophobic tip (stem neutralizer)"),
        ("Human (Polymorphic)", "I  I  P  I", "L", "G  T  A  N", "#2E86C1", "F54/L54 population dimorphism"),
        ("Chimpanzee / Bonobo", "I  I  P  I", "L", "G  T  A  N", "#7F8C8D", "Ancestral ape Leu54"),
        ("Gorilla", "I  I  P  I", "L", "G  T  A  N", "#7F8C8D", "Ancestral ape Leu54"),
        ("Orangutan / Siamang", "I  I  P  I", "L", "G  T  A  N", "#7F8C8D", "Ancestral ape Leu54"),
        ("Rhesus macaque", "I  I  P  I", "V", "G  T  A  N", "#7F8C8D", "Old World monkey Val54 / Leu54")
    ]

    y_start = 5.8
    y_step = 0.95
    for i, (sp_name, flank1, k54, flank2, k_col, note) in enumerate(align_data):
        y = y_start - i * y_step
        ax_a2.text(0.1, y, sp_name, fontsize=7.5, fontweight="bold", color="#333333", va="center")
        ax_a2.text(3.4, y, flank1, fontsize=8.0, fontfamily="monospace", color="#555555", va="center")
        ax_a2.text(4.7, y, k54, fontsize=8.5, fontfamily="monospace", fontweight="bold", color=k_col, va="center",
                   bbox=dict(boxstyle="square,pad=0.2", fc="#FCF3CF" if k54=="F" else "#EAECEE", ec=k_col, lw=1.0))
        ax_a2.text(5.2, y, flank2, fontsize=8.0, fontfamily="monospace", color="#555555", va="center")
        ax_a2.text(7.0, y, note, fontsize=7.2, color="#444444", va="center")

    ax_a2.text(4.7, 6.6, "Kabat 54", fontsize=7.5, fontweight="bold", color="#C0392B", ha="center")
    ax_a2.text(0.1, 6.6, "Species / Assembly", fontsize=7.5, fontweight="bold", color="#333333")
    ax_a2.text(7.0, 6.6, "Functional Impact", fontsize=7.5, fontweight="bold", color="#333333")

    # -------------------------------------------------------------------------
    # Panel b: 147 Ortholog Groups: Public Expansion vs Primate omega Ratio
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])

    sub_b = df_atlas.dropna(subset=["log2_public_over_olga_pgen", "cdr12_vs_fr_omega_ratio"]).copy()

    strict_grp = sub_b[sub_b["is_strict_1to1"] == 1]
    multi_grp = sub_b[sub_b["is_strict_1to1"] == 0]

    for loc in ["IGHV", "TRBV", "TRAV", "IGKV", "IGLV"]:
        col = LOCUS_COLORS[loc]
        s_loc = strict_grp[strict_grp["locus"] == loc]
        if len(s_loc) > 0:
            ax_b.scatter(s_loc["cdr12_vs_fr_omega_ratio"], s_loc["log2_public_over_olga_pgen"],
                         color=col, marker="o", s=50, alpha=0.85, edgecolors="#1B2631", lw=0.8,
                         label=f"{loc} Strict 1:1" if loc in ["IGHV", "TRBV"] else "")
        m_loc = multi_grp[multi_grp["locus"] == loc]
        if len(m_loc) > 0:
            ax_b.scatter(m_loc["cdr12_vs_fr_omega_ratio"], m_loc["log2_public_over_olga_pgen"],
                         color=col, marker="^", s=55, alpha=0.85, edgecolors="#1B2631", lw=0.8,
                         label=f"{loc} Multi-Copy" if loc in ["IGHV", "TRBV"] else "")

    x_b_vals = sub_b["cdr12_vs_fr_omega_ratio"].astype(float).values
    y_b_vals = sub_b["log2_public_over_olga_pgen"].astype(float).values
    slope_b, int_b, r_b, p_b, _ = sp_stats.linregress(x_b_vals, y_b_vals)
    rho_b, prho_b = sp_stats.spearmanr(x_b_vals, y_b_vals)

    x_line_b = np.linspace(x_b_vals.min(), x_b_vals.max(), 100)
    ax_b.plot(x_line_b, slope_b * x_line_b + int_b, color="#2C3E50", ls="--", lw=1.4, zorder=2)

    # Archetype badges
    archetypes_b = [
        ("IGHV1-69", "Multi-copy (▲)", (25, -18)),
        ("IGHV4-34", "Strict 1:1 (●)", (-35, -22)),
        ("TRBV5-4", "Strict 1:1 (●)", (20, -18)),
        ("TRBV19", "Strict 1:1 (●)", (-45, 18)),
        ("TRBV6-5", "Multi-copy (▲)", (-40, 16)),
    ]

    for arch, status_note, (dx, dy) in archetypes_b:
        row = sub_b[sub_b["human_gene"] == arch]
        if len(row) > 0:
            rx = float(row["cdr12_vs_fr_omega_ratio"].iloc[0])
            ry = float(row["log2_public_over_olga_pgen"].iloc[0])
            ax_b.annotate(
                f"{arch}\n[{status_note}]", xy=(rx, ry), xytext=(dx, dy), textcoords="offset points",
                fontsize=7.2, fontweight="bold", color="#17202A",
                bbox=dict(boxstyle="round,pad=0.2", fc="#FCF3CF", ec="#F39C12", lw=0.8, alpha=0.95),
                arrowprops=dict(arrowstyle="->", color="#34495E", lw=0.9, shrinkB=3)
            )

    stat_text_b = (
        r"$\bf{Repertoire\ Mobilization:}$" "\n"
        f"Spearman $\\rho$ = {rho_b:.3f} ($P$ = {prho_b:.2f}, n={len(sub_b)})\n"
        r"Partial $\rho$ ($P_{\mathrm{gen}}$) = 0.137 ($P$ = 0.107)" "\n"
        r"IGLV Locus: $\rho$ = 0.615 ($P$ = 0.0050)"
    )
    ax_b.text(0.97, 0.96, stat_text_b, transform=ax_b.transAxes, va="top", ha="right",
              fontsize=7.2, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.25", fc="#FFFFFF", ec="#BDC3C7", lw=0.8, alpha=0.92))

    ax_b.set_xlabel(r"Primate Germline Paratope Ratio ($\omega_{\mathrm{CDR}} / \omega_{\mathrm{FR}}$)", fontsize=8.5, fontweight="bold")
    ax_b.set_ylabel(r"Public Repertoire Expansion ($\log_2[\mathrm{Observed} / P_{\mathrm{gen}}]$)", fontsize=8.5, fontweight="bold")
    ax_b.grid(True, ls=":", alpha=0.5)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Strict 1:1 (75 genes)', markerfacecolor='#2C3E50', markersize=7),
        Line2D([0], [0], marker='^', color='w', label='Multi-copy (72 genes)', markerfacecolor='#2C3E50', markersize=8),
        Line2D([0], [0], marker='s', color='w', label='IGHV', markerfacecolor=LOCUS_COLORS["IGHV"], markersize=7),
        Line2D([0], [0], marker='s', color='w', label='TRBV', markerfacecolor=LOCUS_COLORS["TRBV"], markersize=7),
        Line2D([0], [0], marker='s', color='w', label='TRAV', markerfacecolor=LOCUS_COLORS["TRAV"], markersize=7),
    ]
    ax_b.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=7.2)
    ax_b.set_title("b   Public TCR Mobilization vs Germline Selection",
                   loc="left", fontsize=9.5, fontweight="bold", pad=8)

    # -------------------------------------------------------------------------
    # Panel c: Peripheral B-Cell Tolerance & IGHV4-34 in Autoimmunity (vane cohort)
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, :])

    vane_434 = df_vane[(df_vane["v_gene"] == "IGHV4-34") & (df_vane["locus"] == "IGH")].copy()

    hd_n = vane_434[(vane_434["cohort"] == "HD") & (vane_434["subset"] == "N")]["freq_seq"].values * 100
    hd_m = vane_434[(vane_434["cohort"] == "HD") & (vane_434["subset"] == "M")]["freq_seq"].values * 100
    achr_m = vane_434[(vane_434["cohort"] == "AChR_MG") & (vane_434["subset"] == "M")]["freq_seq"].values * 100
    musk_m = vane_434[(vane_434["cohort"] == "MuSK_MG") & (vane_434["subset"] == "M")]["freq_seq"].values * 100
    pooled_mg_m = np.concatenate([achr_m, musk_m])

    plot_data = [hd_n, hd_m, achr_m, musk_m, pooled_mg_m]
    labels_c = [
        "Healthy Donors\nNaive B (n=4)",
        "Healthy Donors\nMemory B (n=4)",
        "AChR-MG\nMemory B (n=4)",
        "MuSK-MG\nMemory B (n=5)",
        "Pooled MG\nMemory B (n=9)"
    ]
    colors_c = ["#2980B9", "#27AE60", "#E67E22", "#8E44AD", "#C0392B"]

    bp_c = ax_c.boxplot(plot_data, tick_labels=labels_c, patch_artist=True, widths=0.48,
                        medianprops=dict(color="#111111", lw=1.8), showfliers=False)

    for patch, col in zip(bp_c["boxes"], colors_c):
        patch.set_facecolor(col)
        patch.set_alpha(0.35)
        patch.set_edgecolor(col)
        patch.set_linewidth(1.3)

    np.random.seed(42)
    for i, (vals, col) in enumerate(zip(plot_data, colors_c)):
        jitter = np.random.normal(0, 0.04, size=len(vals))
        ax_c.scatter(np.full_like(vals, i + 1) + jitter, vals, color=col, s=60,
                     edgecolor="#1B2631", lw=0.9, zorder=4)

    # Paired HD connecting lines with donor labels outside at x = 0.62
    hd_df = vane_434[vane_434["cohort"] == "HD"].pivot(index="donor", columns="subset", values="freq_seq") * 100
    for donor, row in hd_df.iterrows():
        ax_c.plot([1, 2], [row["N"], row["M"]], color="#7F8C8D", ls=":", lw=1.1, alpha=0.8, zorder=2)
        ax_c.text(0.62, row["N"], donor, ha="right", va="center", fontsize=7.0, color="#444444", fontweight="bold")

    means = [np.mean(v) for v in plot_data]
    sds = [np.std(v, ddof=1) for v in plot_data]
    for i, (m, s) in enumerate(zip(means, sds)):
        ax_c.text(i + 1, 0.35, f"Mean:\n{m:.2f}±{s:.2f}%", ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#333333")

    # Brackets
    ax_c.plot([1, 1, 2, 2], [6.6, 6.8, 6.8, 6.6], color="#2980B9", lw=1.2)
    ax_c.text(1.5, 6.9, "Peripheral Purge (-40%)\nMWU P = 0.0571 (Paired Wilcoxon P = 0.125)",
              ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#1B4F72")

    ax_c.plot([2, 2, 5, 5], [7.6, 7.8, 7.8, 7.6], color="#C0392B", lw=1.2)
    ax_c.text(3.5, 7.9, "HD Memory vs Pooled MG Memory: MWU U = 12.0, P = 0.4140 (n.s.)\n(High inter-individual variance eclipses difference: MG IQR 2.12%, range 1.43%–6.16%)",
              ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#78281F")

    ax_c.axvspan(2.5, 5.5, color="#FADBD8", alpha=0.20, zorder=0)
    ax_c.text(4.0, 9.4, "Myasthenia Gravis Autoimmune Cohort (PRJNA338795; Vander Heiden et al. 2017)",
              ha="center", va="top", fontsize=8.0, fontweight="bold", color="#78281F")

    ax_c.set_xlim(0.4, 5.6)
    ax_c.set_ylabel("IGHV4-34 Frequency in B-Cell Repertoire (%)", fontsize=9.0, fontweight="bold")
    ax_c.set_ylim(0, 9.8)
    ax_c.grid(True, axis="y", ls=":", alpha=0.5)
    ax_c.set_title("c   IGHV4-34 Peripheral Tolerance & MG Memory Repertoire (n = 14)",
                   loc="left", fontsize=10.0, fontweight="bold", pad=8)

    plt.savefig(FIG4_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(FIG4_PDF, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved Figure 4:\n  PNG: {FIG4_PNG}\n  PDF: {FIG4_PDF}", flush=True)


if __name__ == "__main__":
    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()
    print("[ALL DONE] Nature Main Figures 1–4 generated successfully.", flush=True)
