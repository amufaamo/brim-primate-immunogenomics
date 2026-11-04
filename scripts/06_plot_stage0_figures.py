#!/usr/bin/env python3
"""
06_plot_stage0_figures.py
Publication-Quality Multi-Panel Summary Figure for Stage 0 Evaluation
T2T Primate Germline V-Segment Adaptive Evolution & Structural Repertoire Integration

Panels:
- Panel A: Dual-track IMGT 1..104 selection & 3D contact profile (IGHV_MetaClans & TRBV_All)
- Panel B: Criterion K1: CDR1+CDR2 vs Core FR selection (Median omega & % MEME p<0.05 sites)
- Panel C: Criterion K2: Primate omega vs 3D Contact Frequency (Raw & Partial Spearman rho)
- Panel D: Criterion K3: Human Public Repertoire Recruitment vs Primate Germline Selection (Archetypes)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy import stats as sp_stats

# Paths
ROOT = os.environ.get("BRIM_DATA_DIR", os.environ.get("BRIM_DATA_DIR", "./data"))
KILL_DIR = os.path.join(ROOT, "04_structure_repertoire", "kill_eval")
SEL_DIR = os.path.join(ROOT, "03_selection")
PART_DIR = os.path.join(SEL_DIR, "recombination_sensitivity")

OUT_PNG = os.path.join(ROOT, "04_structure_repertoire", "kill_eval", "stage0_summary_fig.png")
OUT_PDF = os.path.join(ROOT, "04_structure_repertoire", "kill_eval", "stage0_summary_fig.pdf")

# Publication style
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["xtick.major.width"] = 0.8
plt.rcParams["ytick.major.width"] = 0.8
plt.rcParams["xtick.color"] = "#333333"
plt.rcParams["ytick.color"] = "#333333"


def plot_stage0_summary():
    # -------------------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------------------
    master_file = os.path.join(KILL_DIR, "master_imgt104_selection_and_3d_contacts.tsv")
    k1_file = os.path.join(KILL_DIR, "k1_cdr_vs_fr_selection_summary.tsv")
    part_file = os.path.join(PART_DIR, "partitioned_tree_summary.tsv")
    k2_file = os.path.join(KILL_DIR, "k2_3d_contact_correlations.tsv")
    k3_tests_file = os.path.join(KILL_DIR, "k3_public_repertoire_tests.tsv")
    k3_atlas_file = os.path.join(KILL_DIR, "k3_gene_level_evolution_repertoire_integrated.tsv")

    print("[INFO] Loading datasets for plotting...")
    df_master = pd.read_csv(master_file, sep="\t")
    df_k1 = pd.read_csv(k1_file, sep="\t")
    df_part = pd.read_csv(part_file, sep="\t") if os.path.exists(part_file) else None
    df_k2 = pd.read_csv(k2_file, sep="\t")
    df_k3_tests = pd.read_csv(k3_tests_file, sep="\t")
    df_k3_atlas = pd.read_csv(k3_atlas_file, sep="\t")

    # Figure Setup: 2x2 major layout
    fig = plt.figure(figsize=(18, 14), dpi=300)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28,
                           left=0.06, right=0.96, top=0.95, bottom=0.06)

    # =========================================================================
    # Panel A: Dual-Track IMGT 1..104 Selection & 3D Crystal Contacts
    # Subdivide top-left grid into 2 stacked tracks (A1: IGHV, A2: TRBV)
    # =========================================================================
    gs_a = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 0], hspace=0.35)
    ax_a1 = fig.add_subplot(gs_a[0, 0])
    ax_a2 = fig.add_subplot(gs_a[1, 0])

    # Region shading helper
    def shade_imgt_regions(ax):
        # CDR1: 27..38 (red/orange)
        ax.axvspan(26.5, 38.5, color="#FADBD8", alpha=0.55, zorder=0)
        # CDR2: 56..65 (red/orange)
        ax.axvspan(55.5, 65.5, color="#FADBD8", alpha=0.55, zorder=0)
        # HV4: 74..85 (lavender/purple)
        ax.axvspan(73.5, 85.5, color="#E8DAEF", alpha=0.50, zorder=0)

    # --- Track A1: IGHV MetaClans ---
    df_ighv = df_master[df_master["eval_unit"] == "IGHV_MetaClans"].sort_values("imgt_pos")
    shade_imgt_regions(ax_a1)

    # Left Y: FUBAR Omega
    color_om = "#1F618D"
    ax_a1.plot(df_ighv["imgt_pos"], df_ighv["fubar_omega"], color=color_om, lw=1.6,
               marker="o", markersize=3, label=r"Primate dN/dS (FUBAR $\omega$)", zorder=3)
    ax_a1.set_ylabel(r"Primate $\omega$ ($dN/dS$)", color=color_om, fontsize=10, fontweight="bold")
    ax_a1.tick_params(axis="y", labelcolor=color_om, labelsize=9)
    ax_a1.set_xlim(0.5, 104.5)
    ax_a1.set_ylim(0, max(2.5, df_ighv["fubar_omega"].max() * 1.15))
    ax_a1.axhline(1.0, color="#7F8C8D", ls="--", lw=0.9, alpha=0.8, zorder=1)

    # Right Y: 3D Antigen Contact Frequency (<5 A)
    ax_a1_r = ax_a1.twinx()
    color_ct = "#C0392B"
    ax_a1_r.plot(df_ighv["imgt_pos"], df_ighv["contact5_freq_combined"], color=color_ct, lw=1.5,
                 ls="-", marker="s", markersize=2.5, label="Empirical Ag Contact Freq (<5 Å)", zorder=2)
    ax_a1_r.set_ylabel("3D Antigen Contact Freq", color=color_ct, fontsize=10, fontweight="bold")
    ax_a1_r.tick_params(axis="y", labelcolor=color_ct, labelsize=9)
    ax_a1_r.set_ylim(0, 1.0)

    # Annotate regions in track A1
    ax_a1.text(32.5, ax_a1.get_ylim()[1]*0.88, "CDR1", ha="center", va="center", fontsize=9, fontweight="bold", color="#922B21")
    ax_a1.text(60.5, ax_a1.get_ylim()[1]*0.88, "CDR2", ha="center", va="center", fontsize=9, fontweight="bold", color="#922B21")
    ax_a1.text(79.5, ax_a1.get_ylim()[1]*0.88, "HV4", ha="center", va="center", fontsize=9, fontweight="bold", color="#6C3483")
    ax_a1.set_title("A   Primate Germline Positive Selection vs Empirical 3D Contacts (IMGT 1..104)\n"
                    "IGHV MetaClans (n=329 unique functional V-genes across 8 T2T primates)",
                    loc="left", fontsize=11, fontweight="bold", pad=8)
    ax_a1.set_xlabel("")
    ax_a1.set_xticks(range(10, 105, 10))

    # --- Track A2: TRBV All ---
    df_trbv = df_master[df_master["eval_unit"] == "TRBV_All"].sort_values("imgt_pos")
    shade_imgt_regions(ax_a2)

    color_trb = "#2E7D32"
    ax_a2.plot(df_trbv["imgt_pos"], df_trbv["fubar_omega"], color=color_om, lw=1.6,
               marker="o", markersize=3, label=r"Primate $\omega$", zorder=3)
    ax_a2.set_ylabel(r"Primate $\omega$ ($dN/dS$)", color=color_om, fontsize=10, fontweight="bold")
    ax_a2.tick_params(axis="y", labelcolor=color_om, labelsize=9)
    ax_a2.set_xlim(0.5, 104.5)
    ax_a2.set_ylim(0, max(2.5, df_trbv["fubar_omega"].max() * 1.15))
    ax_a2.axhline(1.0, color="#7F8C8D", ls="--", lw=0.9, alpha=0.8, zorder=1)

    ax_a2_r = ax_a2.twinx()
    ax_a2_r.plot(df_trbv["imgt_pos"], df_trbv["contact5_pmhc_freq"], color=color_trb, lw=1.5,
                 ls="-", marker="^", markersize=2.5, label="pMHC Contact Freq (<5 Å)", zorder=2)
    ax_a2_r.set_ylabel("3D pMHC Contact Freq", color=color_trb, fontsize=10, fontweight="bold")
    ax_a2_r.tick_params(axis="y", labelcolor=color_trb, labelsize=9)
    ax_a2_r.set_ylim(0, 1.0)

    ax_a2.text(32.5, ax_a2.get_ylim()[1]*0.88, "CDR1", ha="center", va="center", fontsize=9, fontweight="bold", color="#922B21")
    ax_a2.text(60.5, ax_a2.get_ylim()[1]*0.88, "CDR2", ha="center", va="center", fontsize=9, fontweight="bold", color="#922B21")
    ax_a2.text(79.5, ax_a2.get_ylim()[1]*0.88, "HV4", ha="center", va="center", fontsize=9, fontweight="bold", color="#6C3483")
    ax_a2.set_title("TRBV All (n=383 unique functional V-genes across 8 T2T primates)",
                    loc="left", fontsize=10, fontweight="bold", pad=6)
    ax_a2.set_xlabel("IMGT Core V-Segment Position (1..104)", fontsize=10, fontweight="bold")
    ax_a2.set_xticks(range(10, 105, 10))

    # =========================================================================
    # Panel B: Criterion K1: CDR1+CDR2 vs Core FR Positive Selection
    # Paired Bar Chart: Median omega & % MEME p<0.05
    # =========================================================================
    ax_b = fig.add_subplot(gs[0, 1])

    units_to_show = [
        "IGHV_ClanI", "IGHV_ClanII", "IGHV_ClanIII",
        "IGKV_All", "IGLV_All", "TRAV_All", "TRBV_All",
        "Strict1to1_AllLoci"
    ]
    sub_k1 = df_k1[df_k1["dataset"].isin(units_to_show)].copy()
    sub_k1["order"] = sub_k1["dataset"].apply(lambda x: units_to_show.index(x) if x in units_to_show else 99)
    sub_k1 = sub_k1.sort_values("order").reset_index(drop=True)

    y_pos = np.arange(len(sub_k1))
    bar_width = 0.38

    bars_cdr = ax_b.barh(y_pos - bar_width/2, sub_k1["median_omega_cdr12"], height=bar_width,
                         color="#C0392B", alpha=0.88, label="CDR1 + CDR2 (Paratope)", zorder=3)
    bars_fr = ax_b.barh(y_pos + bar_width/2, sub_k1["median_omega_core_fr"], height=bar_width,
                        color="#2874A6", alpha=0.88, label="Core Framework (Scaffold)", zorder=3)

    ax_b.set_yticks(y_pos)
    clean_labels = [
        d.replace("_All", "").replace("IGHV_", "").replace("Strict1to1_AllLoci", "Strict 1:1 (All Loci)")
        for d in sub_k1["dataset"]
    ]
    ax_b.set_yticklabels(clean_labels, fontsize=10, fontweight="bold")
    ax_b.invert_yaxis()
    ax_b.set_xlabel(r"Primate Germline Median $\omega$ ($dN/dS$)", fontsize=10, fontweight="bold")
    ax_b.axvline(1.0, color="#7F8C8D", ls="--", lw=0.9, alpha=0.8, zorder=1)
    ax_b.grid(axis="x", ls=":", alpha=0.5, zorder=0)
    ax_b.set_xlim(0, max(sub_k1["median_omega_cdr12"].max() * 1.35, 1.8))

    for idx, row in sub_k1.iterrows():
        ratio = row["omega_median_ratio_cdr_vs_fr"]
        p_val = row["mwu_p_fubar_omega"]
        p_txt = f"$p$={p_val:.1e}" if p_val < 0.001 else f"$p$={p_val:.3f}"
        if p_val < 1e-6:
            p_txt = "$p$<10⁻⁶"
        txt = f" Ratio: {ratio:.1f}× ({p_txt})"
        ax_b.text(row["median_omega_cdr12"] + 0.02, idx - bar_width/2, txt,
                  va="center", ha="left", fontsize=8.5, fontweight="bold", color="#78281F")

    ax_b.set_title("B   Criterion K1: Accelerated Divergence in Paratope vs Structural Core\n"
                   r"Primate Germline Median $\omega$ across BCR & TCR Lineages",
                   loc="left", fontsize=11, fontweight="bold", pad=8)
    ax_b.legend(loc="lower right", frameon=True, fontsize=9.5)

    # =========================================================================
    # Panel C: Criterion K2: Primate Selection vs 3D Crystal Contacts
    # Grouped Bar Plot: Raw vs Partial Spearman rho (controlling surface exposure)
    # =========================================================================
    ax_c = fig.add_subplot(gs[1, 0])

    target_k2 = [
        ("IGHV_MetaClans", "AbAg_Combined_5A", "IGHV (Ag Combined)"),
        ("IGKV_All", "AbAg_Combined_5A", "IGKV (Ag Combined)"),
        ("IGLV_All", "AbAg_Combined_5A", "IGLV (Ag Combined)"),
        ("TRAV_All", "TCR_pMHC_Total_5A", "TRAV (pMHC Total)"),
        ("TRBV_All", "TCR_pMHC_Total_5A", "TRBV (pMHC Total)"),
        ("TRBV_All", "TCR_MHC_Helix_5A", "TRBV (MHC-Helix)"),
        ("TRBV_All", "TCR_Peptide_5A", "TRBV (Peptide)"),
    ]

    c_rows = []
    for uname, metric, display_name in target_k2:
        m = df_k2[(df_k2["unit"] == uname) & (df_k2["contact_metric"] == metric)]
        if len(m) > 0:
            r = m.iloc[0]
            c_rows.append({
                "display": display_name,
                "raw_rho": r["spearman_rho_fubar_omega"],
                "raw_p": r["spearman_p_fubar_omega"],
                "part_rho": r["partial_rho_controlling_surface_exposure"],
                "part_p": r["partial_p_controlling_surface_exposure"],
            })

    df_c_plot = pd.DataFrame(c_rows)
    x_indices = np.arange(len(df_c_plot))
    bwidth = 0.35

    bars_raw = ax_c.bar(x_indices - bwidth/2, df_c_plot["raw_rho"], width=bwidth,
                        color="#884EA0", alpha=0.9, label=r"Raw Spearman $\rho$", zorder=3)
    bars_part = ax_c.bar(x_indices + bwidth/2, df_c_plot["part_rho"], width=bwidth,
                         color="#F39C12", alpha=0.9, label=r"Partial $\rho$ (Controlling 3D Surface Exposure)", zorder=3)

    ax_c.set_xticks(x_indices)
    ax_c.set_xticklabels(df_c_plot["display"], rotation=30, ha="right", fontsize=9.5, fontweight="bold")
    ax_c.set_ylabel(r"Spearman Rank Correlation ($\rho$)" + "\nwith 3D Contact Frequency (<5 Å)", fontsize=10, fontweight="bold")
    ax_c.axhline(0, color="#333333", lw=0.8, ls="-", zorder=1)
    ax_c.grid(axis="y", ls=":", alpha=0.5, zorder=0)
    ax_c.set_ylim(-0.05, max(0.7, df_c_plot["raw_rho"].max() * 1.25))

    def p_to_stars(p):
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return "ns"

    for idx, r in df_c_plot.iterrows():
        stars_r = p_to_stars(r["raw_p"])
        if stars_r != "ns":
            ax_c.text(idx - bwidth/2, r["raw_rho"] + 0.02, stars_r, ha="center", va="bottom",
                      fontsize=10, fontweight="bold", color="#512E5F")
        stars_p = p_to_stars(r["part_p"])
        if stars_p != "ns":
            ax_c.text(idx + bwidth/2, r["part_rho"] + 0.02, stars_p, ha="center", va="bottom",
                      fontsize=10, fontweight="bold", color="#7E5109")

    ax_c.set_title("C   Criterion K2: Structural Fidelity to Physical Binding Interfaces\n"
                   "Primate Site Selection Correlates with 3D Contacts Beyond Surface Accessibility",
                   loc="left", fontsize=11, fontweight="bold", pad=8)
    ax_c.legend(loc="upper right", frameon=True, fontsize=9.5)

    # =========================================================================
    # Panel D: Criterion K3: Human Public Repertoire Recruitment vs Primate Evolution
    # Scatter Plot of Public Expansion vs Primate omega with Archetype Callouts
    # =========================================================================
    ax_d = fig.add_subplot(gs[1, 1])

    sub_d = df_k3_atlas.dropna(subset=["log2_public_over_olga_pgen", "cdr12_vs_fr_omega_ratio"]).copy()

    locus_colors = {
        "IGHV": "#1F618D",
        "TRBV": "#27AE60",
        "TRAV": "#8E44AD",
        "IGKV": "#D35400",
        "IGLV": "#16A085"
    }

    for loc, group in sub_d.groupby("locus"):
        c = locus_colors.get(loc, "#7F8C8D")
        pos_grp = group[group["busted_mh_q_global"] < 0.05]
        neg_grp = group[group["busted_mh_q_global"] >= 0.05]

        ax_d.scatter(neg_grp["cdr12_vs_fr_omega_ratio"], neg_grp["log2_public_over_olga_pgen"],
                     color=c, alpha=0.45, s=40, edgecolors="none", label=f"{loc} (q≥0.05)")
        ax_d.scatter(pos_grp["cdr12_vs_fr_omega_ratio"], pos_grp["log2_public_over_olga_pgen"],
                     color=c, alpha=0.90, s=80, edgecolors="#1B2631", lw=1.2, label=f"{loc} (q<0.05)")

    x_vals = sub_d["cdr12_vs_fr_omega_ratio"].astype(float).values
    y_vals = sub_d["log2_public_over_olga_pgen"].astype(float).values
    slope, intercept, r_val, p_val, _ = sp_stats.linregress(x_vals, y_vals)
    rho, p_rho = sp_stats.spearmanr(x_vals, y_vals)

    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
    ax_d.plot(x_line, slope * x_line + intercept, color="#2C3E50", ls="--", lw=1.5, zorder=2)

    archetypes = [
        "IGHV1-69", "IGHV3-66", "IGHV3-30", "IGHV4-34",
        "TRBV19", "TRBV6-5", "TRAV12-2", "TRBV20-1"
    ]

    label_offsets = {
        "IGHV1-69": (35, -22),
        "IGHV3-66": (-50, 20),
        "IGHV3-30": (20, 15),
        "IGHV4-34": (-45, -20),
        "TRBV19": (-50, 22),
        "TRBV6-5": (-40, 18),
        "TRAV12-2": (20, -18),
        "TRBV20-1": (25, -10),
    }

    for arch in archetypes:
        row = sub_d[sub_d["human_gene"] == arch]
        if len(row) > 0:
            rx = float(row["cdr12_vs_fr_omega_ratio"].iloc[0])
            ry = float(row["log2_public_over_olga_pgen"].iloc[0])
            dx, dy = label_offsets.get(arch, (15, 15))
            ax_d.annotate(
                arch, xy=(rx, ry), xytext=(dx, dy), textcoords="offset points",
                fontsize=8.5, fontweight="bold", color="#17202A",
                bbox=dict(boxstyle="round,pad=0.22", fc="#FCF3CF", ec="#F39C12", lw=0.9, alpha=0.92),
                arrowprops=dict(arrowstyle="->", color="#34495E", lw=0.9, shrinkB=3)
            )

    ax_d.set_xlabel(r"Primate Germline CDR1+CDR2 vs Core FR $\omega$ Ratio", fontsize=10, fontweight="bold")
    ax_d.set_ylabel(r"Human Public Repertoire Recruitment" + "\n" + r"($\log_2$[Observed Public / Expected $P_{\mathrm{gen}}$])",
                    fontsize=10, fontweight="bold")
    ax_d.grid(True, ls=":", alpha=0.5, zorder=0)

    partial_pgen_str = f"{df_k3_tests['partial_rho_controlling_olga_pgen'].iloc[0]:.3f}" if len(df_k3_tests) > 0 else "N/A"
    stat_text = (
        f"Spearman $\\rho$ = {rho:.3f}\n"
        f"$p$ = {p_rho:.2e} (n={len(sub_d)})\n"
        f"Partial $\\rho$ (OLGA) = {partial_pgen_str}"
    )
    ax_d.text(0.04, 0.94, stat_text, transform=ax_d.transAxes, va="top", ha="left",
              fontsize=9.5, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.4", fc="#FFFFFF", ec="#BDC3C7", lw=1.0, alpha=0.95))

    ax_d.set_title("D   Criterion K3: Convergence of Primate Evolution & Human Public Immunity\n"
                   "Public Antiviral Archetypes Exhibit Heightened Germline CDR Divergence",
                   loc="left", fontsize=11, fontweight="bold", pad=8)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='BCR (IGHV)', markerfacecolor='#1F618D', markersize=8),
        Line2D([0], [0], marker='o', color='w', label=r'TCR$\beta$ (TRBV)', markerfacecolor='#27AE60', markersize=8),
        Line2D([0], [0], marker='o', color='w', label=r'TCR$\alpha$ (TRAV)', markerfacecolor='#8E44AD', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='q < 0.05 (BUSTED-MH)', markerfacecolor='#2C3E50',
               markeredgecolor='#1B2631', markeredgewidth=1.2, markersize=9),
    ]
    ax_d.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=9)

    # -------------------------------------------------------------------------
    # Save Outputs
    # -------------------------------------------------------------------------
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUT_PDF, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved Stage 0 figure to:\n  PNG: {OUT_PNG}\n  PDF: {OUT_PDF}", flush=True)


if __name__ == "__main__":
    plot_stage0_summary()
