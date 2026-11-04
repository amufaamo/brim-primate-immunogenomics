# Natural selection sculpts primate germline antibody and T cell repertoires

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

This repository contains the complete reproducible codebase, structural paratope mapping workflows, and evolutionary data matrices for the study:

> **Natural selection sculpts primate germline antibody and T cell repertoires**  
> Masakazu Ishikawa$^{1,2,*}$  
> $^{1}$Bioinformatics Analysis Center, Kagawa University, Kagawa 761-0793, Japan.  
> $^{2}$Department of Pharmacology, Faculty of Medicine, Kagawa University, Kagawa 761-0793, Japan.  
> $^{*}$Correspondence: `ishikawa.masakazu@kagawa-u.ac.jp`

---

## Overview

Adaptive immunity is traditionally viewed as an open-ended somatic generator whose specificity is discovered almost entirely de novo through combinatorial recombination and somatic hypermutation. In this study, we capitalize on gapless telomere-to-telomere (T2T) chromosome assemblies across eight catarrhine primates spanning 30 million years of divergence, integrated with 1,235 high-resolution crystallographic complexes and human clinical repertoire deep sequencing, to demonstrate that primate germline variable gene repertoires are intensely sculpted by positive Darwinian selection.

Key findings implemented and verified in this codebase:
1. **Gapless T2T Primate Immunogenomics**: Unbiased census of 1,588 functional V genes (3,671 total V segments) across eight catarrhine primates (*Homo sapiens*, *Pan troglodytes*, *Pan paniscus*, *Gorilla gorilla*, *Pongo abelii*, *Pongo pygmaeus*, *Symphalangus syndactylus*, *Macaca mulatta*).
2. **Exposed Non-Contact Surface Controls**: Refutation of structural relaxation models by demonstrating 2.0- to 3.0-fold accelerated diversification (*P* < 0.003) of CDR paratopes compared to solvent-exposed, non-contacting framework residues.
3. **3D Crystallographic Interface Mapping**: Residue-level positive diversifying selection quantitatively tracks empirical heavy-atom antigen contact frequencies (*P* = 2.43 × 10⁻⁴).
4. **Public Antiviral Archetypes & Clinical Trade-offs**: Macroevolutionary basis of human *IGHV1-69* broadly neutralizing antiviral efficacy (Phe62 anchor) and evolutionary retention of self-reactive *IGHV4-34* regulated by peripheral checkpoints.

---

## Repository Structure

```
├── README.md               # Repository documentation and guide
├── LICENSE                 # MIT License (2026 Masakazu Ishikawa)
├── environment.yml         # Conda environment specification
├── .gitignore              # Standard git exclusion rules
├── data/                   # Processed datasets and evolutionary test tables
│   ├── brim_t2t_01_manifest.tsv
│   ├── brim_t2t_02_germline_counts.tsv
│   ├── brim_t2t_03_tier1_site_selection.tsv
│   ├── brim_t2t_03_tier2_orthogroup_selection.tsv
│   ├── brim_t2t_03b_partition_sensitivity.tsv
│   ├── brim_t2t_04_k1_summary.tsv
│   ├── brim_t2t_04_k2_3d_correlations.tsv
│   ├── brim_t2t_04_k3_repertoire_tests.tsv
│   ├── brim_t2t_04_gene_integrated_atlas.tsv
│   ├── brim_t2t_04_master_imgt104_3d_selection.tsv
│   └── vane_bcr_v_usage_summary.tsv
└── scripts/                # Analysis, statistics, and figure generation scripts
    ├── 01_download_t2t_genomes.sh
    ├── 01b_download_structure_repertoire_dbs.py
    ├── 01c_fix_olga_pgen.py
    ├── 02_extract_t2t_germline_v.py
    ├── 03_run_codon_selection.py
    ├── 03b_recombination_gard_sensitivity.py
    ├── 04a_compute_3d_contacts.py
    ├── 04b_stage0_kill_integration.py
    ├── 05_master_driver.sh
    ├── 06_plot_stage0_figures.py
    ├── 07_plot_supplementary_figures.py
    ├── 08_reconcile_and_compute_controls.py
    ├── 09_verify_vane_and_stats.py
    ├── 10_plot_nature_main_figures.py
    ├── 11_plot_extended_data_figures.py
    └── 12_audit_final_package.py
```

---

## Installation & Environment Setup

Clone the repository and install the required dependencies using Conda:

```bash
git clone https://github.com/<username>/brim-primate-immunogenomics.git
cd brim-primate-immunogenomics

conda env create -f environment.yml
conda activate brim
```

### Core External Tools
- **IgBLAST** (v1.22.0)
- **IQ-TREE 2** (v2.2.2.7)
- **HyPhy** (v2.5.101; BUSTED[S]+MH, FEL, MEME, FUBAR)
- **OLGA** (v1.2.4)

---

## Reproducing the Analysis

All analytical steps can be executed sequentially or driven via the master script:

```bash
# Set environment variable pointing to data directory (defaults to ./data)
export BRIM_DATA_DIR="./data"

# Execute table reconciliation and statistical verification
python3 scripts/08_reconcile_and_compute_controls.py

# Verify repertoire frequencies and donor statistics
python3 scripts/09_verify_vane_and_stats.py

# Re-generate Main Figures 1-4
python3 scripts/10_plot_nature_main_figures.py

# Re-generate Extended Data Figures 1-4
python3 scripts/11_plot_extended_data_figures.py
```

---

## Data Availability

All primary genome assemblies analyzed are available through NCBI RefSeq:
- *Homo sapiens* T2T-CHM13v2.0 (`GCF_009914755.1`), GRCh38.p14 (`GCF_000001405.40`)
- *Pan troglodytes* (`GCF_028858775.2`), *Pan paniscus* (`GCF_029289425.2`)
- *Gorilla gorilla* (`GCF_029281585.2`)
- *Pongo abelii* (`GCF_028885655.2`), *Pongo pygmaeus* (`GCF_028885625.2`)
- *Symphalangus syndactylus* (`GCF_028878055.3`)
- *Macaca mulatta* (`GCF_049350105.2`)

Structural coordinates were derived from SAbDab, CoV-AbDab, STCRDab, and TCR3d. Deep-sequencing repertoire datasets were accessed from NCBI BioProject `PRJNA338795` (SRA Study `SRP081539`).

---

## Citation & License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

If you use this code or data in your research, please cite:
Ishikawa, M. Natural selection sculpts primate germline antibody and T cell repertoires. *Nature* (2026).
