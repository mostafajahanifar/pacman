# 🟡●●● Pan-Cancer Profiling of Mitotic Topology & Mitotic Errors

PAn-Cancer Mitotic Activity Network (PACMAN 🟡) for profiling mitotic topology and mitotic arrors tumor microenvrionment. 

---

## 🔬 Overview

This repository contains the complete codebase used in the associated publication:

> 📄 _"Pan-Cancer Profiling of Mitotic Topology & Mitotic Errors: Insights into Prognosis, Genomic Alterations, and Immune Landscape"_  
> **[Mostafa Jahanifar et al.]**  
> [Add preprint or DOI link]


📊 Features
- 🧬 Genomic correlation, CNV, mutation, and methylation analysis

- 🏥 Mitotic morphology analyses such as survival and GSEA

- 📈 Exploring mitotic landscape of cancer

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/pacman.git
cd pacman
```

### 2. Set up the environment
```bash
conda env create -f environment.yml
conda activate pacman
```

### 3. Download the data


### 4. Configuration
All directory paths and global parameters are defined in:

```arduino
pacman/config.py
```

### 5. Run an analysis script
Run any analysis module using:
```bash
python -m pacman.genomic.cnv_anova
```

---

## 📚 Citation
If you use this repository in your work, please cite:

```bibtex
@article{your2025pacman,
  title={Integrative Analysis of Genomic and Morphological Features for Cancer Prognosis},
  author={Your Name and Others},
  journal={Bioinformatics / Nature Methods / etc.},
  year={2025},
  doi={your-doi-here}
}
```

## 📜 License
MIT License