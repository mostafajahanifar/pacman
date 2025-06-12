# Pan-Cancer Profiling of Mitotic Topology & Mitotic Errors 🟡●●●

PAn-Cancer Mitotic Activity Network (PACMAN 🟡) for profiling mitotic topology and mitotic arrors tumor microenvironment.

<img src="https://github.com/user-attachments/assets/7bbb2428-285e-4a15-ab3f-6962ab2c930f" alt="pacman" width="400"/>



---

## 🔬 Overview

This repository contains the complete codebase used in the associated publication:

> 📄 **"Pan-Cancer Profiling of Mitotic Topology & Mitotic Errors: Insights into Prognosis, Genomic Alterations, and Immune Landscape"**  
> _Mostafa Jahanifar, Muhammad Dawood, Neda Zamanitajeddin, Adam Shephard, Brinder Singh Chohan, Christof A Bertram, Noorul Wahab, Mark Eastwood, Marc Aubreville, Shan E Ahmed Raza, Fayyaz Minhas, Nasir Rajpoot_  
> doi: https://doi.org/10.1101/2025.06.07.25329181

### 🧰 Resources
- Our Demo website for visualizing mitotic activity networks: [PACMAN Viewer](https://tiademos.dcs.warwick.ac.uk/bokeh_app?demo=pacman)
- Detailed results on genomics, immune, and survival analyses using this repository: [Supplementary Tables](https://zenodo.org/records/14793678)
- Mitosis detection for TCGA slides: [Mitosis Detections and Mitotic Network in TCGA](https://zenodo.org/records/14548480)
- [Mitosis subtyping dataset](https://zenodo.org/records/15390543)

### 📊 Features
- 🧬 Genomic correlation, CNV, mutation, and methylation analysis

- 🏥 Mitotic morphology analyses such as survival and GSEA

- 📈 Exploring the mitotic landscape of cancer

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
To run the experiment, you will need external data tables on genomic, immune, or mitotic attributes. You can download selected or all of these dataset files from Zenodo by running:
```bash
python pacman/download_from_zenodo.py
```
You will be prompted to select which files to download. You can eather select the files by number or choose to download `all` files.


### 4. Configuration
All directory paths and global parameters are defined in:

```arduino
pacman/config.py
```
Especially, the path to the data (for reading or downloading) and the path for saaving the results can be changed in this module.

### 5. Run an analysis script
Run any analysis module using:
```bash
python -m pacman.genomic.cnv_anova
```

---

## 📚 Citation
If you use this repository in your work, please cite:

```bibtex
@article {Jahanifar2025pacman,
	author = {Jahanifar, Mostafa and Dawood, Muhammad and Zamanitajeddin, Neda and Shephard, Adam and Chohan, Brinder Singh and Bertram, Christof A and Wahab, Noorul and Eastwood, Mark and Aubreville, Marc and Raza, Shan E Ahmed and Minhas, Fayyaz and Rajpoot, Nasir},
	title = {Pan-Cancer Profiling of Mitotic Topology \& Mitotic Errors: Insights into Prognosis, Genomic Alterations, and Immune Landscape},
	elocation-id = {2025.06.07.25329181},
	year = {2025},
	doi = {10.1101/2025.06.07.25329181},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/early/2025/06/08/2025.06.07.25329181},
	eprint = {https://www.medrxiv.org/content/early/2025/06/08/2025.06.07.25329181.full.pdf},
	journal = {medRxiv}
}

```

## 📜 License
MIT License
