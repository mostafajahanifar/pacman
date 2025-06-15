#!/usr/bin/env python3

"""
Fallback script to run all pacman analyses sequentially
for users who don't have 'make' installed.
"""

import subprocess

commands = [
    "python -m pacman.landscape.features_box_plots",
    "python -m pacman.landscape.features_over_ethnicities",
    "python -m pacman.landscape.temprature_distribution",
    "python -m pacman.landscape.features_radar_plots",

    "python -m pacman.genomic.canonical_correlation_analysis",
    "python -m pacman.genomic.correlation_measuring --mode methylation --method spearman",
    "python -m pacman.genomic.correlation_measuring --mode expression --method pearson",
    "python -m pacman.genomic.methylation_results_aggregation",
    "python -m pacman.genomic.methylation_visualization",
    "python -m pacman.genomic.correlation_pacman_set_extraction",
    "python -m pacman.genomic.mutation_auc_pval",
    "python -m pacman.genomic.mutation_results_aggregation",
    "python -m pacman.genomic.cnv_anova",
    "python -m pacman.genomic.cnv_results_aggregation",
    "python -m pacman.genomic.gsea_analysis",
    "python -m pacman.genomic.gsea_results_aggregation",

    "python -m pacman.survival.cindex_analysis",
    "python -m pacman.survival.kmplots_cv",
    "python -m pacman.survival.results_cv_collect",

    "python -m pacman.immune.correlation",
    "python -m pacman.immune.distributions",
    "python -m pacman.immune.survival",

    "python -m pacman.morphology.cin_correlation",
    "python -m pacman.morphology.survival_multivariate",
    "python -m pacman.morphology.survival_univariate_kmplot",
    "python -m pacman.morphology.survival_univariate_hr",
    "python -m pacman.morphology.gsea_analysis",
    "python -m pacman.morphology.gsea_results_aggregation",
]

if __name__ == "__main__":
    for cmd in commands:
        print(f"\n🚀 Running: {cmd}")
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed: {cmd}")
            print(f"   Error: {e}")
            break
