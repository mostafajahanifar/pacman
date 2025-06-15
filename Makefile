.PHONY: all \
	landscape \
	genomic \
	survival \
	immune \
	morphology

PYTHON := python

all: landscape genomic survival immune morphology

landscape:
	@echo "Running Landscape Analysis..."
	$(PYTHON) -m pacman.landscape.features_box_plots
	$(PYTHON) -m pacman.landscape.features_over_ethnicities
	$(PYTHON) -m pacman.landscape.temeprature_distribution
	$(PYTHON) -m pacman.landscape.features_radar_plots

genomic:
	@echo "Running Genomic Analysis..."
	$(PYTHON) -m pacman.genomic.canonical_correlation_analysis
	$(PYTHON) -m pacman.genomic.correlation_measuring --mode methylation --method spearman
	$(PYTHON) -m pacman.genomic.correlation_measuring --mode expression --method pearson
	$(PYTHON) -m pacman.genomic.methylation_results_aggregation
	$(PYTHON) -m pacman.genomic.methylation_visualization
	$(PYTHON) -m pacman.genomic.correlation_pacman_set_extraction
	$(PYTHON) -m pacman.genomic.mutation_auc_pval
	$(PYTHON) -m pacman.genomic.mutation_results_aggregation
	$(PYTHON) -m pacman.genomic.cnv_anova
	$(PYTHON) -m pacman.genomic.cnv_results_aggregation
	$(PYTHON) -m pacman.genomic.gsea_analysis
	$(PYTHON) -m pacman.genomic.gsea_results_aggregation

survival:
	@echo "Running Survival Analysis..."
	$(PYTHON) -m pacman.survival.cindex_analysis
	$(PYTHON) -m pacman.survival.kmplots_cv
	$(PYTHON) -m pacman.survival.results_cv_collect

immune:
	@echo "Running Immune Analysis..."
	$(PYTHON) -m pacman.immune.correlation
	$(PYTHON) -m pacman.immune.distributions
	$(PYTHON) -m pacman.immune.survival

morphology:
	@echo "Running Morphology Analysis..."
	$(PYTHON) -m pacman.morphology.cin_correlation
	$(PYTHON) -m pacman.morphology.survival_multivariate
	$(PYTHON) -m pacman.morphology.survival_univariate_kmplot
	$(PYTHON) -m pacman.morphology.survival_univariate_hr
	$(PYTHON) -m pacman.morphology.gsea_analysis
	$(PYTHON) -m pacman.morphology.gsea_results_aggregation
