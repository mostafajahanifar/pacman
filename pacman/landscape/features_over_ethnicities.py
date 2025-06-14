import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statannotations.Annotator import Annotator

from pacman.config import ALL_CANCERS, DATA_DIR, ETHNICITIES_DICT, RESULTS_DIR
from pacman.utils import get_colors_dict

# Load the mitosis_feats
mitosis_feats = pd.read_excel(os.path.join(DATA_DIR, "ST1-tcga_mtfs.xlsx"))


# Define feature(s) of interest
features_list = ["HSC"]

# Filter the mitosis_featsframe to include relevant columns and remove invalid cancer types
mitosis_feats = mitosis_feats[mitosis_feats['type'].isin(ALL_CANCERS)]


# Map and filter race mitosis_feats
ethnicities = list(ETHNICITIES_DICT.keys())

mitosis_feats = mitosis_feats[mitosis_feats['race'].isin(ethnicities)]
mitosis_feats['race'] = mitosis_feats['race'].map(ETHNICITIES_DICT)

# Define color palette for races
group_color_dict = {
    'White': np.array((255, 244, 224)) / 255,
    'Black': np.array((103, 70, 52)) / 255,
    'Asian': np.array((244, 224, 175)) / 255,
    'Pacific Islander': np.array((198, 155, 123)) / 255,
    'Native American': np.array((127, 64, 52)) / 255,
}

# Filter races with less than 5 rows for each cancer type
counts = mitosis_feats.groupby(['type', 'race']).size().reset_index(name='counts')
mitosis_feats = mitosis_feats.merge(counts, on=['type', 'race'])
mitosis_feats = mitosis_feats[mitosis_feats['counts'] >= 5]

# Plot the mitosis_feats for each feature
for feature in features_list:
    # Compute mean feature values for each cancer type and sort them
    mean_feature_by_type = mitosis_feats.groupby('type')[feature].mean()
    sorted_cancer_types = mean_feature_by_type.sort_values().index.tolist()
    
    plt.figure(figsize=(14, 3))
    
    # Create a bar plot in the background
    ax = plt.gca()
    x_positions = range(len(sorted_cancer_types))
    bar_heights = mean_feature_by_type[sorted_cancer_types]
    color_pallete = get_colors_dict()
    cancer_colors = [color_pallete[cancer] for cancer in sorted_cancer_types]
    ax.bar(
        x_positions,
        bar_heights,
        color='lightgreen',
        alpha=1.0,
        width=0.8,
        zorder=0,
        align='center'
    )
    
    # Adjust the x-axis to match cancer type labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(sorted_cancer_types, rotation=90)

    # Add the boxplot
    hue_plot_params = {
        'data':      mitosis_feats,
        'x':         'type',
        'y':         feature,
        "order":     sorted_cancer_types,
        "hue":       "race",
        "hue_order": ["White", "Asian", "Black"],
        "palette":   group_color_dict,
    }
    
    sns.boxplot(
        **hue_plot_params,
                showfliers=False,
                linewidth=0.5,
                width=0.6,
                zorder=1,  # Place boxplots on top of the bars
                ax=ax)

    # Customize the plot
    plt.xlabel('', fontsize=12)
    plt.legend([], [], frameon=False)

    # Annotating with p-values
    pairs = []
    for c_type in mitosis_feats["type"].unique():
        if len(mitosis_feats.loc[(mitosis_feats["type"] == c_type) & (mitosis_feats["race"] == "White")]) > 0 and len(mitosis_feats.loc[(mitosis_feats["type"] == c_type) & (mitosis_feats["race"] == "Asian")]) > 0:
            pairs.append([(c_type, "White"), (c_type, "Asian")])
        if len(mitosis_feats.loc[(mitosis_feats["type"] == c_type) & (mitosis_feats["race"] == "Black")]) > 0 and len(mitosis_feats.loc[(mitosis_feats["type"] == c_type) & (mitosis_feats["race"] == "Asian")]) > 0:
            pairs.append([(c_type, "Asian"), (c_type, "Black")])
        if len(mitosis_feats.loc[(mitosis_feats["type"] == c_type) & (mitosis_feats["race"] == "White")]) > 0 and len(mitosis_feats.loc[(mitosis_feats["type"] == c_type) & (mitosis_feats["race"] == "Black")]) > 0:
            pairs.append([(c_type, "White"), (c_type, "Black")])

    annotator = Annotator(ax, pairs, **hue_plot_params)
    annotator.configure(test="Mann-Whitney", verbose=False, line_width=1, hide_non_significant=True, comparisons_correction="fdr_bh")
    annotator.apply_and_annotate()
    
    # Save and display the plot
    plt.tight_layout()
    save_dir = os.path.join(RESULTS_DIR, "landscape")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/ethnicities_dist_{feature}.png", dpi=600, bbox_inches='tight', pad_inches=0.01)