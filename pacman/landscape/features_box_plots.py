import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pacman.config import DATA_DIR, ETHNICITIES_DICT, RESULTS_DIR
from pacman.utils import get_colors_dict

# Load the data
mitosis_feats = pd.read_excel(os.path.join(DATA_DIR, "ST1-tcga_mtfs.xlsx"))

# Load feature list
feats_list = ["HSC", "AFW", "AMAH"] # aty_wsi_ratio, aty_ahotspot_count

# Filter the dataframe to include relevant columns and remove invalid cancers
df = mitosis_feats[['type', 'temperature', 'gender', 'race'] + feats_list]
invalid_cancers = ['LAML', 'UVM']
df = df[~df['type'].isin(invalid_cancers)]

# Filter the race column to keep only certain values and shorten the names
ethnicities = list(ETHNICITIES_DICT.keys())
# Keep only the valid races and map to shorter names
df = df[df['race'].isin(ethnicities)]
df['race'] = df['race'].map(ETHNICITIES_DICT)

# Sort dataframe by 'type'
df = df.sort_values(by="type")

# Get color palette
color_pallete = get_colors_dict()

# Plot the data for each feature
for feat in feats_list:
    # Create a figure with four subplots (type, race, gender, temperature - swapped)
    fig, (ax1, ax3, ax4, ax2) = plt.subplots(1, 4, figsize=(14, 3), gridspec_kw={'width_ratios': [30, 2, 5, 2]})

    # Boxplot by type (left plot)
    # **Compute the mean of 'feat' for each 'type' and sort them**
    mean_feat_by_type = df.groupby('type')[feat].mean()
    sorted_types = mean_feat_by_type.sort_values().index.tolist()

    sns.boxplot(ax=ax1, y=feat, x='type', hue='type', data=df, showfliers=False, palette=color_pallete,
                order=sorted_types, hue_order=sorted_types)
    ax1.set_yticklabels(ax1.get_yticklabels(), horizontalalignment='right')
    ax1.set_ylabel(feat)
    ax1.set_xlabel('')
    ax1.set_title('Cancer type')
    ax1.legend([], [], frameon=False)  # Remove the legend for the type boxplot

    # Rotate x-axis ticks on the first plot
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90, ha='center')

    # Boxplot by race (now second plot)
    sns.boxplot(ax=ax4, y=feat, x='race', data=df, showfliers=False, color="gray")
    ax4.set_ylabel('')
    ax4.set_xlabel('')
    ax4.set_title('Race')

    # Remove y-axis ticks and labels for the second plot (race)
    ax4.set_yticklabels([])  # Remove y-axis labels
    ax4.tick_params(left=False, right=False)  # Remove y-axis ticks
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=90, ha='center')

    # Boxplot by gender (third plot)
    gender_palette = {'MALE': 'skyblue', 'FEMALE': 'pink'}
    # Apply the custom color palette to the boxplot
    sns.boxplot(ax=ax3, y=feat, x='gender', data=df, showfliers=False, palette=gender_palette)
    ax3.set_ylabel('')
    ax3.set_xlabel('')
    ax3.set_xticklabels(['Male', 'Female'], rotation=90, ha='center')
    ax3.set_title('Gender')

    # Remove y-axis ticks and labels for the third plot (gender)
    ax3.set_yticklabels([])  # Remove y-axis labels
    ax3.tick_params(left=False, right=False)  # Remove y-axis ticks
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90, ha='center')

    # Boxplot by temperature (now fourth plot)
    cmap = plt.get_cmap('coolwarm')
    hot_cold_palette = {'Hot': cmap(.999), 'Cold': cmap(0)}
    sns.boxplot(ax=ax2, y=feat, x='temperature', data=df, showfliers=False, palette=hot_cold_palette)
    ax2.set_ylabel('')

    ax2.set_xlabel('')
    ax2.set_title('Mitosis')

    # Remove y-axis ticks and labels for the fourth plot (temperature)
    ax2.set_yticklabels([])  # Remove y-axis labels
    ax2.tick_params(left=False, right=False)  # Remove y-axis ticks
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90, ha='center') # , rotation=45, ha='right'

    # Adjust layout to bring subplots closer
    plt.subplots_adjust(wspace=0.02)

    save_dir = os.path.join(RESULTS_DIR, "landscape")
    plt.savefig(f"{save_dir}/boxplot_{feat}.png", dpi=600, bbox_inches = 'tight', pad_inches = 0.01)
