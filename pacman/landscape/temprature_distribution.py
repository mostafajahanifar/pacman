import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pacman.config import ALL_CANCERS, DATA_DIR, ETHNICITIES_DICT, RESULTS_DIR

print(7 * "=" * 7)
print("Ploting Mitotic-Hot and Cold distributions")
print(7 * "=" * 7)

# Load the data
mitosis_feats = pd.read_excel(os.path.join(DATA_DIR, "ST1-tcga_mtfs.xlsx"))

mitosis_feats = mitosis_feats[mitosis_feats["type"].isin(ALL_CANCERS)]

# Create a figure with subplots (3x1) for type, gender, and race
fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3, figsize=(15, 3), gridspec_kw={"width_ratios": [30, 2, 5]}
)

# Set color map for "temperature"
colors = plt.get_cmap("coolwarm")

# ---------- Cancer Type Distribution ----------
# Group by type and temperature to get the count
type_temp_counts = (
    mitosis_feats.groupby(["type", "temperature"]).size().unstack(fill_value=0)
)
type_temp_distribution = type_temp_counts.div(type_temp_counts.sum(axis=1), axis=0)
type_counts = mitosis_feats["type"].value_counts()

# Normalize counts to the desired range
min_width = 0.1
max_width = 1
normalized_widths = (type_counts - type_counts.min()) / (
    type_counts.max() - type_counts.min()
)
bar_widths = normalized_widths * (max_width - min_width) + min_width
bar_widths = bar_widths.sort_index()

# Dynamic x positions for type plot
x_positions = []
next_pos = 0
for iw, width in enumerate(bar_widths):
    if iw == 0:
        next_pos = 0
    else:
        next_pos += bar_widths[iw - 1] / 2 + 0.1 + bar_widths[iw] / 2
    x_positions.append(next_pos)

# Plot cancer type distribution (stacked bar)
for i, (type_name, row) in enumerate(type_temp_distribution.iterrows()):
    bottom = 0
    for j, temp in enumerate(row.index):
        height = row[temp]
        ax1.bar(
            x_positions[i],
            height,
            bottom=bottom,
            color=colors(j / (len(row.index) - 1)),
            width=bar_widths[type_name],
            edgecolor="gray",
            linewidth=0.5,
        )
        bottom += height


# Wrap and add type labels below bars
def wrap_label(label, width=4):
    if label == "GBMLGG":
        return "GBM\nLGG"
    return "\n".join([label[i : i + width] for i in range(0, len(label), width)])


wrapped_labels = [wrap_label(label) for label in type_temp_distribution.index]
# for i, type_name in enumerate(type_temp_distribution.index):
#     ax1.text(x_positions[i], -0.02, wrapped_labels[i], ha='center', va='top', fontsize=12, rotation=90)

ax1.set_title("Cancer Type")
ax1.set_yticks([])
ax1.set_yticklabels("")
ax1.spines[["right", "left", "bottom"]].set_visible(False)
ax1.set_xticks(x_positions)
ax1.set_xticklabels(wrapped_labels, rotation=90)
ax1.set_xlim([min(x_positions), max(x_positions)])
ax1.tick_params(axis="both", which="both", length=0)

# ---------- Gender Distribution ----------
gender_temp_counts = (
    mitosis_feats.groupby(["gender", "temperature"]).size().unstack(fill_value=0)
)
gender_temp_distribution = gender_temp_counts.div(
    gender_temp_counts.sum(axis=1), axis=0
)
gender_counts = mitosis_feats["gender"].value_counts()
gender_bar_widths = [0.6, 0.6]  # Fixed bar widths for gender

for i, (gender, row) in enumerate(gender_temp_distribution.iterrows()):
    bottom = 0
    for j, temp in enumerate(row.index):
        height = row[temp]
        ax2.bar(
            i,
            height,
            bottom=bottom,
            color=colors(j / (len(row.index) - 1)),
            width=gender_bar_widths[i],
            edgecolor="gray",
            linewidth=0.5,
        )
        bottom += height

ax2.set_xticks([0, 1])
ax2.set_xticklabels(["Male", "Female"], rotation=90)
ax2.set_title("Gender")
ax2.set_yticks([])
ax2.set_yticklabels("")
ax2.spines[["right", "left", "bottom"]].set_visible(False)
ax2.tick_params(axis="both", which="both", length=0)

# ---------- Race Distribution ----------
# Preprocess race data
ethnicities = list(ETHNICITIES_DICT.keys())

mitosis_feats = mitosis_feats[mitosis_feats["race"].isin(ethnicities)]
mitosis_feats["race"] = mitosis_feats["race"].map(ETHNICITIES_DICT)

# Group by race and temperature
race_temp_counts = (
    mitosis_feats.groupby(["race", "temperature"]).size().unstack(fill_value=0)
)
race_temp_distribution = race_temp_counts.div(race_temp_counts.sum(axis=1), axis=0)
race_bar_widths = [0.6] * len(race_temp_distribution)

for i, (race, row) in enumerate(race_temp_distribution.iterrows()):
    bottom = 0
    for j, temp in enumerate(row.index):
        height = row[temp]
        ax3.bar(
            i,
            height,
            bottom=bottom,
            color=colors(j / (len(row.index) - 1)),
            width=race_bar_widths[i],
            edgecolor="gray",
            linewidth=0.5,
        )
        bottom += height

ax3.set_xticks(np.arange(len(race_temp_distribution)))
ax3.set_xticklabels(race_temp_distribution.index, rotation=90)
ax3.set_title("Race")
ax3.set_yticks([])
ax3.set_yticklabels("")
ax3.spines[["right", "left", "bottom"]].set_visible(False)
ax3.tick_params(axis="both", which="both", length=0)

plt.subplots_adjust(wspace=0.05)
# Adjust layout and save figure
plt.tight_layout()
save_dir = os.path.join(RESULTS_DIR, "landscape")
os.makedirs(save_dir, exist_ok=True)
plt.savefig(
    f"{save_dir}/hot-cold-distribution.png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.01,
)
