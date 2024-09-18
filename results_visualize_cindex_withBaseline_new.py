import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu
from utils import get_colors_dict
import textwrap

def wrap_domain_names(unique_domains):
    wrapped_domains = []
    for domain in unique_domains:
        if domain=="COADREAD":
            wrapped_domains.append(textwrap.fill(domain, 4))
        elif domain=="GBMLGG":
            wrapped_domains.append(textwrap.fill(domain, 3))
        else:
            wrapped_domains.append(domain)
    return wrapped_domains
    

def plot_c_index(csv_files, baseline_csv_files):
    # Initialize an empty list to hold dataframes
    dataframes = []
    baseline_dataframes = []
    color_dict = get_colors_dict()

    # Loop through the csv files
    for domain, csv_file in csv_files.items():
        # Check if the csv file exists and is not empty
        if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
            # Read the csv file into a dataframe
            df = pd.read_csv(csv_file)
            
            # Check if the dataframe has at least 500 rows
            if len(df) >= 800:
                # Add a column for the domain
                df['domain'] = domain
                
                # Append the dataframe to the list
                dataframes.append(df)

                # Check if the csv file exists and is not empty
                if os.path.exists(baseline_csv_files[domain]) and os.path.getsize(baseline_csv_files[domain]) > 0:
                    # Read the csv file into a dataframe
                    df_baseline = pd.read_csv(baseline_csv_files[domain])
                    
                    # Check if the dataframe has at least 500 rows
                    if len(df_baseline) >= 800:
                        # Add a column for the domain
                        df_baseline['domain'] = domain
                        
                        # Append the dataframe to the list
                        baseline_dataframes.append(df_baseline)
            else:
                print(f"Skipping domain {domain} because it has less than 800 rows.")
        else:
            print(f"Skipping domain {domain} because the csv file does not exist or is empty.")

    # Concatenate all dataframes
    df_original = pd.concat(dataframes)
    df_baseline = pd.concat(baseline_dataframes)

    
    fig = plt.figure(figsize=(40,6))  # Increase the figure size
    ax = fig.add_subplot(111)

    # Get the unique domains
    unique_domains = sorted(df_original['domain'].unique())

    gap = 12
    width = 5 # 6
    for i, domain in enumerate(unique_domains):
        print(domain)
        # Filter data for the current domain
        data1 = df_baseline[df_baseline['domain'] == domain]['c_index']
        data2 = df_original[df_original['domain'] == domain]['c_index']
        print(len(data1), len(data1))

        # Plot boxplots at specified positions
        box2 = ax.boxplot(data2, positions=[i*gap + .53*width], widths=width, patch_artist=True, showfliers=False, boxprops=dict(facecolor=color_dict[domain]), medianprops=dict(color='black'))
        box1 = ax.boxplot(data1, positions=[i*gap - .53*width], widths=width, patch_artist=True, showfliers=False, boxprops=dict(facecolor=color_dict[domain], hatch='//'), medianprops=dict(color='black'))

        # Get upper whisker position for both boxplots
        whisker1_top = box1['whiskers'][1].get_ydata()[1]
        whisker2_top = box2['whiskers'][1].get_ydata()[1]

        y_max = max(whisker1_top, whisker2_top)

        # Perform Mann-Whitney U Test
        u_statistic, p_value = mannwhitneyu(data1, data2, alternative='less')

        # Determine the significance level based on p-value
        if p_value < 0.0001:
            asterisks = "***"
        elif p_value < 0.01:
            asterisks = "**"
        elif p_value < 0.05:
            asterisks = "*"
        else:
            asterisks = False

        # Draw a horizontal line between the two boxplots
        if asterisks:
            line_x_start = i * gap - 0.53 * width
            line_x_end = i * gap + 0.53 * width
            line_y = y_max + 0.03  # Small offset above the higher whisker

            ax.plot([line_x_start, line_x_end], [line_y, line_y], color='black', lw=1.5)
            # Add small vertical lines (caps) at the start and end of the horizontal line
            cap_length = 0.01  # Length of the caps
            ax.plot([line_x_start, line_x_start], [line_y - cap_length, line_y], color='black', lw=1.5)
            ax.plot([line_x_end, line_x_end], [line_y - cap_length, line_y], color='black', lw=1.5)

            # Add asterisks at the midpoint of the line with increased font size
            ax.text((line_x_start + line_x_end) / 2, line_y, asterisks, ha='center', va='bottom', fontsize=15)

    # Set xticks and labels
    ax.set_xticks(range(0, len(unique_domains) * gap, gap))
    ax.set_xticklabels(wrap_domain_names(unique_domains))

    plt.xlim([-1.1 * width, i * gap + 1.1 * width])

    # Draw a horizontal line at the average c-index
    ax.axhline(0.5, color='r', linestyle='--')

    # Set the title and labels
    # plt.title('Boxplot of c-index values for each domain', fontsize=24)
    # plt.xlabel('Domain', fontsize=24)
    # plt.ylabel('c-index values', fontsize=24)

    # Rotate x-axis labels if they overlap
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    return fig


event_types = ['DFI', 'PFI', 'OS', 'DSS']
for event_type in event_types:
    print("**************",event_type)
    censor_at = 120
    cv_experiment = f'results_final/survival/bootstrap_10years_feat10sel'
    cancer_types = [["ACC"], ["BLCA"], ["BRCA"], ["CESC"], ["COAD", "READ"], ["ESCA"], ["GBM", "LGG"], ["HNSC"], ["KIRC"], ["KIRP"], ["LIHC"], ["LUAD"], ["LUSC"], ["OV"], ["PAAD"], ["SKCM"], ["STAD"], ["UCEC"], ["MESO"], ["PRAD"], ["SARC"], ["TGCT"], ["THCA"], ["KICH"]]

    csv_files = {}
    baseline_csv_files = {}
    for cancer_type in cancer_types:
        csv_path = f"{cv_experiment}/CV_{cancer_type}/bootstrap_results_{cancer_type}_{event_type}_censor{censor_at}.csv"
        csv_files[''.join(cancer_type)] = csv_path
        baseline_csv_files[''.join(cancer_type)] = csv_path.replace('bootstrap_10years_feat10sel/', 'baseline_bootstrap_10years/')

    # Call the function
    fig = plot_c_index(csv_files, baseline_csv_files)
    # plt.title(f'Boxplot of c-index values, Event: {event_type}', fontsize=24)
    plt.ylabel(f'C-index for {event_type}', fontsize=24)
    plt.ylim([-0.1, 1.1])

    save_path = cv_experiment + f'/bootstrap_cindex_{event_type}_censor{censor_at}_withBaseline'
    fig.savefig(save_path + ".png", dpi=600, bbox_inches='tight', pad_inches=0.01)
    fig.savefig(save_path + ".pdf", dpi=600, bbox_inches='tight', pad_inches=0.01)

