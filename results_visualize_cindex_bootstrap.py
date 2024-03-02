import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from pathlib import Path
import numpy as np
from utils import get_colors_dict

def plot_c_index(csv_files, sig_p_values):
    # Initialize an empty list to hold dataframes
    dataframes = []
    color_dict = get_colors_dict()
    fig = plt.figure(figsize=(15,3))  # Increase the figure size
    ax = fig.add_subplot(111)
    # Loop through the csv files
    for domain, csv_file in csv_files.items():
        # Check if the csv file exists and is not empty
        if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
            # Read the csv file into a dataframe
            df = pd.read_csv(csv_file)
            
            # Check if the dataframe has at least 500 rows
            if len(df) >= 500:
                # Add a column for the domain
                df['domain'] = domain
                df['significant'] = sig_p_values[domain]
                
                # Append the dataframe to the list
                dataframes.append(df)
            else:
                print(f"Skipping domain {domain} because it has less than 500 rows.")
        else:
            print(f"Skipping domain {domain} because the csv file does not exist or is empty.")

    # Concatenate all dataframes
    df_concat = pd.concat(dataframes)

    # Calculate the average c-index across all domains
    avg_c_index = df_concat['c_index'].mean()

    # Create a boxplot
    sns.boxplot(x='domain', y='c_index', data=df_concat, ax=ax, showfliers=False, palette=color_dict)

    # Draw a horizontal line at the average c-index
    ax.axhline(avg_c_index, color='r', linestyle='--')

    # Add "*" on top of the box for the domain where sig_p_values[domain] is True
    # Add significance asterisks
    for i, domain in enumerate(df_concat['domain'].unique()):
        sig_value = sig_p_values.get(domain, False)
        if sig_value is not None and sig_value < 0.05:
            max_value1 = df_concat[df_concat['domain'] == domain]['c_index'].max() * 0.98
            max_value2 = np.percentile(df_concat[df_concat['domain'] == domain]['c_index'].to_numpy(), 99)
            max_value = min(max_value1, max_value2) + 0.01
            astrisks = "*"
            if sig_value <= 0.01:
                astrisks = "**"
            if sig_value <= 0.0001:
                astrisks = "***"
            ax.text(i, max_value, astrisks, ha='center', va='bottom', color='black', fontsize=10)

    # Set the title and labels
    plt.title('Boxplot of c-index values for each domain', fontsize=12)
    plt.xlabel('Domain', fontsize=12)
    plt.ylabel('c-index values', fontsize=12)

    # Rotate x-axis labels if they overlap
    plt.xticks(rotation=45)

    return fig


event_types = ['DFI', 'PFI', 'OS', 'DSS']
for event_type in event_types:
    censor_at = -1
    cv_experiment = f'CV_results_NoCensor_median'
    cancer_types = [["BLCA"], ["BRCA"], ["CESC"], ["COAD", "READ"], ["ESCA"], ["GBM"], ["HNSC"], ["KICH"], ["KIRC"], ["KIRP"], ["LGG"], ["LIHC"], ["LUAD"], ["LUSC"], ["OV"], ["PAAD"], ["SKCM"], ["STAD"], ["UCEC"]]

    csv_files = {}
    sig_p_values = {}
    for cancer_type in cancer_types:
        csv_path = f"{cv_experiment}/CV_{cancer_type}/bootstrap_results_{cancer_type}_{event_type}_censor{censor_at}.csv"
        csv_files[''.join(cancer_type)] = csv_path

        # find the p-value related to this experiment
        directory = f"{cv_experiment.replace('CV_results', 'CV_corrected_results')}/CV_{cancer_type}_Corrected/" # cv_results_{cancer_type}_{event_type}_censor{censor_at}_*.csv"
        km_path_pattern = f"cv_results_{cancer_type}_{event_type}_censor{censor_at}"
        # Walk through the directory
        p_value = None
        for file in os.listdir(directory):
            # Check if the file name matches the pattern
            if file.startswith(km_path_pattern) and file.endswith('.csv'):
                p_value = float(file.strip('.csv').split('pvalue')[-1])
                break
        sig_p_values[''.join(cancer_type)] = p_value

    # Call the function
    fig = plot_c_index(csv_files, sig_p_values)
    plt.title(f'Boxplot of c-index values, Event: {event_type}', fontsize=12)
    plt.ylim([-0.1,1.1])
    save_path = cv_experiment + f'/bootstrap_cindex_{event_type}_censor{censor_at}'
    fig.savefig(save_path+".png", dpi=600, bbox_inches = 'tight', pad_inches = 0)
    fig.savefig(save_path+".pdf", dpi=600, bbox_inches = 'tight', pad_inches = 0)
