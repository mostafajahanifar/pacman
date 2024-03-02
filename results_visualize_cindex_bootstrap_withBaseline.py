import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from pathlib import Path
import numpy as np
from utils import get_colors_dict

def plot_c_index(csv_files, sig_p_values, baseline_csv_files, baseline_sig_p_values):
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
            if len(df) >= 500:
                # Add a column for the domain
                df['domain'] = domain
                df['significant'] = sig_p_values[domain]
                
                # Append the dataframe to the list
                dataframes.append(df)

                # Check if the csv file exists and is not empty
                if os.path.exists(baseline_csv_files[domain]) and os.path.getsize(baseline_csv_files[domain]) > 0:
                    # Read the csv file into a dataframe
                    df = pd.read_csv(baseline_csv_files[domain])
                    
                    # Check if the dataframe has at least 500 rows
                    if len(df) >= 500:
                        # Add a column for the domain
                        df['domain'] = domain
                        df['significant'] = baseline_sig_p_values[domain]
                        
                        # Append the dataframe to the list
                        baseline_dataframes.append(df)
            else:
                print(f"Skipping domain {domain} because it has less than 500 rows.")
        else:
            print(f"Skipping domain {domain} because the csv file does not exist or is empty.")

    # Concatenate all dataframes
    df_original = pd.concat(dataframes)
    df_basesline = pd.concat(baseline_dataframes)

    
    fig = plt.figure(figsize=(40,6))  # Increase the figure size
    ax = fig.add_subplot(111)

    # Get the unique domains
    # unique_domains = pd.concat([df1['domain'], df2['domain']]).unique()
    unique_domains = df_original['domain'].unique()

    gap = 14
    width = 6
    for i, domain in enumerate(unique_domains):
        # Filter data for the current domain
        data1 = df_basesline[df_basesline['domain'] == domain]['c_index']
        data2 = df_original[df_original['domain'] == domain]['c_index']

        # Plot boxplots at specified positions
        box2 = ax.boxplot(data2, positions=[i*gap + .53*width], widths=width, patch_artist=True, showfliers=False, boxprops=dict(facecolor=color_dict[domain]), medianprops=dict(color='black'))
        box1 = ax.boxplot(data1, positions=[i*gap - .53*width], widths=width, patch_artist=True, showfliers=False, boxprops=dict(facecolor=color_dict[domain], hatch='//'), medianprops=dict(color='black'))

        # Get upper whisker position
        whisker1_top = box1['whiskers'][1].get_ydata()[1]
        whisker2_top = box2['whiskers'][1].get_ydata()[1]

        # write p-value astriks
        sig_value = sig_p_values.get(domain, False)
        if sig_value is not None and sig_value < 0.05:
            astrisks = "*"
            if sig_value <= 0.01:
                astrisks = "**"
            if sig_value <= 0.0001:
                astrisks = "***"
            ax.text(i*gap + .53*width, whisker2_top, astrisks, ha='center')
        
        sig_value = baseline_sig_p_values.get(domain, False)
        if sig_value is not None and sig_value < 0.05:
            astrisks = "*"
            if sig_value <= 0.01:
                astrisks = "**"
            if sig_value <= 0.0001:
                astrisks = "***"
            ax.text(i*gap - .53*width, whisker1_top, '*', ha='center')

    # Set xticks and labels
    ax.set_xticks(range(0, len(unique_domains) * gap, gap))
    ax.set_xticklabels(unique_domains)

    plt.xlim([-1.1*width, i*gap+1.1*width])

    # Draw a horizontal line at the average c-index
    ax.axhline(0.5, color='r', linestyle='--')
    # ax.spines[['right', 'top']].set_visible(False)
   
    # Set the title and labels
    plt.title('Boxplot of c-index values for each domain', fontsize=24)
    plt.xlabel('Domain', fontsize=24)
    plt.ylabel('c-index values', fontsize=24)

    # Rotate x-axis labels if they overlap
    plt.xticks(fontsize=24) # rotation=45, 
    plt.yticks(fontsize=24)

    return fig


event_types = ['OS'] #  ['DFI', 'PFI', 'OS', 'DSS']
for event_type in event_types:
    censor_at = 120
    cv_experiment = f'CV_results_10years_median'
    cancer_types = [["BLCA"], ["BRCA"], ["CESC"], ["COAD", "READ"], ["ESCA"], ["GBM"], ["HNSC"], ["KICH"], ["KIRC"], ["KIRP"], ["LGG"], ["LIHC"], ["LUAD"], ["LUSC"], ["OV"], ["PAAD"], ["SKCM"], ["STAD"], ["UCEC"]]

    csv_files = {}
    sig_p_values = {}
    baseline_csv_files = {}
    baseline_sig_p_values = {}
    for cancer_type in cancer_types:
        csv_path = f"{cv_experiment}/CV_{cancer_type}/bootstrap_results_{cancer_type}_{event_type}_censor{censor_at}.csv"
        csv_files[''.join(cancer_type)] = csv_path
        baseline_csv_files[''.join(cancer_type)] = "baseline_results/"+csv_path

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

        # find the p-value related to this experiment
        directory = "baseline_results/" + f"{cv_experiment.replace('CV_results', 'CV_corrected_results')}/CV_{cancer_type}_Corrected/"
        km_path_pattern = f"cv_results_{cancer_type}_{event_type}_censor{censor_at}"
        # Walk through the directory
        p_value = None
        for file in os.listdir(directory):
            # Check if the file name matches the pattern
            if file.startswith(km_path_pattern) and file.endswith('.csv'):
                p_value = float(file.strip('.csv').split('pvalue')[-1])
                break
        baseline_sig_p_values[''.join(cancer_type)] = p_value

    # Call the function
    fig = plot_c_index(csv_files, sig_p_values, baseline_csv_files, baseline_sig_p_values)
    plt.title(f'Boxplot of c-index values, Event: {event_type}', fontsize=24)
    plt.ylim([-0.1,1.1])
    
    save_path = 'mosi'##cv_experiment + f'/bootstrap_cindex_{event_type}_censor{censor_at}'
    fig.savefig(save_path+".png", dpi=600, bbox_inches = 'tight', pad_inches = 0)
    fig.savefig(save_path+".pdf", dpi=600, bbox_inches = 'tight', pad_inches = 0)
