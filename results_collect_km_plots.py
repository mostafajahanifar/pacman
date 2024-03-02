import os, glob
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from matplotlib.patches import Rectangle

cv_experiment = 'baseline_results/CV_corrected_results_NoCensor_median'

df = pd.read_excel(f'{cv_experiment}/aggregated_results.xlsx')
event_types = ['DFI', 'PFI', 'OS', 'DSS']
cancer_types = [["BLCA"], ["BRCA"], ["CESC"], ["COAD", "READ"], ["ESCA"], ["GBM"], ["HNSC"], ["KICH"], ["KIRC"], ["KIRP"], ["LGG"], ["LIHC"], ["LUAD"], ["LUSC"], ["OV"], ["PAAD"], ["SKCM"], ["STAD"], ["UCEC"]]
for event_type in event_types:
    fig = plt.figure(figsize=(20, 12))
    i = 1
    for cancer_type in cancer_types:
        ctc = ''.join(cancer_type).upper()
        cindex_mean, cindex_std, pvalue = df.loc[(df['event_type'] == event_type) & (df['cancer_type'] == ctc), ['cindex_mean', 'cindex_std', 'pvalue']].values.tolist()[0]
        exp_folder = f"{cv_experiment}/CV_{cancer_type}_Corrected/"
        # Get a list of all files that match the pattern
        all_files = os.listdir(exp_folder)
        # Filter out the files that end with "_withCounts.png"
        filtered_files = [f for f in all_files if f.startswith(f"cv_results_{cancer_type}_{event_type}") and f.endswith(".png") and not f.endswith("_withCounts.png")]
        ax = plt.subplot(4, 5, i)
        try:
            km_plot = cv2.imread(exp_folder + filtered_files[0])[:,:,::-1]
            ax.imshow(km_plot)
            subtitle = f"C-index: {cindex_mean:.2f} ({cindex_std:.2f})"
        except:
            subtitle = f"{ctc} - Failed to fit the model."
            print('No experiment exists')
        
        color = 'green' if pvalue <= 0.05 else 'red'
        t = ax.text(0.5, -0.1, subtitle, size=9, ha="center", transform=ax.transAxes)
        t.set_bbox(dict(facecolor=color, alpha=0.5, edgecolor=color))
        # ax.text(0.5, -0.1, subtitle, size=9, ha="center", transform=ax.transAxes)
        ax.axis('off')
        
        i += 1

    save_path=f"{cv_experiment}/aggregated_km_plots_{event_type}"
    plt.savefig(save_path+'.png', dpi=600, bbox_inches = 'tight', pad_inches = 0)
