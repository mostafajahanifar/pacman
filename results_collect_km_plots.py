import os, glob
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from matplotlib.patches import Rectangle

cv_experiment = 'results_final/survival/CV_KM_10years_feat10sel_1000bs'

df = pd.read_excel(f'{cv_experiment}/aggregated_results.xlsx')
print(df)
event_types = ['PFI', 'DFI', 'OS', 'DSS']
cancer_types = [["ACC"], ["BLCA"], ["BRCA"], ["CESC"], ["COAD", "READ"], ["ESCA"], ["GBM", "LGG"], ["HNSC"], ["KIRC"], ["KIRP"], ["LIHC"], ["LUAD"], ["LUSC"], ["OV"], ["PAAD"], ["SKCM"], ["STAD"], ["UCEC"], ["MESO"], ["PRAD"], ["SARC"], ["TGCT"], ["THCA"]]
for event_type in event_types:
    
    i = 1
    num_rows = 6  # You are using 4 rows in your subplot grid
    num_cols = 4  # You are using 8 columns in your subplot grid

    fig = plt.figure(figsize=(num_cols*2.5, num_rows*2.5))

    for cancer_type in cancer_types:
        ctc = ''.join(cancer_type).upper()
        # if ctc in ["THCA", "HNSC", "STAD"]: # remove these ones for selected_PFI
        #     continue
        print(ctc, event_type)
        cindex_mean, cindex_std, pvalue = df.loc[(df['event_type'] == event_type) & (df['cancer_type'] == ctc), ['cindex_mean', 'cindex_std', 'pvalue']].values.tolist()[0]
        exp_folder = f"{cv_experiment}/CV_{cancer_type}_Corrected/"
        # Get a list of all files that match the pattern
        all_files = os.listdir(exp_folder)
        # Filter out the files that end with "_withCounts.png"
        filtered_files = [f for f in all_files if f.startswith(f"cv_results_{cancer_type}_{event_type}") and f.endswith(".png") and "_withCounts" not in f]
        
        ax = plt.subplot(num_rows, num_cols, i)
        try:
            km_plot = cv2.imread(exp_folder + filtered_files[0])[:,:,::-1]  # Load the image
            
            # Display cropped image
            ax.imshow(km_plot)
            i += 1
            subtitle = f"C-index: {cindex_mean:.2f} ({cindex_std:.2f})"
        except:
            subtitle = f"{ctc} - Failed to fit the model."
            print('No experiment exists', ctc, event_type)

        ax.axis('off')

    fig.subplots_adjust(hspace=0, wspace=0)

    save_path=f"{cv_experiment}/aggregated_km_plots_{event_type}"
    fig.savefig(save_path+'.png', dpi=600, bbox_inches='tight', pad_inches=0)
