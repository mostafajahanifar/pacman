import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

mit_temp = "Hot"
valid_cancer_for_event = {
    "PFI":['GBMLGG', 'SKCM', 'LUAD', 'HNSC', 'LIHC', 'BLCA', 'COADREAD', 'KIRC',
       'BRCA', 'LUSC', 'STAD', 'SARC', 'UCEC', 'PAAD', 'ESCA', 'OV', 'CESC',
       'KIRP', 'MESO', 'TGCT', 'UCS', 'ACC', 'PCPG'],
    "OS": ['GBMLGG', 'HNSC', 'LUSC', 'SKCM', 'BLCA', 'KIRC', 'LUAD', 'BRCA',
       'STAD', 'LIHC', 'COADREAD', 'PAAD', 'SARC', 'UCEC', 'OV', 'ESCA',
       'CESC', 'MESO', 'UCS', 'ACC', 'DLBC'],
    "DSS": ['GBMLGG', 'SKCM', 'HNSC', 'BLCA', 'KIRC', 'LUAD', 'STAD', 'LUSC',
       'LIHC', 'BRCA', 'PAAD', 'COADREAD', 'SARC', 'OV', 'UCEC', 'CESC',
       'ESCA', 'MESO', 'UCS', 'ACC'],
    "DFI": ['LIHC', 'LUAD', 'BRCA', 'LUSC', 'SARC', 'UCEC', 'STAD', 'BLCA',
            'TGCT', 'OV', 'PAAD', 'ESCA'],
}

# Function to perform univariate analysis and calculate c-index
def cox_univariate_analysis(df, feature, event_col, time_col):
    # Initialize the Cox proportional hazard model
    cph = CoxPHFitter()

    data = df[[feature, time_col, event_col]]
    data = data.dropna(how="any")  # Remove missing values

    # Normalize feature
    data[feature] = (data[feature] - data[feature].mean()) / (data[feature].std() + 1e-9)

    # Fit the Cox model to the dataframe
    cph.fit(data, duration_col=time_col, event_col=event_col)

    # Predict the risk scores
    risk_scores = cph.predict_partial_hazard(data)

    # Calculate the c-index
    c_index = concordance_index(data[time_col], -risk_scores, data[event_col])

    return c_index

# Function to perform univariate analysis across different event types and cancer types
def analyze_univariate_c_index_by_event_and_cancer(df, selected_feats):
    results = {}
    event_types = ["OS", "PFI", "DSS", "DFI"]  # Define event types

    for event_type in event_types:
        # For each event type, prepare the event and event time columns
        event_col = event_type
        time_col = f"{event_type}.time"

        # Store results for each event type
        results[event_type] = {}
        cancer_types = sorted(valid_cancer_for_event[event_type])
        for cancer in cancer_types:
            print(f"{event_type} - {cancer}")
            # Filter the dataframe by cancer type
            cancer_df = df[df['type'] == cancer]
            cancer_df = cancer_df[cancer_df["temperature"] == mit_temp]

            # Store c-indices for each feature in this cancer type
            c_indices = {}
            for feat in selected_feats:
                try:
                    # Perform univariate analysis to calculate c-index for each feature
                    c_index = cox_univariate_analysis(cancer_df, feat, event_col, time_col)
                    c_indices[feat] = c_index
                except Exception as e:
                    print(f"Failed {event_type} - {cancer} - {feat}: because: {e}")
                    c_indices[feat] = None

            results[event_type][cancer] = c_indices

    return results

# Function to plot bar plots for c-index
def plot_c_index_barplots(results, selected_feats, feat_to_names):
    for event_type, cancers in results.items():
        # Prepare the plot for each event type
        fig_width = 0.7 + len(valid_cancer_for_event[event_type])*(6/23)
        fig, ax = plt.subplots(figsize=(fig_width, 1))
        
        # Data for plotting
        cancer_types = list(cancers.keys())
        feature_names = [feat_to_names[feat] for feat in selected_feats]
        
        # For each cancer, we plot three bars (one for each feature)
        bar_width = 0.2
        index = range(len(cancer_types))  # Position of ticks on the x-axis
        
        for i, feat in enumerate(selected_feats):
            c_indices = [cancers[cancer][feat] if feat in cancers[cancer] else None for cancer in cancer_types]
            ax.bar([x + i * bar_width for x in index], c_indices, bar_width, label=feat_to_names[feat])
        
        # add a horinzontal line at y=0.5
        ax.axhline(y=0.5, color='lightgray', linestyle='--', zorder=0, linewidth=1)
        # Set labels, title, and legend
        # ax.set_xlabel('Cancer Type')
        ax.set_ylabel('C-Index')
        ax.set_title(f'Mitotic-{mit_temp} ; {event_type} Event')
        ax.set_xticks([x + bar_width for x in index])
        ax.set_xticklabels(cancer_types, rotation=45, ha='right')
        # ax.legend()

        # Show the plot
        # plt.tight_layout()
        plt.savefig(f"results_final/morphology/univariate/cindex_{mit_temp}_{event_type}.png", dpi=600, bbox_inches = 'tight', pad_inches = 0.01)


df = pd.read_csv('/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/tcga_features_final_ClusterByCancerNew_withAtypicalNew.csv')

selected_feats = [
    # "mit_hotspot_count",
    "aty_hotspot_count",
    "aty_hotspot_ratio",
    "aty_wsi_ratio",
]

feat_to_names = {
    "aty_hotspot_count": "Hotspot Atypical Count (HAC)",
    "aty_hotspot_ratio": "Hotspot Atypical Fraction (HAF)",
    "aty_wsi_ratio": "Slide Atypical Fraction (WAF)",
    "mit_hotspot_count": "Hotspot Mitotic Count (HSC)"
    }


# Run the univariate analysis and plot the c-index
univariate_results = analyze_univariate_c_index_by_event_and_cancer(df, selected_feats)

# Plot the c-index bar plots
plot_c_index_barplots(univariate_results, selected_feats, feat_to_names)
