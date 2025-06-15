import os

import pandas as pd
from lifelines import CoxPHFitter

from pacman.config import DATA_DIR, RESULTS_DIR, SURV_CANCERS

print(7 * "=" * 7)
print("Running Multivariate Survival Analysis using AMFs")
print(7 * "=" * 7)

mit_temp = "all"

save_root = f"{RESULTS_DIR}/morphology/survival_multivariate/"
os.makedirs(save_root, exist_ok=True)


valid_cancer_for_event = {
    "PFI": [
        "GBMLGG",
        "SKCM",
        "LUAD",
        "HNSC",
        "LIHC",
        "BLCA",
        "COADREAD",
        "KIRC",
        "BRCA",
        "LUSC",
        "STAD",
        "SARC",
        "UCEC",
        "PAAD",
        "ESCA",
        "OV",
        "CESC",
        "KIRP",
        "THCA",
        "MESO",
        "TGCT",
        "UCS",
        "ACC",
        "CHOL",
        "THYM",
        "KICH",
        "PCPG",
    ],
    "OS": [
        "GBMLGG",
        "HNSC",
        "LUSC",
        "SKCM",
        "BLCA",
        "KIRC",
        "LUAD",
        "BRCA",
        "STAD",
        "LIHC",
        "COADREAD",
        "PAAD",
        "SARC",
        "UCEC",
        "OV",
        "ESCA",
        "CESC",
        "MESO",
        "UCS",
        "ACC",
        "CHOL",
        "THCA",
        "KICH",
        "DLBC",
    ],
    "DSS": [
        "GBMLGG",
        "SKCM",
        "HNSC",
        "BLCA",
        "KIRC",
        "LUAD",
        "STAD",
        "LUSC",
        "LIHC",
        "BRCA",
        "PAAD",
        "COADREAD",
        "SARC",
        "OV",
        "UCEC",
        "CESC",
        "ESCA",
        "MESO",
        "UCS",
        "ACC",
        "CHOL",
    ],
    "DFI": [
        "LIHC",
        "LUAD",
        "BRCA",
        "LUSC",
        "SARC",
        "UCEC",
        "STAD",
        "BLCA",
        "KIRP",
        "TGCT",
        "OV",
        "PAAD",
        "ESCA",
        "CHOL",
    ],
}


# Function to perform Cox Proportional Hazard analysis and extract HR, CI, and p-value
def cox_multivariate_analysis(df, selected_feats, event_col, time_col):
    # Initialize the Cox proportional hazard model
    cph = CoxPHFitter()

    data = df[selected_feats + [time_col, event_col]]
    data = data.dropna(how="any")  # Remove missing values

    # normalize features
    data[selected_feats] = (data[selected_feats] - data[selected_feats].mean()) / (
        data[selected_feats].std() + 0.000000001
    )

    # Fit the Cox model to the dataframe
    cph.fit(data, duration_col=time_col, event_col=event_col)

    # Extract the hazard ratios, confidence intervals, and p-values
    summary = cph.summary[
        ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]
    ]

    # Rename the columns to match the desired output format
    summary.columns = ["HR", "95% CI Lower", "95% CI Upper", "P-value"]

    return summary


# Main function to perform analysis across different event types and cancers
def analyze_survival_by_event_and_cancer(df, selected_feats):
    results = {}
    event_types = ["PFI", "DSS", "OS", "DFI"]  # Define event types

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
            cancer_df = df[df["type"] == cancer]
            if mit_temp in ["Hot", "Cold"]:
                cancer_df = cancer_df[cancer_df["temperature"] == "Hot"]

            try:
                # Perform multivariate analysis using Cox proportional hazards
                summary = cox_multivariate_analysis(
                    cancer_df, selected_feats, event_col, time_col
                )

                # Save the result for each cancer type
                results[event_type][cancer] = summary
            except Exception as e:
                print(f"Failed {event_type} - {cancer}: because: {e}")

    return results


# Function to save the results into an Excel file with proper formatting
def save_results_to_excel(results, save_path):
    # Create an Excel writer object
    with pd.ExcelWriter(save_path, engine="xlsxwriter") as writer:
        for event_type, cancers in results.items():
            # Create a new sheet for each event type
            sheet_name = event_type
            all_cancers_result = pd.DataFrame()

            for cancer, summary in cancers.items():
                # Add the cancer type as a bold label row
                cancer_heading = pd.DataFrame([[f"{cancer}"]], columns=["Feature"])
                cancer_heading.index = [""]  # Set an empty index for formatting

                # Append the cancer heading and then the feature rows for that cancer
                summary["Feature"] = selected_feats  # Set the Feature column

                # Append the cancer heading and the summary
                all_cancers_result = pd.concat(
                    [all_cancers_result, cancer_heading, summary], axis=0
                )

            # Write the combined dataframe to an Excel sheet
            all_cancers_result.to_excel(writer, sheet_name=sheet_name, index=None)

            # Access the sheet via the writer object to apply formatting
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]

            # Set column widths and apply formatting
            worksheet.set_column("A:A", 30)  # Set the width for feature/label column
            worksheet.set_column("B:D", 15)  # Set width for hazard ratio and CI columns
            worksheet.set_column("E:E", 10)  # Set width for p-value column


# reading mitotic features
mitosis_feats = pd.read_excel(os.path.join(DATA_DIR, "ST1-tcga_mtfs.xlsx"))

fixed_features = ["HSC"]  # can add other covariates: age, sex, etc.
AMFs = [
    "AMAH",
    "AFW",
    "AMH",
]

experiments_dict = {AMF: [AMF] + fixed_features for AMF in AMFs}
experiments_dict["all"] = AMFs
# create selected features list

for AMF, selected_feats in experiments_dict.items():
    print(f"Running multivariate analyses using feature set: {selected_feats}")
    # Run the analysis
    results = analyze_survival_by_event_and_cancer(mitosis_feats, selected_feats)

    # Save the results in a formatted Excel sheet
    save_path = save_root + f"multivariate_analysis_{mit_temp}_{AMF}.xlsx"
    save_results_to_excel(results, save_path)
