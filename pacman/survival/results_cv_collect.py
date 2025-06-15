import os

import pandas as pd

from pacman.config import DATA_DIR, RESULTS_DIR, SURV_CANCERS

print(7 * "=" * 7)
print("Collecting results of corss-validated survival analyses")
print(7 * "=" * 7)

censor_at = 120
cv_experiment = f"{RESULTS_DIR}/survival/kmplots_cv/"

results_dict = {
    "censoring": [],
    "event_type": [],
    "cancer_type": [],
    "success": [],
    "pvalue": [],
    "significant": [],
    "cindex_mean": [],
    "cindex_std": [],
}

event_types = ["DFI", "PFI", "OS", "DSS"]
for event_type in event_types:
    csv_files = {}
    sig_p_values = {}
    for cancer_type in SURV_CANCERS:
        results_dict["censoring"].append(censor_at)
        results_dict["event_type"].append(event_type)
        results_dict["cancer_type"].append(cancer_type)

        # find the p-value related to this experiment
        directory = f"{cv_experiment}/{cancer_type}/"
        km_path_pattern = f"kmplot_cv_{cancer_type}_{event_type}_censor{censor_at}"
        # Walk through the directory
        p_value = None
        csv_df = None
        for file in os.listdir(directory):
            # Check if the file name matches the pattern
            if file.startswith(km_path_pattern) and file.endswith(".csv"):
                p_value = float(file.strip(".csv").split("pvalue")[-1])
                csv_df = pd.read_csv(directory + file)
                break
        if p_value is not None:
            sig_p_value = p_value < 0.05
            results_dict["pvalue"].append(p_value)
            results_dict["significant"].append(int(sig_p_value))
            results_dict["cindex_mean"].append(csv_df["c_index"].mean())
            results_dict["cindex_std"].append(csv_df["c_index"].std())
            results_dict["success"].append(0 if len(csv_df) < 3 else 1)
        else:
            results_dict["pvalue"].append(1)
            results_dict["significant"].append(0)
            results_dict["cindex_mean"].append(0)
            results_dict["cindex_std"].append(0)
            results_dict["success"].append(0)


results_df = pd.DataFrame(results_dict)
results_df.to_excel(f"{cv_experiment}/cv_aggregated_results.xlsx")

df = results_df
pivot_mean = df.pivot(index="cancer_type", columns="event_type", values="cindex_mean")
pivot_std = df.pivot(index="cancer_type", columns="event_type", values="cindex_std")
pivot_success = df.pivot(index="cancer_type", columns="event_type", values="success")
pivot_significant = df.pivot(
    index="cancer_type", columns="event_type", values="significant"
)

# Format the cells
formatted_df = (
    pivot_mean.applymap("{:.3f}".format)
    + " ("
    + pivot_std.applymap("{:.2f}".format)
    + ")"
)

# Replace the cells based on 'success' and 'significant'
formatted_df[pivot_success == 0] = "CF"
formatted_df[pivot_significant == 0] = "NS"
formatted_df[(pivot_success == 0) & (pivot_significant == 0)] = "NS/CF"

# Calculate the mean for each column and append it to the DataFrame
mean_row = (
    pivot_mean[(pivot_success == 1) & (pivot_significant == 1)]
    .mean()
    .apply("{:.3f}".format)
)
formatted_df.loc["Average"] = mean_row

# Save the DataFrame to an Excel file
formatted_df.to_excel(f"{cv_experiment}/cv_summary_table.xlsx")
