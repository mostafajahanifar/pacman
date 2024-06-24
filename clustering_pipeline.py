import os
import numpy as np
import pandas as pd
# import umap
# import hdbscan
from tqdm import tqdm
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

# Define the directories
input_dir = "/home/u2070124/lsf_workspace/Data/Data/pancancer/mitosis_patches_embeddings"
output_dir = "/home/u2070124/lsf_workspace/Data/Data/pancancer/mitosis_patches_clustered"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the preprocessed data
all_features = np.load(os.path.join(input_dir, 'mitosis_patches_embeddings_uncorrelated_normalized.npy'))
undersampled_data = np.load(os.path.join(input_dir, 'mitosis_patches_embeddings_uncorrelated_normalized_undersampled.npy'))
all_points = np.load(os.path.join(input_dir, 'clustering_all_points.npy'))
type_list = np.load(os.path.join(input_dir, 'clustering_all_type_list.npy'))
case_list = np.load(os.path.join(input_dir, 'clustering_all_case_list.npy'))
index_list = np.load(os.path.join(input_dir, 'clustering_all_index_list.npy'))
print("Data loaded successfully")

# Train classifier on balanced data using best best_num_clusters
best_num_clusters = 7
kmeans = KMeans(n_clusters=best_num_clusters, n_init=10, random_state=42)
kmeans = kmeans.fit(undersampled_data)
print("Training KMeans done")

# predict clusters for full data
cluster_labels = kmeans.predict(all_features)
np.save(os.path.join(input_dir, f'KMean{best_num_clusters}_clustering_all_labels.npy'), cluster_labels)
print("KMeans prediction done")

# Create a DataFrame with point information and cluster labels
df = pd.DataFrame({
    'type': type_list,
    'case': case_list,
    'x': all_points[:, 0],
    'y': all_points[:, 1],
    'score': all_points[:, 2],
    'cluster': cluster_labels
})
print("Dataframe is created")

# Free up memory by deleting large numpy arrays
del all_features, all_points, type_list, case_list, index_list, undersampled_data

# Save clustered points in their respective directories
print("start saving the results ...")
for (typ, case), group in tqdm(df.groupby(['type', 'case'])):
    save_dir = os.path.join(output_dir, typ, case)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'point_list_clustered.npy')
    np.save(save_path, group[['x', 'y', 'score', 'cluster']].to_numpy())

print(f"Clustered points saved in {output_dir}")

df.to_csv(os.path.join(input_dir, f'KMean{best_num_clusters}_clustering_all_labels_data.csv'), index=None)