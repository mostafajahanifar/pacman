import os
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Define the root directories
features_root = "/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/mitosis_patches_embeddings"
points_root = "/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/mitosis_patches"

cancer_types = ["BRCA"]
# Function to load data
def load_data(features_root, points_root):
    feature_vectors = []
    point_lists = []
    case_info = []

    for type_dir in cancer_types:
        type_path = os.path.join(features_root, type_dir)
        if os.path.isdir(type_path):
            for case_dir in os.listdir(type_path):
                case_feature_path = os.path.join(type_path, case_dir, 'phikon_embedding.npy')
                case_point_path = os.path.join(points_root, type_dir, case_dir, 'point_list.npy')
                
                if os.path.exists(case_feature_path) and os.path.exists(case_point_path):
                    features = np.load(case_feature_path)
                    points = np.load(case_point_path)
                    
                    if features.size > 0 and points.size > 0:
                        feature_vectors.append(features)
                        point_lists.append(points)
                        case_info.append((type_dir, case_dir))

    return feature_vectors, point_lists, case_info

# Load the feature vectors and point lists
feature_vectors, point_lists, case_info = load_data(features_root, points_root)

# Concatenate all features into a single array for UMAP
if feature_vectors:
    all_features = np.concatenate(feature_vectors, axis=0)
    all_points = np.concatenate(point_lists, axis=0)

    # Apply UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    umap_embedding = reducer.fit_transform(all_features)

    # Cluster using DBSCAN (or any clustering algorithm of your choice)
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(umap_embedding)
    labels = clustering.labels_

    # Plot the UMAP embedding with cluster labels
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=labels, cmap='Spectral', s=5)
    plt.colorbar(scatter, boundaries=np.arange(min(labels), max(labels)+2)-0.5).set_ticks(np.arange(min(labels), max(labels)+1))
    plt.title('UMAP projection with DBSCAN clusters')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.show()

    # Output the cluster assignment along with point information
    clustered_points_info = []
    for idx, (point, label) in enumerate(zip(all_points, labels)):
        clustered_points_info.append({
            'point': point,
            'cluster_label': label,
            'case_info': case_info[idx // len(feature_vectors[0])]
        })

    # Example of how to access the clustered points information
    for info in clustered_points_info[:10]:  # Display the first 10 for brevity
        print(f"Point: {info['point']}, Cluster: {info['cluster_label']}, Case Info: {info['case_info']}")
else:
    print("No valid features or points found.")
