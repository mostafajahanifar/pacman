import os
import numpy as np
from tqdm import tqdm

# Define the root directories
features_root = "/home/u2070124/lsf_workspace/Data/Data/pancancer/mitosis_patches_embeddings"
points_root = "/home/u2070124/lsf_workspace/Data/Data/pancancer/mitosis_patches"

# Function to load data and capture metadata
def load_data(features_root, points_root):
    feature_vectors = []
    point_lists = []
    type_list = []
    case_list = []
    index_list = []

    for type_dir in os.listdir(features_root):
        type_path = os.path.join(features_root, type_dir)
        if os.path.isdir(type_path):
            print(f"Working on {type_dir}")
            case_folders = os.listdir(type_path)
            for case_dir in tqdm(case_folders, total=len(case_folders), desc=type_dir):
                case_feature_path = os.path.join(type_path, case_dir, 'phikon_embedding.npy')
                case_point_path = os.path.join(points_root, type_dir, case_dir, 'point_list.npy')
                
                if os.path.exists(case_feature_path) and os.path.exists(case_point_path):
                    features = np.load(case_feature_path)
                    points = np.load(case_point_path)
                    
                    if features.size > 0 and points.size > 0:
                        feature_vectors.append(features)
                        point_lists.append(points)
                        type_list.extend([type_dir] * features.shape[0])
                        case_list.extend([case_dir] * features.shape[0])
                        index_list.extend(list(range(features.shape[0])))

    return feature_vectors, point_lists, type_list, case_list, index_list

# Load the feature vectors and point lists along with metadata
feature_vectors, point_lists, type_list, case_list, index_list = load_data(features_root, points_root)

# Concatenate all features and points into single arrays
all_features = np.concatenate(feature_vectors, axis=0)
all_points = np.concatenate(point_lists, axis=0)

# Save the data and metadata
np.save(os.path.join(features_root, 'clustering_all_features.npy'), all_features)
np.save(os.path.join(features_root, 'clustering_all_points.npy'), all_points)
np.save(os.path.join(features_root, 'clustering_all_type_list.npy'), np.array(type_list))
np.save(os.path.join(features_root, 'clustering_all_case_list.npy'), np.array(case_list))
np.save(os.path.join(features_root, 'clustering_all_index_list.npy'), np.array(index_list))
