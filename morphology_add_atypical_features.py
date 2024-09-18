import os
import numpy as np
import pandas as pd

def find_npy_files(directory):
    npy_files = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.npy'):
                file_name = os.path.splitext(file)[0]
                file_path = os.path.join(root, file)
                npy_files[file_name] = file_path
    return npy_files

def count_points_in_window(npy_data, x1, y1, x2, y2):
    # Extract x, y, and type columns
    x_coords = npy_data[:, 0]
    y_coords = npy_data[:, 1]
    types = npy_data[:, 2]
    
    # Create a mask for points within the specified window and with type == 2
    mask = (x_coords >= x1) & (x_coords <= x2) & (y_coords >= y1) & (y_coords <= y2) & (types == 2)
    
    # Count the number of points that satisfy the mask
    count = np.sum(mask)
    return count

mitosis_feats = pd.read_csv('/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/tcga_features_final_ClusterByCancer.csv')
directory_path = '/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/mitosis_cls_candidates_final(4classes)'
npy_files_dict = find_npy_files(directory_path)


# Initialize a new column for the number of rows in the numpy files
mitosis_feats['aty_wsi_count'] = 0
mitosis_feats['aty_wsi_ratio'] = 0
mitosis_feats['aty_hotspot_count'] = 0
mitosis_feats['aty_hotspot_ratio'] = 0

# Iterate through the DataFrame and update the new column
for index, row in mitosis_feats.iterrows():
    slide_name = row['slide']
    print(f"working on {index+1}/{len(mitosis_feats)}: {slide_name}")
    if slide_name in npy_files_dict:
        npy_file_path = npy_files_dict[slide_name]
        npy_data = np.load(npy_file_path)
        mitosis_feats.at[index, 'aty_wsi_count'] = len(np.where(npy_data[:,2]==2)[0])
        mitosis_feats.at[index, 'aty_wsi_ratio'] = len(np.where(npy_data[:,2]==2)[0])/mitosis_feats.loc[index, 'mit_wsi_count']

        x1, y1 = row['mit_hotspot_x1'], row['mit_hotspot_y1']
        x2, y2 = row['mit_hotspot_x2'], row['mit_hotspot_y2']
        
        atyp_hotspot_count = count_points_in_window(npy_data, x1, y1, x2, y2)
        mitosis_feats.at[index, 'aty_hotspot_count'] = atyp_hotspot_count
        mitosis_feats.at[index, 'aty_hotspot_ratio'] = atyp_hotspot_count/mitosis_feats.loc[index, 'mit_hotspot_count']

mitosis_feats.to_csv('/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/tcga_features_final_ClusterByCancer_withAtypical.csv', index=False)