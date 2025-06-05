import os
import numpy as np
import pandas as pd

from feature_extraction.feature_utils import count_in_bbox, flat_mesh_grid_coord, checking_wsi_info
from tiatoolbox.wsicore.wsireader import OpenSlideWSIReader

HOTSPOT_WINDOW = 3
skip_hs = 1
skip_wsi = 5 

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
    
    # Create a mask for points within the specified window and with type == 2
    mask = (x_coords >= x1) & (x_coords <= x2) & (y_coords >= y1) & (y_coords <= y2)
    
    # Count the number of points that satisfy the mask
    count = np.sum(mask)
    return count

def atypical_mitosis_hotspot(cen_list, wsi_path):
    if isinstance(wsi_path, str):
        wsi = OpenSlideWSIReader.open(wsi_path)
        info_dict = wsi.info.as_dict()
        wsi_check = checking_wsi_info(info_dict)
    elif isinstance(wsi_path, dict):
        out_mag = 20.0 if np.abs(wsi_path['mpp']-0.5)<np.abs(wsi_path['mpp']-0.25) else 40.0
        wsi_check = out_mag, wsi_path['mpp']
        # set a default slide size 
        info_dict = {'slide_dimensions': (250000, 150000)}
    else:
        wsi_check = -1

    if wsi_check == -1:
        # writing default features to feature dictionary
        feature_dict = dict()
        feature_dict['aty_ahotspot_count'] = -1
        feature_dict['aty_ahotspot_ratio'] = -1
        feature_dict['mit_ahotspot_count'] = -1
        feature_dict['aty_ahotspot_x1'] = -1
        feature_dict['aty_ahotspot_y1'] = -1
        feature_dict['aty_ahotspot_x2'] = -1
        feature_dict['aty_ahotspot_y2'] = -1

        return feature_dict

    wsi_power, wsi_mpp = wsi_check
    # calculating the windows size in pixels and baseline resolution
    bs = int(np.round(1000*np.sqrt(HOTSPOT_WINDOW)/wsi_mpp)) # bound size in pixels
    stride = bs//6

    img_h = info_dict['slide_dimensions'][1]
    img_w = info_dict['slide_dimensions'][0]

    if cen_list.shape[0] == 0: # no mitosis
        # writing default features to feature dictionary
        feature_dict = dict()
        feature_dict['aty_ahotspot_count'] = 0
        feature_dict['aty_ahotspot_ratio'] = 0
        feature_dict['mit_ahotspot_count'] = 0
        feature_dict['aty_ahotspot_x1'] = 0
        feature_dict['aty_ahotspot_y1'] = 0
        feature_dict['aty_ahotspot_x2'] = 0
        feature_dict['aty_ahotspot_y2'] = 0

        return feature_dict

    # find the bounds to check for mitosis count
    output_x_list = np.arange(0, int(img_w), stride)
    output_y_list = np.arange(0, int(img_h), stride)
    output_tl_list = flat_mesh_grid_coord(output_x_list, output_y_list)
    output_br_list = output_tl_list + bs

    bounds = np.concatenate([output_tl_list, output_br_list], axis=1)

    # finding atypical mitotic count in each bound and put in canvas
    counts = []
    atypical_cen_list = cen_list[cen_list[:,2]==2]
    for bound in bounds:
        # b_count = count_in_bbox(bound, atypical_cen_list)
        b_count = count_points_in_window(atypical_cen_list, bound[0], bound[1], bound[2], bound[3])
        counts.append(b_count)

    i_max = np.argmax(counts)
    atypical_mitosis_count = int(counts[i_max])
    # skip the first detection as FP
    atypical_mitosis_count = max(atypical_mitosis_count-skip_hs , 0)
    cb = list(bounds[i_max])

    all_mitosis_count = count_points_in_window(cen_list, cb[0], cb[1], cb[2], cb[3])


    # writing features to feature dictionary
    feature_dict = dict()
    feature_dict['aty_ahotspot_count'] = atypical_mitosis_count
    feature_dict['aty_ahotspot_ratio'] = atypical_mitosis_count/(all_mitosis_count+0.0000001)
    feature_dict['mit_ahotspot_count'] = all_mitosis_count
    feature_dict['aty_ahotspot_x1'] = cb[0]
    feature_dict['aty_ahotspot_y1'] = cb[1]
    feature_dict['aty_ahotspot_x2'] = cb[2]
    feature_dict['aty_ahotspot_y2'] = cb[3]
    print(feature_dict)
    return feature_dict

mitosis_feats = pd.read_csv('/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/tcga_features_final_ClusterByCancerNew.csv')
directory_path = '/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/mitosis_cls_candidates_final(4classes)'
slide_root = '/mnt/web-public/tcga/'
npy_files_dict = find_npy_files(directory_path)


# Initialize a new column for the number of rows in the numpy files
mitosis_feats['aty_wsi_count'] = 0
mitosis_feats['aty_wsi_ratio'] = 0
mitosis_feats['aty_hotspot_count'] = 0
mitosis_feats['aty_hotspot_ratio'] = 0

mitosis_feats['aty_ahotspot_count'] = 0
mitosis_feats['aty_ahotspot_ratio'] = 0
mitosis_feats['mit_ahotspot_count'] = 0
mitosis_feats['aty_ahotspot_x1'] = 0
mitosis_feats['aty_ahotspot_y1'] = 0
mitosis_feats['aty_ahotspot_x2'] = 0
mitosis_feats['aty_ahotspot_y2'] = 0


# Iterate through the DataFrame and update the new column
for index, row in mitosis_feats.iterrows():

    slide_name = row['slide']
    print(f"working on {index+1}/{len(mitosis_feats)}: {slide_name}")
    if slide_name in npy_files_dict:
        npy_file_path = npy_files_dict[slide_name]
        npy_data = np.load(npy_file_path)
        aty_wsi_count = max(len(np.where(npy_data[:,2]==2)[0])-skip_wsi, 0)
        mitosis_feats.at[index, 'aty_wsi_count'] = aty_wsi_count
        mitosis_feats.at[index, 'aty_wsi_ratio'] = aty_wsi_count/(mitosis_feats.loc[index, 'mit_wsi_count']+0.00000001)

        x1, y1 = row['mit_hotspot_x1'], row['mit_hotspot_y1']
        x2, y2 = row['mit_hotspot_x2'], row['mit_hotspot_y2']
        
        atyp_hotspot_count = count_points_in_window(npy_data[npy_data[:,2]==2], x1, y1, x2, y2)
        aty_wsi_count = max(atyp_hotspot_count-skip_hs, 0)
        mitosis_feats.at[index, 'aty_hotspot_count'] = aty_wsi_count
        mitosis_feats.at[index, 'aty_hotspot_ratio'] = aty_wsi_count/(mitosis_feats.loc[index, 'mit_hotspot_count']+0.00000001)

        # add features related to `atypical hotspot`
        wsi_path = slide_root + slide_name + ".svs"
        feature_dict = atypical_mitosis_hotspot(npy_data, wsi_path)

        mitosis_feats.at[index, 'aty_ahotspot_count'] = feature_dict['aty_ahotspot_count']
        mitosis_feats.at[index, 'aty_ahotspot_ratio'] = feature_dict['aty_ahotspot_ratio']
        mitosis_feats.at[index, 'mit_ahotspot_count'] = feature_dict['mit_ahotspot_count']
        mitosis_feats.at[index, 'aty_ahotspot_x1'] = feature_dict['aty_ahotspot_x1']
        mitosis_feats.at[index, 'aty_ahotspot_y1'] = feature_dict['aty_ahotspot_y1']
        mitosis_feats.at[index, 'aty_ahotspot_x2'] = feature_dict['aty_ahotspot_x2']
        mitosis_feats.at[index, 'aty_ahotspot_y2'] = feature_dict['aty_ahotspot_y2']

mitosis_feats.to_csv('/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/tcga_features_final_ClusterByCancer_withAtypical.csv', index=False)