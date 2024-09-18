import pandas as pd
import cv2
import numpy as np
import pickle
import random
import string

def generate_random_name(length=16):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))


MASK_RATIO = 16

def checking_wsi_info(info_dict):
    '''This function checks the info dict returned by TIAToolbox and complete the mpp and magnification
       information based on some huristics. Slides with less than 3 levels are neglected as well as the
       ones with no mpp nor magnification information.
    '''
    # if info_dict['level_count']<3: # low number of levels
    #     return -1 # reject this wsi
    
    if info_dict['objective_power']==None and info_dict['mpp'][0]==None: # low number of levels
        return -1 # reject this wsi

    if info_dict['objective_power']==None and info_dict['mpp'][0]!=None: # decide objective based on mpp
        mpp_diff_25 = np.abs(info_dict['mpp'][0]-0.25)
        mpp_diff_50 = np.abs(info_dict['mpp'][0]-0.5)

        out_mag = 20.0 if mpp_diff_50<mpp_diff_25 else 40.0
        out_mpp = info_dict['mpp'][0]
    
    if info_dict['objective_power']!=None and info_dict['mpp'][0]==None: # decide objective based on mpp
        mag_diff_20 = np.abs(info_dict['objective_power']-20)
        mag_diff_40 = np.abs(info_dict['objective_power']-40)

        out_mpp = 0.25 if mag_diff_40<mag_diff_20 else 0.5
        out_mag = info_dict['objective_power']
    
    if info_dict['objective_power']!=None and info_dict['mpp'][0]!=None: # wsi is alright retrun the normal things
        out_mag, out_mpp = info_dict['objective_power'], info_dict['mpp'][0]

    return out_mag, out_mpp

def filter_candidates(mitosis_candidates, tumour_mask, candidate_thresh=100, coord_ratio=1, filter_by_mask=False):
    '''Filters mitosis candidates based on their score and whether they are in tumour region.
    
    Mitosis candidates with scores higher than `candidate_thresh` will be tested for tumour
    inclusion. Prior to check if a candidate is in tumour region, it's coordinates are scaled
    by `coord_ratio` to compensate for tumour mask size compression. The output is an array of
    shape Nx2 where columns are x,y cordinates of candidates.
    '''
    # filter based on candidate_thresh
    mitosis_candidates = mitosis_candidates[mitosis_candidates['score']>candidate_thresh]
    mitosis_candidates.reset_index(drop=True, inplace=True)
    
    # filter by tumour mask
    # First, scale the coordinates
    if tumour_mask is not None and filter_by_mask==True:
        x_can= np.uint32(np.floor(mitosis_candidates['x'].values * coord_ratio))
        y_can = np.uint32(np.floor(mitosis_candidates['y'].values * coord_ratio))
        # x_can= np.uint32(np.floor(mitosis_candidates[:, 0] * coord_ratio))
        # y_can = np.uint32(np.floor(mitosis_candidates[:, 1] * coord_ratio))
        x_can = np.clip(x_can, a_min=0, a_max=tumour_mask.shape[1]-1)
        y_can = np.clip(y_can, a_min=0, a_max=tumour_mask.shape[0]-1)
        mask_values = tumour_mask[y_can, x_can]
        mitosis_candidates = mitosis_candidates.iloc[mask_values>0]
        mitosis_candidates.reset_index(drop=True, inplace=True)
    return mitosis_candidates

def feature_extractor_tcga(mitosis_path, wsi_path, graph_path, feature_funcs, temp_save_path=None):
    '''Feature extraction from a single slide mitosis and tumour mask predictions.
    
    This functin is to be used in multiprocessing and is designed to work in extendable
    fashion where feature extraction function is provided as an argument. This design
    help adding new feature to the previouisely extracted features.
    feature_funcs:
        should be a list of functions that accept mitosis candidates (and tumour mask)
        and returns a feacture dictionary with keys being feature names and their
        values as dictinary values.
    tumour_path:
        Whether to filter mitosis candidates based on tumour mask if `tumour_path`
        is a valid path, otherwise, it should be set to `None`.
    '''
    # reading inputs
    # mitosis_candidates = pd.read_csv(mitosis_path)
    temp = np.load(mitosis_path)
    mitosis_candidates = pd.DataFrame(temp, columns=['x','y','type'])
    
        
    # # filter mitosis candidates
    # mitosis_candidates = filter_candidates(mitosis_candidates, None, candidate_thresh=0.5, coord_ratio=1/MASK_RATIO, filter_by_mask=False)

    # getting mpp if wsi_path is None
    if wsi_path is None:
        wsi_path = {'mpp': 0.25 if 'Philips' in mitosis_path else 0.243}
    
    # extract features
    feature_names = ['slide']
    feature_values = [mitosis_path.split('/')[-1].strip('.npy')]
    for feature_func in feature_funcs:
        feature_dict = feature_func(mitosis_candidates, wsi_path, graph_path)
        feature_names.extend(list(feature_dict.keys()))
        feature_values.extend(list(feature_dict.values()))
        
    # create output dictionary
    out_dict = dict(zip(feature_names, feature_values))

    if temp_save_path is not None:
        random_name = generate_random_name()
        with open(f'{temp_save_path}/{random_name}.pickle', 'wb') as file:
            pickle.dump(out_dict, file)
    
    return None# out_dict
    
def feature_extractor(mitosis_path, tumour_path, feature_funcs):
    '''Feature extraction from a single slide mitosis and tumour mask predictions.
    
    This functin is to be used in multiprocessing and is designed to work in extendable
    fashion where feature extraction function is provided as an argument. This design
    help adding new feature to the previouisely extracted features.
    feature_funcs:
        should be a list of functions that accept mitosis candidates (and tumour mask)
        and returns a feacture dictionary with keys being feature names and their
        values as dictinary values.
    tumour_path:
        Whether to filter mitosis candidates based on tumour mask if `tumour_path`
        is a valid path, otherwise, it should be set to `None`.
    '''
    # reading inputs
    # mitosis_candidates = pd.read_csv(mitosis_path)
    temp = np.load(mitosis_path)
    mitosis_candidates = pd.DataFrame(temp, columns=['x','y','score'])
    if tumour_path is not None:
        tumour_mask = cv2.imread(tumour_path, 0)
        tumour_mask = np.uint8(tumour_mask==255)
    else:
        tumour_mask = None
        
    # filter mitosis candidates
    mitosis_candidates = filter_candidates(mitosis_candidates, tumour_mask, candidate_thresh=0.5, coord_ratio=1/MASK_RATIO, filter_by_mask=False)
    
    # extract features
    feature_names = ['slide']
    feature_values = [mitosis_path.split('/')[-1].strip('.npy')]
    for feature_func in feature_funcs:
        feature_dict = feature_func(mitosis_candidates, tumour_mask)
        feature_names.extend(list(feature_dict.keys()))
        feature_values.extend(list(feature_dict.values()))
        
    # create output dictionary
    out_dict = dict(zip(feature_names, feature_values))
    
    return out_dict

def flat_mesh_grid_coord(x, y):
    """Helper function to obtain coordinate grid."""
    x, y = np.meshgrid(x, y)
    return np.stack([x.flatten(), y.flatten()], axis=-1)

def count_in_bbox(bound, cen_list):
    ''' Returns the number of points in `cen_list` that falls in `bounds`.
    Bounds should be in [x1,y1,x2,y2] format.'''
    
    sel = np.ones(cen_list.shape[0], dtype=bool)
    sel &= cen_list[:, 0]>bound[0]
    sel &= cen_list[:, 0]<bound[2]
    sel &= cen_list[:, 1]>bound[1]
    sel &= cen_list[:, 1]<bound[3]
    
    count = np.sum(sel)
    
    return count