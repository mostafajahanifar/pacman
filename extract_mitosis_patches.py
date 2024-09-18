import os, glob
import numpy as np
import pandas as pd
from tiatoolbox.tools import patchextraction
from tiatoolbox.wsicore.wsireader import OpenSlideWSIReader, WSIMeta
import cv2
from multiprocessing import Pool, freeze_support, RLock
from tqdm import tqdm
import argparse

def find_slide_path(slide_name, slide_paths):
    for path in slide_paths:
        if slide_name in path:
            return path
    return None

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

def extract_patches(wsi_path, npy_path):
    wsi_name = wsi_path.split("/")[-1].strip(".svs")
    save_path = os.path.join(save_root, wsi_name) + "/"
    os.makedirs(save_path, exist_ok=True) 

    
    # reading the wsi and correcting its metadata
    wsi = OpenSlideWSIReader.open(wsi_path)
    info_dict = wsi.info.as_dict()
    wsi_check = checking_wsi_info(info_dict)
    if wsi_check == -1: # if it is a bad mask, go to next one
        print(f'This is a bad WSI (not enough levels): {wsi_path}')
        return -1
    wsi_mag, wsi_mpp = wsi_check
    wsi_meta = WSIMeta(
            file_path=wsi_path,
            axes="YSX",
            objective_power=wsi_mag,
            slide_dimensions=info_dict['slide_dimensions'],
            level_count=info_dict['level_count'],
            level_dimensions=info_dict['level_dimensions'],
            level_downsamples=info_dict['level_downsamples'],
            vendor='aperio',
            mpp=(wsi_mpp,wsi_mpp),
            raw=None,
        )
    wsi._m_info = wsi_meta
    use_mag = 20 if np.abs(wsi_mpp-0.5) < np.abs(wsi_mpp-0.25) else 40
    patch_size = patch_size_dict[use_mag]


    points = np.load(npy_path, allow_pickle=True)
    points = points[points[:,2]>0.5]
    # Define the patch_extractor
    patch_extractor = patchextraction.get_patch_extractor(
                input_img=wsi, 
                method_name="point",
                patch_size=patch_size,
                locations_list=points[:, :2],
                resolution=0,
                units="level",
                pad_mode="constant",
                pad_constant_values=0,
            )
    np.save(save_path+"point_list.npy", points)
    for i, patch in enumerate(patch_extractor):
        patch_name = f"{i}_{int(points[i, 0])}_{int(points[i, 1])}.png"
        cv2.imwrite(save_path+patch_name, patch[...,::-1])


parser = argparse.ArgumentParser(description='Gene to mitosis features analysis')
parser.add_argument('--cancer_types', nargs='+', required=True)
args = parser.parse_args()
cancer_type = ''.join(args.cancer_types).upper() 
patch_size = 64
num_processes = 8
patch_size_dict = {20: 32, 40: 64}

df = pd.read_csv('/home/u2070124/lsf_workspace/Data/Data/pancancer/tcga_features_clinical_merged.csv')

invalid_cancers = ['MESO', 'UVM', 'TGCT', 'THYM', 'THCA', 'LAML', 'DLBC', 'UCS', 'SARC', 'CHOL', 'PRAD', 'ACC']  # with kept PCPG
df = df[~df['type'].isin(invalid_cancers)]
df['type'] = df['type'].replace('COAD', 'COADREAD')
df['type'] = df['type'].replace('READ', 'COADREAD')

df = df[df['type']==cancer_type]

root_mitosis_path = '/home/u2070124/lsf_workspace/Data/Data/pancancer/mitosis_cls_candidates_MIDOGnTUPACnBRACE/' # "/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/mitosis_cls_candidates_MIDOGnTUPACnBRACE/" # 
root_wsi_path = '/home/u2070124/web-public/tcga/' #"/mnt/web-public/tcga/" #  
root_save_path = '/home/u2070124/lsf_workspace/Data/Data/pancancer/mitosis_patches/' #'/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/mitosis_patches/' # 

all_npy_files = glob.glob(root_mitosis_path+"**/*.npy", recursive=True)

npy_paths = [find_slide_path(slide_name, all_npy_files) for slide_name in df['slide']]
wsi_paths = [root_wsi_path+slide_name+".svs" for slide_name in df['slide']]

save_root = root_save_path + cancer_type

# for wsi_path, npy_path in zip(wsi_paths[120:121], npy_paths[120:121]):
#     extract_patches(wsi_path, npy_path)

# prepare the multi_processing
freeze_support() # For Windows support
num_jobs = len(wsi_paths)
pool = Pool(processes=num_processes, initargs=(RLock(),), initializer=tqdm.set_lock)

# initialize the progress bar
pbar = tqdm(total=num_jobs, ascii=True, leave=True, desc=cancer_type)
def update_pbar(*x):
    pbar.update()

# do multiprocessing
jobs = [pool.apply_async(extract_patches, args=(i,n), callback=update_pbar) for i, n in zip(wsi_paths, npy_paths)]
pool.close()
result_list = [job.get() for job in jobs]
pool.join()
pbar.close()
