from ntpath import join
import os, glob
import pandas as pd
import cv2
import numpy as np
from feature_bank import sna_features_concise, mitosis_hotspot
from feature_utils import feature_extractor, feature_extractor_tcga
from multiprocessing import Pool, freeze_support, RLock
from tqdm import tqdm
import argparse
import pickle
import shutil

parser = argparse.ArgumentParser(description='Select the cohort to be processed')
parser.add_argument('-c', '--cohort') 
args = parser.parse_args()
s = args.cohort

root_mitosis_path = '/home/u2070124/lsf_workspace/Data/Data/pancancer/mitosis_cls_candidates_final(4classes)/'
root_wsi_path = '/home/u2070124/web-public/tcga/'
old_root_save_path = None# '/root/lsf_workspace/Data/Data/Brace_WSI_mitosis/features/' # this points to the path that previous feature extractions were done. In order to append new features to them
root_save_path = '/home/u2070124/lsf_workspace/Data/Data/pancancer/features_final/' # this points to the new place to store feaures in. Can be the same as `old_root_save_path`
root_graph_path = f'/home/u2070124/lsf_workspace/Data/Data/pancancer/graphs_final/' 
os.makedirs(root_save_path, exist_ok=True)
os.makedirs(root_graph_path, exist_ok=True)

temp_save_path = f'/home/u2070124/lsf_workspace/Data/Data/pancancer/features_final/{s}/'
os.makedirs(temp_save_path, exist_ok=True) 


set_path_mitosis = os.path.join(root_mitosis_path, s+'/')
csv_paths = glob.glob(set_path_mitosis+'*.npy')
wsi_paths = [os.path.join(root_wsi_path,case.split('/')[-1].strip('.npy')+'.svs') for case in csv_paths]
graph_paths = [None for case in csv_paths] # [os.path.join(root_graph_path,case.split('/')[-1].strip('.npy')+'.json') for case in csv_paths]

print(f'Working on set: {s} -- Total number of cases: {len(csv_paths)}')

# prepare the multi_processing
num_processes = 32
freeze_support() # For Windows support
num_jobs = len(csv_paths)
pool = Pool(processes=num_processes, initargs=(RLock(),), initializer=tqdm.set_lock)

# initialize the progress bar
pbar = tqdm(total=num_jobs, ascii=True, leave=True, desc=s)
def update_pbar(*x):
    pbar.update()

# do multiprocessing
jobs = [pool.apply_async(feature_extractor_tcga, args=(i,n,g,[mitosis_hotspot,sna_features_concise], temp_save_path), callback=update_pbar) for i, n, g in zip(csv_paths, wsi_paths, graph_paths)]
pool.close()
result_list = [job.get() for job in jobs]
pool.join()
pbar.close()

result_list = []
for temp_path in os.listdir(temp_save_path):
    with open(temp_save_path+temp_path, 'rb') as file:
        loaded_dict = pickle.load(file)
        result_list.append(loaded_dict)
shutil.rmtree(temp_save_path)

# Create the new df
new_df = pd.DataFrame(result_list)
# merge wit old DF if exists
if old_root_save_path is not None: # read the old dfs and add new results to them
    old_df = pd.read_csv(os.path.join(old_root_save_path, s+'_header.csv'))
    new_df = pd.merge(old_df, new_df, on="case", how='outer')
# Save the new (updated) DF to disk
save_df_path = os.path.join(root_save_path, s)
new_df.to_csv(save_df_path+'.csv', sep=',', index=False, header=False) # save headerless
new_df.to_csv(save_df_path+'_header.csv', sep=',', index=False) # save with headers