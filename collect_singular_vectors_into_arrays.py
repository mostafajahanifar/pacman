import os
import numpy as np
from multiprocessing import Pool

source_dir = "/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/MSI_TUMOUR_PATCHES_RES0.5_SIZE512_STRIDE512_Embeddings/"
target_dir = "/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/msi-pancancer/MSI_TUMOUR_PATCHES_RES0.5_SIZE512_STRIDE512_Embeddings_PHIKON/"

def process_case(case_path):
    embeddings = []
    paths = []
    
    for file in os.listdir(case_path):
        if file.endswith(".png.npy"):
            file_path = os.path.join(case_path, file)
            embedding = np.load(file_path)
            embeddings.append(embedding)
            paths.append(file_path[:-4])
    
    embeddings = np.array(embeddings)
    paths = np.array(paths)
    
    target_case_path = case_path.replace(source_dir, target_dir)
    os.makedirs(target_case_path, exist_ok=True)
    
    np.save(os.path.join(target_case_path, "embeddings.npy"), embeddings)
    np.save(os.path.join(target_case_path, "paths.npy"), paths)
    print(f"processed case: {case_path}")

if __name__ == "__main__":
    case_paths = []
    
    for domain in os.listdir(source_dir):
        domain_path = os.path.join(source_dir, domain)
        for case in os.listdir(domain_path):
            case_path = os.path.join(domain_path, case)
            case_paths.append(case_path)
    
    with Pool(processes=8) as p:
        p.map(process_case, case_paths)
