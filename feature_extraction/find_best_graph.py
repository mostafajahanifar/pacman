import json
import networkx as nx
import numpy as np
from scipy.spatial import distance

# Load the data from the file
slide_name = 'TCGA-E9-A1R5-01Z-00-DX1.b870d5eb-6259-4b49-86e4-569a9e32ecd5'
data = np.load(f'/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/mitosis_cls_candidates_final(4classes)/breast/{slide_name}.npy')
data = data[data[:,2]>0.5, :]
print("num nodes: ", len(data))

# Compute the distance matrix
dist_matrix = distance.cdist(data[:, :2], data[:, :2], 'euclidean')

# Set your threshold here
mpp = 0.25
for threshold in [250, 300, 400, 500]:
    # threshold = 250 # in um
    threshold_at_mpp = threshold / mpp

    # Form the edge index based on the threshold
    edge_index = np.argwhere(dist_matrix < threshold_at_mpp)

    # Prepare the data for saving
    graph_dict = {
        'edge_index': edge_index.T.tolist(),
        'coordinates': data[:, :2].tolist(),
        'feats': data[:, 1:].tolist(),
        'feat_names': ['score', 'mosi']
    }

    # Save the graph to a JSON file
    with open(f'overlays/{slide_name}_{threshold}.json', 'w') as f:
        json.dump(graph_dict, f)