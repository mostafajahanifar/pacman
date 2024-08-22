import numpy as np
# from torch_geometric.nn import radius_graph
# import torch
# from torch_geometric.data import Data
# from torch_geometric.utils import to_networkx
import networkx as nx
from networkx.algorithms import centrality, assortativity
from feature_utils import count_in_bbox, flat_mesh_grid_coord, checking_wsi_info
import math
from scipy.spatial import distance
from tiatoolbox.wsicore.wsireader import WSIReader, OpenSlideWSIReader
import pickle

MASK_RATIO = 16
MAX_NEIGHBORS = 20
EDGE_RADIUS = 500 # in um
HOTSPOT_WINDOW = 3 # in mm

BINS_DICT = {
    'nodeDegrees': [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 25.0],
    'clusterCoff': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'cenDegree': [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.2, 0.3, 0.4, 0.5],
    'cenCloseness': [0.0, 0.002, 0.005, 0.010, 0.015, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3],
    'cenEigen': [0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
    # 'cenKatz': [0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
    'cenHarmonic': [0, 0.5, 1, 5, 10, 20, 25.0, 50.0, 75.0, 100.0, 130.0],
    # 'cenBetweenness': [0, 0.5, 1, 5, 10, 20, 25.0, 50.0, 75.0, 100.0, 130.0],
            }


def create_network(mitosis_candidates, radius_threshold):
    """ Get proximity network by finding close neighbors of each node.
    
    args:
        mitosis_candidates: a numpy array of shape Nx2 containing mitosis points in x,y format
        radius_threshold: radius of mitosis proximity in WSI baseline resolution
    """
    coordinates = mitosis_candidates[['x', 'y']].to_numpy().astype(np.float32)
    scores = mitosis_candidates['score'].to_numpy()  # assuming 'score' is the column name
    dist_matrix = distance.cdist(coordinates, coordinates, 'euclidean')
    edge_index = np.argwhere(dist_matrix < radius_threshold)

    # Create a new graph
    G = nx.Graph()

    # Add nodes to the graph with coordinate and score attributes
    nodes = [(i, {'pos': coordinates[i].tolist(), 'score': scores[i]}) for i in range(coordinates.shape[0])]
    G.add_nodes_from(nodes)

    # Add edges to the graph
    edges = list(map(tuple, edge_index))
    G.add_edges_from(edges)

    return G

def create_network_knn (mitosis_candidates, nodes_features, radius_threshold):
    nodes_features = mitosis_candidates[['score']].to_numpy().astype(np.float32)
    coordinates = torch.from_numpy(coordinates).to(torch.float)
    nodes_features = torch.from_numpy(nodes_features).to(torch.float)
    y = torch.tensor([1], dtype=torch.long) # define a dummy label

    # Create the network
    net_data = Data(x=nodes_features, pos=coordinates, y=y)
    edge_index = radius_graph(net_data.pos, radius_threshold, None, False, MAX_NEIGHBORS)
    net_data.edge_index=edge_index

    # Convert the graph to NetworkX network
    G = to_networkx (net_data, to_undirected=True)

    return G

def sna_features_concise(mitosis_candidates, wsi_path, graph_path=None):
    if isinstance(wsi_path, str):
        wsi = OpenSlideWSIReader.open(wsi_path)
        info_dict = wsi.info.as_dict()
        wsi_check = checking_wsi_info(info_dict)
    elif isinstance(wsi_path, dict):
        wsi_check = 40, wsi_path['mpp']
    else:
        wsi_check = -1
    
    # check if there is no node, return all zeros features
    if wsi_check == -1:
        feature_dict = dict()

        # pushing features of the node-based measures
        for sna in BINS_DICT.keys():
            # statistical features
            feature_dict[f'mit_{sna}_mean'] = -1
            feature_dict[f'mit_{sna}_max'] = -1
            feature_dict[f'mit_{sna}_min'] = -1
            feature_dict[f'mit_{sna}_med'] = -1
            feature_dict[f'mit_{sna}_std'] = -1
            feature_dict[f'mit_{sna}_perc10'] = -1
            feature_dict[f'mit_{sna}_perc20'] = -1
            feature_dict[f'mit_{sna}_perc80'] = -1
            feature_dict[f'mit_{sna}_perc90'] = -1
        return feature_dict

    # check if there is no node, return all zeros features
    if mitosis_candidates.size == 0:
        feature_dict = dict()

        # pushing features of the node-based measures
        for sna in BINS_DICT.keys():
            # statistical features
            feature_dict[f'mit_{sna}_mean'] = 0
            feature_dict[f'mit_{sna}_max'] = 0
            feature_dict[f'mit_{sna}_min'] = 0
            feature_dict[f'mit_{sna}_med'] = 0
            feature_dict[f'mit_{sna}_std'] = 0
            feature_dict[f'mit_{sna}_perc10'] = 0
            feature_dict[f'mit_{sna}_perc20'] = 0
            feature_dict[f'mit_{sna}_perc80'] = 0
            feature_dict[f'mit_{sna}_perc90'] = 0
        return feature_dict
    
    _, wsi_mpp = wsi_check
    radius_threshold = EDGE_RADIUS / wsi_mpp
    
    save_graph=False
    if graph_path is not None:
        try: # try reading graph
            with open(graph_path, 'rb') as f:
                G = pickle.load(f)
        except:
            save_graph=True
            G = create_network(mitosis_candidates=mitosis_candidates, radius_threshold=radius_threshold)
    else:
        G = create_network(mitosis_candidates=mitosis_candidates, radius_threshold=radius_threshold)

    if save_graph:
        with open(graph_path, 'wb') as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)

    # Extract SNA measures
    sna_measures = dict()
    # Node degree
    sna_measures['nodeDegrees'] = [d for n, d in G.degree()]
    # Clustering Coefficient
    clusterCoff = nx.clustering(G)
    sna_measures['clusterCoff'] = list(clusterCoff.values())
    # Degree Centrality
    cenDegree = centrality.degree_centrality(G)
    sna_measures['cenDegree'] = list(cenDegree.values())
    # Closeness Centrality
    cenCloseness = centrality.closeness_centrality(G)
    sna_measures['cenCloseness'] = list(cenCloseness.values())
    # Eigenvector Centrality
    try:
        cenEigen = centrality.eigenvector_centrality(G, max_iter=2000, tol=1e-02, nstart=None, weight=None)
    except:
        cenEigen = centrality.eigenvector_centrality(G, max_iter=3000, tol=1e-01, nstart=None, weight=None)
    sna_measures['cenEigen'] = list(cenEigen.values())
    
    # Harmonic Centrality
    cenHarmonic = centrality.harmonic_centrality(G)
    sna_measures['cenHarmonic'] = list(cenHarmonic.values())

    # Extracting features and creating feature directory
    feature_dict = dict()

    # pushing features of the node-based measures
    for sna in BINS_DICT.keys():
        # statistical features
        feature_dict[f'mit_{sna}_mean'] = np.mean(sna_measures[sna])
        feature_dict[f'mit_{sna}_max'] = np.max(sna_measures[sna])
        feature_dict[f'mit_{sna}_min'] = np.min(sna_measures[sna])
        feature_dict[f'mit_{sna}_med'] = np.median(sna_measures[sna])
        feature_dict[f'mit_{sna}_std'] = np.std(sna_measures[sna])
        feature_dict[f'mit_{sna}_perc10'] = np.percentile(sna_measures[sna], 10)
        feature_dict[f'mit_{sna}_perc20'] = np.percentile(sna_measures[sna], 20)
        feature_dict[f'mit_{sna}_perc80'] = np.percentile(sna_measures[sna], 80)
        feature_dict[f'mit_{sna}_perc90'] = np.percentile(sna_measures[sna], 90)
    return feature_dict


def sna_features(mitosis_candidates, tumour_mask, radius_threshold=500, graph_path=None):
    # extract information from DF
    coordinates = mitosis_candidates[['x', 'y']].to_numpy().astype(np.float32)
    

    # check if there is no node, return all zeros features
    if coordinates.size(0)==0:
        feature_dict = dict()
        feature_dict['mit_assortCoeff'] = 0
        # # pushing the Average Degree Connectivity
        # for i in range(10):
        #     feature_dict[f'mit_avrDegree_{i+1}'] = 0

        # pushing features of the node-based measures
        for sna in BINS_DICT.keys():
            # statistical features
            feature_dict[f'mit_{sna}_mean'] = 0
            feature_dict[f'mit_{sna}_max'] = 0
            feature_dict[f'mit_{sna}_min'] = 0
            feature_dict[f'mit_{sna}_med'] = 0
            feature_dict[f'mit_{sna}_std'] = 0
            # histogram values
            for i in range(len(BINS_DICT[sna])-1):
                feature_dict[f'mit_{sna}_h{i+1}'] = 0
        return feature_dict
    
    save_graph=False
    if graph_path is not None:
        try: # try reading graph
            with open(graph_path, 'rb') as f:
                G = pickle.load(f)
        except:
            save_graph=True
    G = create_network(mitosis_candidates=coordinates, radius_threshold=radius_threshold)

    if save_graph:
        with open(graph_path, 'wb') as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)

    # Extract SNA measures
    sna_measures = dict()
    # Node degree
    sna_measures['nodeDegrees'] = [d for n, d in G.degree()]
    # Clustering Coefficient
    clusterCoff = nx.clustering(G)
    sna_measures['clusterCoff'] = list(clusterCoff.values())
    # Degree Centrality
    cenDegree = centrality.degree_centrality(G)
    sna_measures['cenDegree'] = list(cenDegree.values())
    # Betweenness Centrality
    cenBetweenness = centrality.betweenness_centrality(G)
    cenBetweenness = list(cenBetweenness.values())
    # Closeness Centrality
    cenCloseness = centrality.closeness_centrality(G)
    sna_measures['cenCloseness'] = list(cenCloseness.values())
    # Eigenvector Centrality
    try:
        cenEigen = centrality.eigenvector_centrality(G, max_iter=2000, tol=1e-02, nstart=None, weight=None)
    except:
        cenEigen = centrality.eigenvector_centrality(G, max_iter=3000, tol=1e-01, nstart=None, weight=None)
    sna_measures['cenEigen'] = list(cenEigen.values())
    # Katz Centrality
    try:
        cenKatz = centrality.katz_centrality(G, alpha=0.05, max_iter=4000, tol=1e-06)
    except:
        cenKatz = centrality.katz_centrality(G, alpha=0.02, max_iter=4000, tol=1e-05)
    sna_measures['cenKatz'] = list(cenKatz.values())
    # Harmonic Centrality
    cenHarmonic = centrality.harmonic_centrality(G)
    sna_measures['cenHarmonic'] = list(cenHarmonic.values())
    # # Average Degree Connectivity: always have 10 keys because we have maximum node degree of 10
    # avrDegree = assortativity.average_degree_connectivity(G)
    # avrDegree = list(avrDegree.values()) # this should be a list of length 10, otherwise, make it like that
    # if len(avrDegree) > 10:
    #     avrDegree = avrDegree[:10]
    # if len(avrDegree)<10:
    #     avrDegree += [0] * (10 - len(avrDegree))
    # sna_measures['avrDegree'] = avrDegree

    # Degree Assortativity Coefficient: single number for graph
    sna_measures['assortCoeff'] = assortativity.degree_assortativity_coefficient(G)
    if math.isnan(sna_measures['assortCoeff']): # check if Assortativity is nan
        sna_measures['assortCoeff'] = 0

    # Extracting features and creating feature directory
    feature_dict = dict()

    feature_dict['mit_assortCoeff'] = sna_measures['assortCoeff']
    # # pushing the Average Degree Connectivity
    # for i, ad in enumerate(sna_measures['avrDegree']):
    #     feature_dict[f'mit_avrDegree_{i+1}'] = ad

    # pushing features of the node-based measures
    for sna in BINS_DICT.keys():
        # statistical features
        feature_dict[f'mit_{sna}_mean'] = np.mean(sna_measures[sna])
        feature_dict[f'mit_{sna}_max'] = np.max(sna_measures[sna])
        feature_dict[f'mit_{sna}_min'] = np.min(sna_measures[sna])
        feature_dict[f'mit_{sna}_med'] = np.median(sna_measures[sna])
        feature_dict[f'mit_{sna}_std'] = np.std(sna_measures[sna])
        # histogram values
        hist, _ = np.histogram(sna_measures[sna], bins=BINS_DICT[sna])
        for i, f in enumerate(hist):
            feature_dict[f'mit_{sna}_h{i+1}'] = f
    return feature_dict

def density_features(mitosis_candidates, tumour_mask):
    feature_dict = dict()
    
    total_mitosis = mitosis_candidates.shape[0]
    total_tumour_area = np.sum(tumour_mask>0) * MASK_RATIO * 0.25 * 1e-6 # in mm^2
    
    # number of mitosis
    feature_dict['mit_num'] = total_mitosis   
    # tumour area
    feature_dict['tum_area'] = total_tumour_area 
    # mitosis to tumour ration
    feature_dict['mit_tum_ratio'] = total_mitosis/(total_tumour_area+np.finfo(np.float32).eps)
    
    return feature_dict

def mitosis_hotspot(mitosis_candidates, wsi_path, graph_path=None):
    if isinstance(wsi_path, str):
        wsi = OpenSlideWSIReader.open(wsi_path)
        info_dict = wsi.info.as_dict()
        wsi_check = checking_wsi_info(info_dict)
    elif isinstance(wsi_path, dict):
        wsi_check = 40, wsi_path['mpp']
        # set a default slide size 
        info_dict = {'slide_dimensions': (250000, 150000)}
    else:
        wsi_check = -1

    if wsi_check == -1:
        # writing default features to feature dictionary
        feature_dict = dict()
        feature_dict['mit_wsi_count'] = -1
        feature_dict['mit_hotspot_count'] = -1
        feature_dict['mit_hotspot_score'] = -1
        feature_dict['mit_hotspot_x1'] = -1
        feature_dict['mit_hotspot_y1'] = -1
        feature_dict['mit_hotspot_x2'] = -1
        feature_dict['mit_hotspot_y2'] = -1

        return feature_dict

    _, wsi_mpp = wsi_check
    # calculating the windows size in pixels and baseline resolution
    bs = int(np.round(1000*np.sqrt(HOTSPOT_WINDOW)/wsi_mpp)) # bound size in pixels
    stride = bs//6

    img_h = info_dict['slide_dimensions'][1]
    img_w = info_dict['slide_dimensions'][0]
    cen_list = mitosis_candidates[['x', 'y']].to_numpy().astype(np.float32)

    if cen_list.shape[0] == 0: # no mitosis
        # writing default features to feature dictionary
        feature_dict = dict()
        feature_dict['mit_wsi_count'] = 0
        feature_dict['mit_hotspot_count'] = 0
        feature_dict['mit_hotspot_score'] = 0
        feature_dict['mit_hotspot_x1'] = 0
        feature_dict['mit_hotspot_y1'] = 0
        feature_dict['mit_hotspot_x2'] = 0
        feature_dict['mit_hotspot_y2'] = 0

        return feature_dict

    # find the bounds to check for mitosis count
    output_x_list = np.arange(0, int(img_w), stride)
    output_y_list = np.arange(0, int(img_h), stride)
    output_tl_list = flat_mesh_grid_coord(output_x_list, output_y_list)
    output_br_list = output_tl_list + bs

    bounds = np.concatenate([output_tl_list, output_br_list], axis=1)

    # finding count in each bound and put in canvas
    counts = []
    for bound in bounds:
        b_count = count_in_bbox(bound, cen_list)
        counts.append(b_count)

    i_max = np.argmax(counts)
    mitosis_count = int(counts[i_max])
    cb = list(bounds[i_max])

    if mitosis_count <= 23:
        mitosis_score = 1
    elif mitosis_count >= 51:
        mitosis_score = 3
    else:
        mitosis_score = 2

    # writing features to feature dictionary
    feature_dict = dict()
    feature_dict['mit_wsi_count'] = cen_list.shape[0]
    feature_dict['mit_hotspot_count'] = mitosis_count
    feature_dict['mit_hotspot_score'] = mitosis_score
    feature_dict['mit_hotspot_x1'] = cb[0]
    feature_dict['mit_hotspot_y1'] = cb[1]
    feature_dict['mit_hotspot_x2'] = cb[2]
    feature_dict['mit_hotspot_y2'] = cb[3]

    return feature_dict