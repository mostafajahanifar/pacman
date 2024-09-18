import numpy as np
import math
import networkx as nx
from networkx.algorithms import centrality, assortativity
from feature_utils import count_in_bbox, flat_mesh_grid_coord, checking_wsi_info
from scipy.spatial import distance
from tiatoolbox.wsicore.wsireader import OpenSlideWSIReader
import json

EDGE_RADIUS = 400 # in um
HOTSPOT_WINDOW = 3 # in mm

BINS_DICT = {
    'nodeDegrees': [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 25.0],
    'clusterCoff': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'cenHarmonic': [0, 0.5, 1, 5, 10, 20, 25.0, 50.0, 75.0, 100.0, 130.0],
    'cenEigen': [0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
            }


def create_network(mitosis_candidates, radius_threshold):
    """ Get proximity network by finding close neighbors of each node.
    
    args:
        mitosis_candidates: a numpy array of shape Nx2 containing mitosis points in x,y format
        radius_threshold: radius of mitosis proximity in WSI baseline resolution
    """
    coordinates = mitosis_candidates[['x', 'y']].to_numpy().astype(np.float32)
    types = mitosis_candidates['type'].to_numpy()  
    dist_matrix = distance.cdist(coordinates, coordinates, 'euclidean')
    edge_index = np.argwhere(dist_matrix < radius_threshold)

    # Create a new graph
    G = nx.Graph()

    # Add nodes to the graph with coordinate and score attributes
    nodes = [(i, {'pos': coordinates[i].tolist(), 'type': types[i]}) for i in range(coordinates.shape[0])]
    G.add_nodes_from(nodes)

    # Add edges to the graph
    edges = list(map(tuple, edge_index))
    G.add_edges_from(edges)

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
            feature_dict[f'mit_{sna}_cv'] = -1
            feature_dict[f'mit_{sna}_perc1'] = -1
            feature_dict[f'mit_{sna}_perc10'] = -1
            feature_dict[f'mit_{sna}_perc90'] = -1
            feature_dict[f'mit_{sna}_perc99'] = -1
        feature_dict[f'mit_assortCoeff'] = -1
        return feature_dict

    # check if there is no node, return all zeros features
    if len(mitosis_candidates) <= 1:
        feature_dict = dict()

        # pushing features of the node-based measures
        for sna in BINS_DICT.keys():
            # statistical features
            feature_dict[f'mit_{sna}_mean'] = 0
            feature_dict[f'mit_{sna}_max'] = 0
            feature_dict[f'mit_{sna}_min'] = 0
            feature_dict[f'mit_{sna}_med'] = 0
            feature_dict[f'mit_{sna}_std'] = 0
            feature_dict[f'mit_{sna}_cv'] = 0
            feature_dict[f'mit_{sna}_perc1'] = 0
            feature_dict[f'mit_{sna}_perc10'] = 0
            feature_dict[f'mit_{sna}_perc90'] = 0
            feature_dict[f'mit_{sna}_perc99'] = 0
        feature_dict[f'mit_assortCoeff'] = 1 # similar to the scenario where everything is connected similarly
        return feature_dict
    
    _, wsi_mpp = wsi_check
    radius_threshold = EDGE_RADIUS / wsi_mpp
    
    # create the radius graph
    G = create_network(mitosis_candidates=mitosis_candidates, radius_threshold=radius_threshold)


    # Extract SNA measures
    sna_measures = dict()
    # Node degree
    sna_measures['nodeDegrees'] = [d for n, d in G.degree()]
    # Clustering Coefficient
    clusterCoff = nx.clustering(G)
    sna_measures['clusterCoff'] = list(clusterCoff.values())

    # "Normalized" Harmonic Centrality
    cenHarmonic = centrality.harmonic_centrality(G)
    norm_factor = 1/(G.number_of_nodes()-1)
    sna_measures['cenHarmonic'] = [hc*norm_factor for hc in cenHarmonic.values()]

    # Eigenvector Centrality
    try:
        cenEigen = centrality.eigenvector_centrality(G, max_iter=2000, tol=1e-02, nstart=None, weight=None)
    except:
        cenEigen = centrality.eigenvector_centrality(G, max_iter=3000, tol=1e-01, nstart=None, weight=None)
    sna_measures['cenEigen'] = list(cenEigen.values())
    

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
        feature_dict[f'mit_{sna}_cv'] = np.std(sna_measures[sna])/np.mean(sna_measures[sna])
        feature_dict[f'mit_{sna}_perc1'] = np.percentile(sna_measures[sna], 1)
        feature_dict[f'mit_{sna}_perc10'] = np.percentile(sna_measures[sna], 10)
        feature_dict[f'mit_{sna}_perc90'] = np.percentile(sna_measures[sna], 90)
        feature_dict[f'mit_{sna}_perc99'] = np.percentile(sna_measures[sna], 99)
    mit_assort = assortativity.degree_pearson_correlation_coefficient(G)
    if math.isnan(mit_assort): # check if Assortativity is nan
        mit_assort = 1 # usally happens when no connection is there, therefore considered 1 (like when there is only 1 connection between every pair)
    feature_dict[f'mit_assortCoeff'] = mit_assort

    if graph_path is not None:
        # create the graph in geojson format
        # Prepare the data for saving
        nodes_type = np.array([data['type'] for _, data in G.nodes(data=True)])[:,np.newaxis]
        sna_measures_array = np.array(list(sna_measures.values())).T
        node_features = np.concatenate([nodes_type, sna_measures_array], axis=1).tolist()
        graph_dict = {
            'edge_index': np.array(G.edges()).T.tolist(),
            'coordinates': [data['pos'] for _, data in G.nodes(data=True)],
            'feats': node_features,
            'feat_names': ['type', 'Node_Degree', 'Clustering_Coeff', 'Harmonic_Cen', 'Eigenvector_cen'],
        }
        with open(graph_path, 'w') as f:
            json.dump(graph_dict, f)

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