import matplotlib.pyplot as plt
from itertools import cycle

def get_colors_dict():
    domain_list = ['BLCA', 'BRCA', 'CESC', 'COADREAD', 'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'PAAD', 'SKCM', 'STAD', 'UCEC', 'PCPG']# + ['MESO', 'UVM', 'TGCT', 'THYM', 'THCA', 'LAML', 'DLBC', 'UCS', 'SARC', 'CHOL', 'PRAD', 'ACC']
    # Use Set3 color palette from Matplotlib
    set3_palette = plt.cm.tab20.colors + plt.cm.tab20b.colors
    
    # Create a cycle iterator for the colors
    color_cycle = cycle(set3_palette)

    # Generate custom color dictionary
    custom_colors = {domain: next(color_cycle) for domain in domain_list}

    return custom_colors

def featre_to_tick(feat):
    coversion_dict = {'cenEigen': 'EC',
                      'wsi_count': 'WC',
                      'clusterCoff': 'CL',
                      'cenDegree': 'DC',
                      'cenCloseness': 'CC',
                      'nodeDegrees': 'ND',
                      'cenHarmonic': 'HC',
                      'hotspot_count': 'HS',
                      }
    feat_parts = feat.split('_')
    if feat_parts[1]=='wsi':
        feat_out = 'WSC'
    elif feat_parts[1]=='hotspot':
        feat_out = 'HSC'   
    else:
        feat_out = f'{feat_parts[-1]}({coversion_dict[feat_parts[1]]})'
    return feat_out