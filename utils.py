import matplotlib.pyplot as plt
from itertools import cycle

def get_colors_dict():
    # domain_list = ['BLCA', 'BRCA', 'CESC', 'COADREAD', 'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'PAAD', 'SKCM', 'STAD', 'UCEC', 'PCPG']# + ['MESO', 'UVM', 'TGCT', 'THYM', 'THCA', 'LAML', 'DLBC', 'UCS', 'SARC', 'CHOL', 'PRAD', 'ACC']
    domain_list = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COADREAD', 'DLBC', 'ESCA', 'GBMLGG', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS']
    # Use Set3 color palette from Matplotlib
    set3_palette = list(plt.cm.tab20.colors) + [plt.cm.tab20b.colors[i] for i in [0, 2, 4, 5, 8, 9, 13, 16]] + [plt.cm.tab20c.colors[i] for i in [4, 16]] # plt.cm.tab20.colors + plt.cm.tab20b.colors
    
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
    if feat_parts[0]=="aty": # "aty_wsi_ratio", "aty_hotspot_ratio", "aty_hotspot_count"
        if feat_parts[1]=='wsi' and feat_parts[2]=='ratio' :
            return "WAF"
        elif feat_parts[1]=='wsi' and feat_parts[2]=='count' :
            return "WAC"
        elif feat_parts[1]=='hotspot' and feat_parts[2]=='ratio' :
            return "HAF"
        elif feat_parts[1]=='hotspot' and feat_parts[2]=='count' :
            return "HAC"
    if feat_parts[1]=='wsi':
        feat_out = 'WSC'
    elif feat_parts[1]=='hotspot' and feat_parts[2]=='count':
        feat_out = 'HSC'   
    elif feat_parts[1]=='assortCoeff':
        feat_out = 'Assort'
    else:
        func_name = feat_parts[-1]
        func_name = func_name.replace("perc", "per")
        feat_out = f'{func_name}({coversion_dict[feat_parts[1]]})'
    return feat_out