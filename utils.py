import matplotlib.pyplot as plt
from itertools import cycle

def get_colors_dict():
    domain_list = ['BLCA', 'BRCA', 'CESC', 'COADREAD', 'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'PAAD', 'SKCM', 'STAD', 'UCEC']
    # Use Set3 color palette from Matplotlib
    set3_palette = plt.cm.tab20.colors
    
    # Create a cycle iterator for the colors
    color_cycle = cycle(set3_palette)

    # Generate custom color dictionary
    custom_colors = {domain: next(color_cycle) for domain in domain_list}

    return custom_colors