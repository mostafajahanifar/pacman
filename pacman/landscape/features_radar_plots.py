# radar plots of features
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

from pacman.config import ALL_CANCERS, DATA_DIR, RESULTS_DIR

print(7*"="*7)
print("Radar plots of mitotic features")
print(7*"="*7)


def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    
    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped """
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)


        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)


                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

# Mean-Std Normalization function
def normalize_features(df, feature_cols):
    return (df[feature_cols] - df[feature_cols].mean()) / (df[feature_cols].std() + 1e-14)

# Function to create a radar chart for two subgroups
def radar_plot_with_error(df, feature_cols, temp_col='temperature', offset=0.5):
    # Normalize the feature columns using Min-Max normalization

    # Define Hot and Cold subgroups
    df_hot = df[df[temp_col] == 'Hot']
    df_cold = df[df[temp_col] == 'Cold']
    
    # Calculate mean and std for Hot and Cold groups
    hot_means = df_hot[feature_cols].mean()
    cold_means = df_cold[feature_cols].mean()
    hot_std = df_hot[feature_cols].std()
    cold_std = df_cold[feature_cols].std()
    all_means = df[feature_cols].mean()
    
    # Number of features
    num_vars = len(feature_cols)
    
    # Compute angle for each axis in the radar chart
    # angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles = radar_factory(num_vars, 'polygon')    
    # angles += angles[:1]  # Make the plot circular
    
    # Initialize the radar plot
    fig, ax = plt.subplots(figsize=(2,2), subplot_kw=dict(projection='radar'))
    
    # Colormap (coolwarm) for the two subgroups
    cmap = get_cmap('coolwarm')
    norm = Normalize(vmin=0, vmax=1)
    hot_color = cmap(norm(1))  # Maximum color for Hot
    cold_color = cmap(norm(0))  # Minimum color for Cold
    
    # Plot the radar chart for Hot group with error bars
    ax.fill(angles, hot_means, color=hot_color, alpha=0.25, label='Hot')
    ax.plot(angles, hot_means, color=hot_color, linewidth=2)
    # ax.errorbar(angles, hot_means, yerr=hot_std, fmt='o', color=hot_color, capsize=3)
    
    # Plot the radar chart for Cold group with error bars
    ax.fill(angles, cold_means, color=cold_color, alpha=0.25, label='Cold')
    ax.plot(angles, cold_means, color=cold_color, linewidth=2)
    # ax.errorbar(angles, cold_means, yerr=cold_std, fmt='o', color=cold_color, capsize=3)

    # Plot the radar chart of mitosis_feats, without subgrouping
    ax.plot(angles, all_means, color="black", linewidth=1)
    
    # # Fix the radar chart's labels
    ax.set_xticks(angles)
    ax.set_xticklabels('')
    # Custom padding based on the angle
    for i, angle in enumerate(angles):
        if i==0:
            ax.text(angle, 1.05+offset, feature_cols[i], horizontalalignment='center', verticalalignment='bottom')
        elif i == 1:
            ax.text(angle, 0.9+offset, feature_cols[i], horizontalalignment='right', verticalalignment='center', rotation=75)
        elif i==2:
            ax.text(angle, 1.1+offset, feature_cols[i], horizontalalignment='center', verticalalignment='top')
        elif i==3:
            ax.text(angle, 1.1+offset, feature_cols[i], horizontalalignment='center', verticalalignment='top')
        elif i==4:
            ax.text(angle, 0.9+offset, feature_cols[i], horizontalalignment='left', verticalalignment='center', rotation=-75)
        else:
            ax.text(angle, 0.8+offset, feature_cols[i], horizontalalignment='left', verticalalignment='center', rotation=-65)

    return fig, ax

# Load the mitosis_feats
mitosis_feats = pd.read_excel(os.path.join(DATA_DIR, "ST1-tcga_mtfs.xlsx"))

feature_cols = [
    "mean(ND)",
    "cv(ND)",
    "mean(CL)",
    "mean(HC)",
    "HSC",
]  # Add your actual feature columns here

mitosis_feats[feature_cols] = normalize_features(mitosis_feats, feature_cols)

for cancer in  ALL_CANCERS+["Pan-cancer"]: #
    if cancer == "Pan-cancer":
        df = mitosis_feats[mitosis_feats["type"].isin(ALL_CANCERS)]
    elif cancer == "ACC":
        df = mitosis_feats[mitosis_feats["type"]==cancer]
    else:
        df = mitosis_feats[mitosis_feats["type"]==cancer]

    offset= 0 if cancer=="Pan-cancer" else 3
    fig, ax = radar_plot_with_error(df, feature_cols, offset=offset)

    if cancer == "Pan-cancer":
        ax.set_rgrids([-1, 0, 1], angle=45) # for
    else:
        ax.set_rgrids([-2, 0, 2, 4], angle=45) # for all
    
    save_root = f"{RESULTS_DIR}/landscape/radar_plots/"
    os.makedirs(save_root, exist_ok=True)

    ax.set_title(cancer, pad=15)
    fig.savefig(save_root+f"radar_{cancer}.png", dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)

