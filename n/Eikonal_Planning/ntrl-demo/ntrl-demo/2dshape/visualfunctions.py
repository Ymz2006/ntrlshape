import sys
import ezdxf
import numpy as np
from shapely.geometry import Point, Polygon
from parse_shape import dxf_to_shape, shape_to_points
from scipy.spatial import cKDTree
from normaldxf import visualize_shapes_or_files 
import torch 
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import get_cmap
from shapely.geometry import Polygon
from descartes import PolygonPatch  # For plotting shapely polygons

from matplotlib.colors import Normalize

import time
def rotate_points(points, x, y, theta):


    rot = torch.tensor([
        [np.cos(theta), -np.sin(theta), x],
        [np.sin(theta),  np.cos(theta), y],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    shape_points = torch.tensor(points, dtype=torch.float32)  # n x 2
    new_col = torch.ones((shape_points.shape[0], 1))
    shape_points = torch.cat([shape_points, new_col], dim=1)  # n x 3

    transformed = (rot @ shape_points.T).T  # n x 3

    # Return only x, y as list of tuples for Shapely
    return [tuple(p[:2].numpy()) for p in transformed]



def visual_training(start, shape_points, env_points, cnt, speed, vmin, vmax=None):
    to_visual_shapes = []
    if vmax is None:
        vmax = max(speed)

    cmap = get_cmap('viridis')
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots()

    for i in range(cnt):
        # Rotate points and return as list of (x, y) tuples
        rotated_pts = rotate_points(shape_points, start[i][0], start[i][1], start[i][2])
        
        # Skip if not enough points to make a polygon
        if len(rotated_pts) < 3:
            continue
        
        rotated_shape = Polygon(rotated_pts)
        if not rotated_shape.is_valid:
            continue

        to_visual_shapes.append(rotated_shape)

        # Map speed to color
        color = cmap(norm(speed[i]))
        
        # Use matplotlib Polygon to draw the shape
        patch = plt.Polygon(list(rotated_shape.exterior.coords), facecolor=color, edgecolor='black', alpha=0.7)
        ax.add_patch(patch)

    # Plot environment points
    if len(env_points) > 0:
        ax.scatter(*zip(*env_points), color='grey', s=10)

    ax.set_aspect('equal')



    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Required for ScalarMappable
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Speed", rotation=270, labelpad=15)
    plt.show()


