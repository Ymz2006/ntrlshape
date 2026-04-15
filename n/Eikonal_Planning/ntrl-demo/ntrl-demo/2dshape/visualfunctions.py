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



# def visual_training(start, shape_points, env_points, cnt, speed, vmin, vmax=None):
#     to_visual_shapes = []
#     if vmax is None:
#         vmax = max(speed)

#     cmap = get_cmap('viridis')
#     norm = Normalize(vmin=vmin, vmax=vmax)

#     fig, ax = plt.subplots()

#     for i in range(cnt):
#         # Rotate points and return as list of (x, y) tuples
#         rotated_pts = rotate_points(shape_points, start[i][0], start[i][1], start[i][2])
        
#         # Skip if not enough points to make a polygon
#         if len(rotated_pts) < 3:
#             continue
        
#         rotated_shape = Polygon(rotated_pts)
#         if not rotated_shape.is_valid:
#             continue

#         to_visual_shapes.append(rotated_shape)

#         # Map speed to color
#         color = cmap(norm(speed[i]))
        
#         # Use matplotlib Polygon to draw the shape
#         patch = plt.Polygon(list(rotated_shape.exterior.coords), facecolor=color, edgecolor='black', alpha=0.7)
#         ax.add_patch(patch)

#     # Plot environment points
#     if len(env_points) > 0:
#         ax.scatter(*zip(*env_points), color='grey', s=10)

#     ax.set_aspect('equal')



#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])  # Required for ScalarMappable
#     cbar = plt.colorbar(sm, ax=ax)
#     cbar.set_label("Speed", rotation=270, labelpad=15)
#     plt.show()


import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from shapely.geometry import Polygon
# Make sure to import or define rotate_points wherever this script lives

def visual_training(start, shape_points, env_points, cnt, speed, vmin, vmax=None, begin_point=None, end_point=None):
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

    # --- NEW CODE: Plot begin_point and end_point ---
    # We loop over both points and plot them if they were provided
    # zorder=10 ensures they are drawn completely on top of everything else
    for pt in [begin_point, end_point]:
        if pt is not None:
            # Rotate shape points for the special point
            special_pts = rotate_points(shape_points, pt[0], pt[1], pt[2])
            
            if len(special_pts) >= 3:
                special_shape = Polygon(special_pts)
                if special_shape.is_valid:
                    # Draw the red polygon
                    special_patch = plt.Polygon(
                        list(special_shape.exterior.coords), 
                        facecolor='red', 
                        edgecolor='black', 
                        alpha=1.0, 
                        zorder=10
                    )
                    ax.add_patch(special_patch)
            
            # Optional: Add a high-contrast dot at the exact (x, y) center
            ax.scatter(pt[0], pt[1], color='red', edgecolor='white', s=30, zorder=11)
    # ------------------------------------------------

    ax.set_aspect('equal')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Required for ScalarMappable
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Speed", rotation=270, labelpad=15)
    plt.show()