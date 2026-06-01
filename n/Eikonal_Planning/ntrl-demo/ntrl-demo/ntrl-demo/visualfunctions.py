"""Visualization helpers for 2-D shape path-planning evaluation.

Ported from the flat ``2dshape_new`` layout into the nested ntrl-demo package.
Only the pieces used by ``evaluate_training.py`` are kept (the heavy optional
deps — descartes / igl / normaldxf — were dropped).
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from shapely.geometry import Polygon


def rotate_points(points, x, y, theta):
    """Rotate `points` (n x 2) by `theta` and translate by (x, y).

    Returns a list of (x, y) tuples suitable for building a Shapely polygon.
    """
    rot = torch.tensor([
        [np.cos(theta), -np.sin(theta), x],
        [np.sin(theta),  np.cos(theta), y],
        [0, 0, 1]
    ], dtype=torch.float32)

    shape_points = torch.tensor(points, dtype=torch.float32)  # n x 2
    new_col = torch.ones((shape_points.shape[0], 1))
    shape_points = torch.cat([shape_points, new_col], dim=1)  # n x 3

    transformed = (rot @ shape_points.T).T  # n x 3
    return [tuple(p[:2].numpy()) for p in transformed]


def visual_training(start, shape_points, env_points, cnt, speed, vmin,
                    vmax=None, begin_point=None, end_point=None):
    """Render a trajectory of shape placements colored by speed.

    Each row of `start` is an (x, y, theta) placement; the shape outline is
    rotated/translated to that pose and drawn as a polygon. `begin_point` and
    `end_point`, if given, are drawn on top in red to mark the start/goal poses.
    """
    if vmax is None:
        vmax = max(speed)

    cmap = get_cmap('viridis')
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots()

    for i in range(cnt):
        rotated_pts = rotate_points(shape_points, start[i][0], start[i][1], start[i][2])
        if len(rotated_pts) < 3:
            continue
        rotated_shape = Polygon(rotated_pts)
        if not rotated_shape.is_valid:
            continue

        color = cmap(norm(speed[i]))
        patch = plt.Polygon(list(rotated_shape.exterior.coords),
                            facecolor=color, edgecolor='black', alpha=0.7)
        ax.add_patch(patch)

    # Environment boundary point cloud (only x, y are used)
    if len(env_points) > 0:
        env_xy = np.asarray(env_points)[:, :2]
        ax.scatter(env_xy[:, 0], env_xy[:, 1], color='grey', s=10)

    # Highlight start / goal poses in red, drawn on top of everything else
    for pt in [begin_point, end_point]:
        if pt is not None:
            special_pts = rotate_points(shape_points, pt[0], pt[1], pt[2])
            if len(special_pts) >= 3:
                special_shape = Polygon(special_pts)
                if special_shape.is_valid:
                    special_patch = plt.Polygon(
                        list(special_shape.exterior.coords),
                        facecolor='red', edgecolor='black', alpha=1.0, zorder=10)
                    ax.add_patch(special_patch)
            ax.scatter(pt[0], pt[1], color='red', edgecolor='white', s=30, zorder=11)

    ax.set_aspect('equal')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Speed", rotation=270, labelpad=15)
    plt.show()
