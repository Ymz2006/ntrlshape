"""Visualization helpers for 2-D shape path-planning evaluation.

All plots are built with Plotly and written to HTML files so they can be
served over HTTP and viewed in a browser on a remote host.
"""

import numpy as np
import torch
import plotly.graph_objects as go
import plotly.colors as pc
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

    shape_points = torch.tensor(points, dtype=torch.float32)
    new_col = torch.ones((shape_points.shape[0], 1))
    shape_points = torch.cat([shape_points, new_col], dim=1)

    transformed = (rot @ shape_points.T).T
    return [tuple(p[:2].numpy()) for p in transformed]


def visual_training(start, shape_points, env_points, cnt, speed, vmin,
                    vmax=None, begin_point=None, end_point=None,
                    save_path=None):
    """Render a trajectory of shape placements colored by speed.

    Saves an interactive HTML file to `save_path` (required when running
    headless in Docker).  Falls back to fig.show() if save_path is None.
    """
    if vmax is None:
        vmax = float(np.max(speed))
    vmax = max(float(vmax), vmin + 1e-6)

    fig = go.Figure()

    # Trajectory polygons, one per waypoint, colored by speed
    for i in range(cnt):
        rotated_pts = rotate_points(shape_points, start[i][0], start[i][1], start[i][2])
        if len(rotated_pts) < 3:
            continue
        rotated_shape = Polygon(rotated_pts)
        if not rotated_shape.is_valid:
            continue

        norm_val = float(np.clip((float(speed[i]) - vmin) / (vmax - vmin), 0, 1))
        color = pc.sample_colorscale('Viridis', [norm_val])[0]

        xs, ys = rotated_shape.exterior.xy
        fig.add_trace(go.Scatter(
            x=list(xs), y=list(ys),
            fill='toself',
            mode='lines',
            line=dict(color='black', width=0.5),
            fillcolor=color,
            opacity=0.7,
            showlegend=False,
            hoverinfo='skip',
        ))

    # Environment boundary point cloud
    if len(env_points) > 0:
        env_xy = np.asarray(env_points)[:, :2]
        fig.add_trace(go.Scatter(
            x=env_xy[:, 0].tolist(), y=env_xy[:, 1].tolist(),
            mode='markers',
            marker=dict(color='grey', size=3),
            name='env',
            showlegend=False,
            hoverinfo='skip',
        ))

    # Start / goal poses highlighted in red
    for pt in [begin_point, end_point]:
        if pt is None:
            continue
        special_pts = rotate_points(shape_points, pt[0], pt[1], pt[2])
        if len(special_pts) >= 3:
            special_shape = Polygon(special_pts)
            if special_shape.is_valid:
                xs, ys = special_shape.exterior.xy
                fig.add_trace(go.Scatter(
                    x=list(xs), y=list(ys),
                    fill='toself',
                    mode='lines',
                    line=dict(color='black', width=1),
                    fillcolor='red',
                    opacity=1.0,
                    showlegend=False,
                    hoverinfo='skip',
                ))
        fig.add_trace(go.Scatter(
            x=[pt[0]], y=[pt[1]],
            mode='markers',
            marker=dict(color='red', size=8, line=dict(color='white', width=1)),
            showlegend=False,
        ))

    # Invisible dummy trace solely to render the speed colorbar
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(
            colorscale='Viridis',
            cmin=vmin, cmax=vmax,
            showscale=True,
            colorbar=dict(title='Speed'),
            color=[vmin],
        ),
        showlegend=False,
    ))

    fig.update_layout(
        yaxis=dict(scaleanchor='x', scaleratio=1),
        margin=dict(l=20, r=20, t=40, b=20),
    )

    if save_path:
        fig.write_html(save_path, include_plotlyjs='cdn')
    else:
        fig.show()
