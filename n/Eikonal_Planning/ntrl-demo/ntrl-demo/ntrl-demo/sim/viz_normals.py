"""Look at every boundary sample the push demo builds, and the normal it carries.

``PushIK`` resamples the T's footprint into ``P`` (points) / ``N`` (outward normals) and
every push primitive is defined against that pair -- a contact is a point of ``P`` and the
push directions are a fan about ``-N``.  If a normal points the wrong way the primitive
that uses it drives the pusher straight through the object, so this script draws them.

It rebuilds the geometry exactly as ``push_t_demo.main`` does (same mesh, same
MESH_TO_WORLD, same centroid recentring, same ``PushIK``) so what is plotted is what the
demo actually uses, and writes a self-contained plotly page with a slider: one page of
``--per-page`` samples at a time, arrows drawn from the point along its normal.

Each normal is also CHECKED rather than just drawn.  A correct outward normal at a
boundary point ``p`` has ``p + eps*n`` outside the true footprint polygon and ``p - eps*n``
inside it; arrows are coloured by that test, and the counts are printed.  Corner samples
(the ones ``PushIK.corner_margin`` rejects as contacts) are ringed in grey, because a
normal is genuinely ill-defined at a corner and a failure there means something different
from a failure in the middle of a face.

Usage (from the ntrl-demo root):
    python sim/viz_normals.py
    python sim/viz_normals.py --n-boundary 1000 --per-page 100 --out results/tee_normals.html
"""

import argparse
import os
import sys

import numpy as np
import plotly.graph_objects as go
from shapely.geometry import Point, Polygon

sys.path.append('.')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pymunk_viser_push as base
from push_t_demo import PushIK

COL_GOOD = '#2f9e6e'
COL_BAD = '#d94040'
COL_AMB = '#d9a441'
COL_OUTLINE = '#3a3f4b'
COL_FAINT = '#c9ccd4'
TAGS = (('outward', 1, COL_GOOD), ('inverted', -1, COL_BAD), ('ambiguous', 0, COL_AMB))


def classify(P, N, poly, eps):
    """Per-sample verdict on the normal: +1 outward, -1 inverted, 0 neither side agrees."""
    out = np.empty(len(P), dtype=int)
    for i, (p, n) in enumerate(zip(P, N)):
        fwd = poly.contains(Point(*(p + eps * n)))
        bwd = poly.contains(Point(*(p - eps * n)))
        out[i] = 1 if (not fwd and bwd) else (-1 if (fwd and not bwd) else 0)
    return out


def arrow_xy(P, N, length):
    """Shaft + two barbs per sample, as one None-separated polyline."""
    tip = P + N * length
    back = P + N * (length * 0.62)
    side = np.stack([-N[:, 1], N[:, 0]], axis=1) * (length * 0.22)
    nan = np.full((len(P), 1), np.nan)
    xs = np.concatenate([P[:, :1], tip[:, :1], nan,
                         (back + side)[:, :1], tip[:, :1], (back - side)[:, :1], nan],
                        axis=1).ravel()
    ys = np.concatenate([P[:, 1:], tip[:, 1:], nan,
                         (back + side)[:, 1:], tip[:, 1:], (back - side)[:, 1:], nan],
                        axis=1).ravel()
    return xs, ys


def title(ik, n, n_bad, n_amb, a, b):
    flag = ('all outward' if not (n_bad or n_amb)
            else f'<span style="color:{COL_BAD}">{n_bad} inverted, {n_amb} ambiguous</span>')
    return (f'T footprint normals &mdash; samples {a}&ndash;{b - 1} of {n} &nbsp;|&nbsp; '
            f'{flag} &nbsp;|&nbsp; corner margin {ik.corner_margin:.1f}, '
            f'{int(ik.ok.sum())} usable as contacts')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--shape', default='datasets/3dshape/Tshape3d.obj')
    ap.add_argument('--n-boundary', type=int, default=1000,
                    help='samples spread over the perimeter (PushIK rounds per edge)')
    ap.add_argument('--per-page', type=int, default=100)
    ap.add_argument('--pusher-radius', type=float, default=5.0,
                    help='only used for the default corner margin, as in the demo')
    ap.add_argument('--corner-margin', type=float, default=None,
                    help='default 8/7 * --pusher-radius, matching push_t_demo')
    ap.add_argument('--arrow', type=float, default=0.06,
                    help='arrow length as a fraction of the T bounding radius')
    ap.add_argument('--out', default='results/tee_normals.html')
    args = ap.parse_args()

    # ---- geometry, identical to push_t_demo.main -------------------------------------
    tee_mesh = base.load_mesh(args.shape)
    tee_polys = base.footprint(tee_mesh)
    assert len(tee_polys) == 1, f'expected one footprint for the shape, got {len(tee_polys)}'
    c = tee_polys[0].centroid
    tee_poly = Polygon([(x - c.x, y - c.y) for x, y in tee_polys[0].exterior.coords])

    corner = (args.corner_margin if args.corner_margin is not None
              else 8.0 / 7.0 * args.pusher_radius)
    ik = PushIK(tee_poly, 1.0, n_boundary=args.n_boundary, corner_margin=corner)
    P, N, ring = ik.P, ik.N, ik.ring

    # ---- check -----------------------------------------------------------------------
    spacing = float(np.median(np.linalg.norm(np.diff(P, axis=0), axis=1)))
    eps = 0.25 * spacing
    verdict = classify(P, N, tee_poly, eps)
    unit_err = float(np.abs(np.linalg.norm(N, axis=1) - 1.0).max())
    n_bad = int((verdict == -1).sum())
    n_amb = int((verdict == 0).sum())

    print(f'[geom] {len(P)} samples over a perimeter of {tee_poly.exterior.length:.1f} '
          f'units, spacing {spacing:.2f}, probe eps {eps:.3f}')
    # PushIK re-orients the polygon, so report the winding of the ring it actually used:
    # the normal formula [e_y, -e_x] is only outward for a CCW ring.
    area2 = float(np.cross(ring, np.roll(ring, -1, axis=0)).sum())
    print(f'[geom] ring has {len(ring)} corners, PushIK winding '
          f'{"CCW" if area2 > 0 else "CW"} (input was '
          f'{"CCW" if tee_poly.exterior.is_ccw else "CW"}); '
          f'max ||n||-1 = {unit_err:.2e}')
    print(f'[check] outward {int((verdict == 1).sum())}   inverted {n_bad}   '
          f'ambiguous {n_amb}')
    for label, mask in (('inverted', verdict == -1), ('ambiguous', verdict == 0)):
        idx = np.flatnonzero(mask)
        if len(idx):
            gap = ik.corner_gap[idx]
            print(f'[check] {label}: indices {idx[:12].tolist()}'
                  f'{" ..." if len(idx) > 12 else ""}; corner gap min {gap.min():.2f} '
                  f'median {np.median(gap):.2f} '
                  f'({int((gap < corner).sum())}/{len(idx)} inside the {corner:.1f}-unit '
                  f'corner margin)')

    # ---- figure ----------------------------------------------------------------------
    L = args.arrow * ik.radius
    pages = [(s, min(s + args.per_page, len(P))) for s in range(0, len(P), args.per_page)]

    fig = go.Figure()
    closed = np.vstack([ring, ring[:1]])
    fig.add_trace(go.Scatter(x=closed[:, 0], y=closed[:, 1], mode='lines',
                             line=dict(color=COL_OUTLINE, width=2),
                             name='footprint', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=P[:, 0], y=P[:, 1], mode='markers',
                             marker=dict(size=2.5, color=COL_FAINT),
                             name='all samples', hoverinfo='skip'))
    static = len(fig.data)

    for a, b in pages:
        sl = slice(a, b)
        v, gap = verdict[sl], ik.corner_gap[sl]
        for tag, want, col in TAGS:
            keep = v == want
            p, n = P[sl][keep], N[sl][keep]
            xs, ys = arrow_xy(p, n, L) if len(p) else ([], [])
            fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', visible=False,
                                     line=dict(color=col, width=1.4),
                                     name=tag, legendgroup=tag, hoverinfo='skip'))
            ids = np.flatnonzero(keep) + a
            fig.add_trace(go.Scatter(
                x=p[:, 0] if len(p) else [], y=p[:, 1] if len(p) else [],
                mode='markers', visible=False, showlegend=False, legendgroup=tag,
                marker=dict(size=6, color=col, line=dict(color='white', width=0.8)),
                customdata=(np.column_stack([ids, n, gap[keep]]) if len(p) else None),
                hovertemplate=('sample %{customdata[0]}<br>'
                               'p = (%{x:.2f}, %{y:.2f})<br>'
                               'n = (%{customdata[1]:.3f}, %{customdata[2]:.3f})<br>'
                               'corner gap %{customdata[3]:.2f}<extra></extra>')))
        rej = gap < corner
        fig.add_trace(go.Scatter(
            x=P[sl][rej][:, 0] if rej.any() else [],
            y=P[sl][rej][:, 1] if rej.any() else [],
            mode='markers', visible=False, name='corner-rejected',
            marker=dict(size=11, color='rgba(0,0,0,0)',
                        line=dict(color='#7b7f8c', width=1.2)), hoverinfo='skip'))

    per_page = 2 * len(TAGS) + 1
    for k in range(static, static + per_page):
        fig.data[k].visible = True

    steps = []
    for i, (a, b) in enumerate(pages):
        vis = [True] * static + [False] * (len(pages) * per_page)
        for k in range(per_page):
            vis[static + i * per_page + k] = True
        steps.append(dict(method='update', label=f'{a}-{b - 1}',
                          args=[{'visible': vis},
                                {'title.text': title(ik, len(P), n_bad, n_amb, a, b)}]))

    fig.update_layout(
        title=dict(text=title(ik, len(P), n_bad, n_amb, *pages[0]), x=0.02,
                   font=dict(size=15)),
        sliders=[dict(active=0, currentvalue=dict(prefix='samples '), pad=dict(t=40),
                      steps=steps)],
        template='plotly_white', width=1000, height=900,
        legend=dict(orientation='h', y=1.02, x=0.02),
        margin=dict(l=40, r=20, t=90, b=90))
    fig.update_yaxes(scaleanchor='x', scaleratio=1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.write_html(args.out, include_plotlyjs='cdn')
    print(f'[out] {args.out}')


if __name__ == '__main__':
    main()
