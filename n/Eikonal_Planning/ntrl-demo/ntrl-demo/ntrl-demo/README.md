## About
This is a minimal example.

It contains two pipelines that share the same network:
* the original **mesh** pipeline (Gibson scenes / UR5 arm), and
* a **2-D shape** pipeline that plans a path for a rigid 2-D shape (e.g. the
  F-shape) moving through a 2-D environment described by normalized DXF files.

## Setup
1. git clone this repo
2. run `docker build -t ntrl:demo .` under the root directory of this repo, once you built the docker image, you don't need to build it again unless you change the dockerfile.
3. run `docker run -u $(id -u):$(id -g) --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume="/home/n/Eikonal_Planning/ntrl-demo:/workspace" --volume="/usr/lib/x86_64-linux-gnu/:/glu" --volume="/home/n/.local:/.local" --env="QT_X11_NO_MITSHM=1"  --gpus all -ti --rm ntrl:demo` to start the docker container.
4. run `pip install scipy` inside the container to install the KD-tree dependency

## Mesh pipeline (Gibson)
5. run `python dataprocessing/preprocess.py --config configs/gibson.txt ` to sample training data
6. run `python train/train_gib.py` to start the training.

## 2-D shape pipeline (DXF)
The configuration space is SE(2): every sample is `(x, y, theta)`, with `theta`
stored normalized by `2*pi` so the three coordinates share a comparable scale.
The environment and the moving shape are supplied as *normalized* DXF files
(normalized exactly like in `2dshape_baseline` — see
`dataprocessing/normaldxf.py` if you need to normalize a raw DXF first).

1. **Preprocess** — sample collision-free placements of the shape, compute the
   clearance speed and the normal-loss target:
   ```
   python dataprocessing/preprocess_dxf.py \
       --env   datasets/2dshape/FmazeEasy_norm.dxf \
       --shape datasets/2dshape/Fshape_norm.dxf \
       --out   datasets/2dshape/Fshape_FmazeEasy \
       --num_samples 400000 \
       --shape_scale 1.0
   ```
   `--shape_scale` controls the F-shape size (1.0 = the same size used by
   `2dshape_baseline`; smaller values shrink the moving body).  The chosen
   scale is recorded in `<out>/meta.json` so the visualizers pick it up
   automatically.

   This writes `sampled_points.npy`, `speed.npy`, `normal.npy`, `env.npy` and
   `meta.json` into the `--out` directory.  `normal.npy` is the unit gradient
   of the clearance field in configuration space: for every placement we take
   the closest point between the shape and the environment, build the
   workspace contact normal, and turn it into an SE(2) direction (`d/dx`,
   `d/dy`, `d/dtheta`).  This is exactly the target the network's normal-loss
   term (`models/metric/model_function_metric.py`) consumes.

2. **Inspect the training data** (optional, headless):
   ```
   python visualize_speeds.py --data datasets/2dshape/Fshape_FmazeEasy
   python visualize_speeds.py --data datasets/2dshape/Fshape_FmazeEasy --mode shapes
   python visualize_speeds.py --data datasets/2dshape/Fshape_FmazeEasy --mode hist
   ```
   Modes: `map` (interpolated heatmap, default), `scatter`, `shapes`, `hist`,
   `theta`.  Always saves PNGs to `--out_dir` (default `.`).

3. **Train**:
   ```
   python train/train_2dshape.py
   ```
   Periodic travel-time / speed plots (with the environment overlaid) are
   written into the experiment folder under `Experiments/2dshape/`.

4. **Visualize the trained network**:
   ```
   # single checkpoint
   python visualize_travel_field.py --pt Experiments/2dshape/<run>/Model_Epoch_05000_*.pt
   # evolution across a whole run (+ optional animation)
   python visualize_travel_field.py --run_dir Experiments/2dshape/<run> --animate
   ```
   This queries `T(start -> (x, y, theta_vis))` over a dense `(x, y)` grid and
   renders the travel-time field with isochrone contours, the environment
   boundary and the origin F-shape.  The F-shape size is read from
   `meta.json` in the training-data directory (override with `--shape_scale`).

The network architecture (`models/metric/model_network_metric.py`) is shared
unchanged between both pipelines.
