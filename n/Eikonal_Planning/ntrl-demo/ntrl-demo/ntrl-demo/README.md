## About
ntrl shape

## Setup
1. git clone this repo
2. run `docker build -f Dockerfile.server -t pytorchserver.` under the root directory of this repo, once you built the docker image, you don't need to build it again unless you change the dockerfile.
3. run `docker run --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume="/media/corallab-s1/4tbhdd/Jeffrey/ntrlshape/n/Eikonal_Planning/ntrl-demo/ntrl-demo:/workspace" --volume="/usr/lib/x86_64-linux-gnu/:/glu" --volume="/media/corallab-s1/4tbhdd/Jeffrey/ntrlshape/n/.local:/.local" --env="QT_X11_NO_MITSHM=1"  --gpus all -p 8080:8080 -ti --rm pytorchserver` to start the docker container.



## 2-D shape pipeline (DXF)

### FmazeEasy

1. **Preprocess** 
   ```
   python dataprocessing/preprocess_dxf.py \
       --env   datasets/2dshape/FmazeEasy_norm.dxf \
       --shape datasets/2dshape/Fshape_norm.dxf \
       --out   datasets/2dshape/Fshape_FmazeEasy \
       --num_samples 400000 \
       --visualize
   ```

   ```
   python dataprocessing/preprocess_dxf.py \
       --env   datasets/2dshape/FmazeEasy_norm.dxf \
       --shape datasets/2dshape/Fshape_norm.dxf \
       --out   testing_data/2dshape/Fshape_FmazeEasy \
       --num_samples 1000 \
       --testing_data \
       --visualize
   ```
2. **Train**:
   ```
   python train/train_2dshape.py --dataPath datasets/2dshape/Fshape_FmazeEasy
   ```

3. **Eval**:
   ```
   python evaluate_training.py --dataPath testing_data/2dshape/Fshape_FmazeEasy \
      --out ./results/output_2d/Fshape_FmazeEasy
   ```
### Fmaze2

1. **Preprocess** 
   ```
   python dataprocessing/preprocess_dxf.py \
       --env   datasets/2dshape/Fmaze2_norm.dxf \
       --shape datasets/2dshape/Fshape_norm.dxf \
       --out   datasets/2dshape/Fshape_Fmaze2 \
       --num_samples 400000 \
       --visualize
   ```

   ```
   python dataprocessing/preprocess_dxf.py \
       --env   datasets/2dshape/Fmaze2_norm.dxf \
       --shape datasets/2dshape/Fshape_norm.dxf \
       --out   testing_data/2dshape/Fshape_Fmaze2 \
       --num_samples 1000 \
       --testing_data \
       --visualize
   ```
2. **Train**:
   ```
   python train/train_2dshape.py --dataPath datasets/2dshape/Fshape_Fmaze2
   ```

3. **Eval**:
   ```
   python evaluate_training.py --dataPath testing_data/2dshape/Fshape_Fmaze2 \
      --out ./results/output_2d/Fshape_Fmaze2
   ```

