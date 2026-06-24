## About
ntrl shape

## Setup
1. git clone this repo
2. run `docker build -f Dockerfile.server -t pytorchserver.` under the root directory of this repo, once you built the docker image, you don't need to build it again unless you change the dockerfile.
3. run `docker run --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume="/media/corallab-s1/4tbhdd/Jeffrey/ntrlshape/n/Eikonal_Planning/ntrl-demo/ntrl-demo:/workspace" --volume="/usr/lib/x86_64-linux-gnu/:/glu" --volume="/media/corallab-s1/4tbhdd/Jeffrey/ntrlshape/n/.local:/.local" --env="QT_X11_NO_MITSHM=1"  --gpus all -p 8081:8081 -p 8080:8080 -ti --rm pytorchserver` to start the docker container.




## 3-D shape pipeline (OBJ)

### rectangle_env1

1. **Preprocess** 
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env1.obj \
        --shape datasets/3dshape/rectangle.obj \
        --out   datasets/3dshape/rectangle_env1 \
        --num_samples 400000 \
        --visualize
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env1.obj \
        --shape datasets/3dshape/rectangle.obj \
        --out   testing_data/3dshape/rectangle_env1 \
        --num_samples 1000 \
        --testing_data \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/rectangle_env1
   ```

3. **Eval**:
   ```
   python evaluate_training_3d.py --dataPath testing_data/3dshape/rectangle_env1 \
      --out ./results/output_3d/rectangle_env1
   ```



### Lshape_env1

1. **Preprocess** 
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env1.obj \
        --shape datasets/3dshape/Lshape3d.obj \
        --out   datasets/3dshape/Lshape3d_env1 \
        --num_samples 800000 \
        --visualize \
        --batch_size 2000 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env1.obj \
        --shape datasets/3dshape/Lshape3d.obj \
        --out   testing_data/3dshape/Lshape3d_env1 \
        --num_samples 1000 \
        --testing_data \
        --batch_size 1000 \
        --visualize 
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Lshape3d_env1
   ```

3. **Eval**:
   ```
   python evaluate_training_3d.py --dataPath testing_data/3dshape/Lshape3d_env1 \
      --out ./results/output_3d/Lshape3d_env1
   ```