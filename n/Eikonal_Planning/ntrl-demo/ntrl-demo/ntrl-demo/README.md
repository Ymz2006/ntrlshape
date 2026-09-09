## About
ntrl shape

## Setup
1. git clone this repo
2. run `docker build -f Dockerfile.server -t pytorchserver.` under the root directory of this repo, once you built the docker image, you don't need to build it again unless you change the dockerfile.
3. run `docker run --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume="/media/corallab-s1/4tbhdd/Jeffrey/ntrlshape/n/Eikonal_Planning/ntrl-demo/ntrl-demo:/workspace" --volume="/usr/lib/x86_64-linux-gnu/:/glu" --volume="/media/corallab-s1/4tbhdd/Jeffrey/ntrlshape/n/.local:/.local" --env="QT_X11_NO_MITSHM=1"  --gpus all -p 8081:8081 -p 8080:8080 -ti --rm pytorchserver` to start the docker container.




## 3-D shape pipeline (OBJ)

Every entry below follows the same three steps -- preprocess (training set, then the
held-out testing set), train, evaluate -- and differs only in the `--env` / `--shape`
meshes and the dataset name (`<shape>_<env>`).

### env1

#### rectangle_env1

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env1.obj \
        --shape datasets/3dshape/rectangle.obj \
        --out   datasets/3dshape/rectangle_env1 \
        --num_samples 800000 \
        --visualize \
        --batch_size 2000 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env1.obj \
        --shape datasets/3dshape/rectangle.obj \
        --out   testing_data/3dshape/rectangle_env1 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/rectangle_env1
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/rectangle_env1 \
      --out ./results/output_3d/rectangle_env1
   ```

#### Lshape3d_env1

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
        --offset 0.02 \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Lshape3d_env1
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Lshape3d_env1 \
      --out ./results/output_3d/Lshape3d_env1
   ```

#### Fshape3d_env1

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env1.obj \
        --shape datasets/3dshape/Fshape3d.obj \
        --out   datasets/3dshape/Fshape3d_env1 \
        --num_samples 800000 \
        --visualize \
        --batch_size 2000 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env1.obj \
        --shape datasets/3dshape/Fshape3d.obj \
        --out   testing_data/3dshape/Fshape3d_env1 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Fshape3d_env1
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Fshape3d_env1 \
      --out ./results/output_3d/Fshape3d_env1
   ```

#### Ashape3d_env1

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env1.obj \
        --shape datasets/3dshape/Ashape3d.obj \
        --out   datasets/3dshape/Ashape3d_env1 \
        --num_samples 800000 \
        --visualize \
        --batch_size 2000 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env1.obj \
        --shape datasets/3dshape/Ashape3d.obj \
        --out   testing_data/3dshape/Ashape3d_env1 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Ashape3d_env1
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Ashape3d_env1 \
      --out ./results/output_3d/Ashape3d_env1
   ```

#### Vshape3d_env1

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env1.obj \
        --shape datasets/3dshape/Vshape3d.obj \
        --out   datasets/3dshape/Vshape3d_env1 \
        --num_samples 800000 \
        --visualize \
        --batch_size 2000 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env1.obj \
        --shape datasets/3dshape/Vshape3d.obj \
        --out   testing_data/3dshape/Vshape3d_env1 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Vshape3d_env1
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Vshape3d_env1 \
      --out ./results/output_3d/Vshape3d_env1
   ```

#### 4shape3d_env1

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env1.obj \
        --shape datasets/3dshape/4shape3d.obj \
        --out   datasets/3dshape/4shape3d_env1 \
        --num_samples 800000 \
        --visualize \
        --batch_size 2000 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env1.obj \
        --shape datasets/3dshape/4shape3d.obj \
        --out   testing_data/3dshape/4shape3d_env1 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/4shape3d_env1
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/4shape3d_env1 \
      --out ./results/output_3d/4shape3d_env1
   ```


### env2

#### rectangle_env2

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env2.obj \
        --shape datasets/3dshape/rectangle.obj \
        --out   datasets/3dshape/rectangle_env2 \
        --num_samples 800000 \
        --visualize \
        --batch_size 2000 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env2.obj \
        --shape datasets/3dshape/rectangle.obj \
        --out   testing_data/3dshape/rectangle_env2 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/rectangle_env2
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/rectangle_env2 \
      --out ./results/output_3d/rectangle_env2
   ```

#### Lshape3d_env2

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env2.obj \
        --shape datasets/3dshape/Lshape3d.obj \
        --out   datasets/3dshape/Lshape3d_env2 \
        --num_samples 800000 \
        --visualize \
        --batch_size 2000 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env2.obj \
        --shape datasets/3dshape/Lshape3d.obj \
        --out   testing_data/3dshape/Lshape3d_env2 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Lshape3d_env2
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Lshape3d_env2 \
      --out ./results/output_3d/Lshape3d_env2
   ```

#### Fshape3d_env2

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env2.obj \
        --shape datasets/3dshape/Fshape3d.obj \
        --out   datasets/3dshape/Fshape3d_env2 \
        --num_samples 800000 \
        --visualize \
        --batch_size 2000 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env2.obj \
        --shape datasets/3dshape/Fshape3d.obj \
        --out   testing_data/3dshape/Fshape3d_env2 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Fshape3d_env2
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Fshape3d_env2 \
      --out ./results/output_3d/Fshape3d_env2
   ```

#### Ashape3d_env2

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env2.obj \
        --shape datasets/3dshape/Ashape3d.obj \
        --out   datasets/3dshape/Ashape3d_env2 \
        --num_samples 800000 \
        --visualize \
        --batch_size 2000 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env2.obj \
        --shape datasets/3dshape/Ashape3d.obj \
        --out   testing_data/3dshape/Ashape3d_env2 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Ashape3d_env2
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Ashape3d_env2 \
      --out ./results/output_3d/Ashape3d_env2
   ```

#### Vshape3d_env2

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env2.obj \
        --shape datasets/3dshape/Vshape3d.obj \
        --out   datasets/3dshape/Vshape3d_env2 \
        --num_samples 800000 \
        --visualize \
        --batch_size 2000 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env2.obj \
        --shape datasets/3dshape/Vshape3d.obj \
        --out   testing_data/3dshape/Vshape3d_env2 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Vshape3d_env2
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Vshape3d_env2 \
      --out ./results/output_3d/Vshape3d_env2
   ```

#### 4shape3d_env2

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env2.obj \
        --shape datasets/3dshape/4shape3d.obj \
        --out   datasets/3dshape/4shape3d_env2 \
        --num_samples 800000 \
        --visualize \
        --batch_size 2000 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env2.obj \
        --shape datasets/3dshape/4shape3d.obj \
        --out   testing_data/3dshape/4shape3d_env2 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/4shape3d_env2
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/4shape3d_env2 \
      --out ./results/output_3d/4shape3d_env2
   ```


### env3

#### rectangle_env3

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env3.obj \
        --shape datasets/3dshape/rectangle.obj \
        --out   datasets/3dshape/rectangle_env3 \
        --num_samples 800000 \
        --visualize \
        --batch_size 2000 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env3.obj \
        --shape datasets/3dshape/rectangle.obj \
        --out   testing_data/3dshape/rectangle_env3 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/rectangle_env3
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/rectangle_env3 \
      --out ./results/output_3d/rectangle_env3
   ```

#### Lshape3d_env3

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env3.obj \
        --shape datasets/3dshape/Lshape3d.obj \
        --out   datasets/3dshape/Lshape3d_env3 \
        --num_samples 800000 \
        --visualize \
        --batch_size 2000 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env3.obj \
        --shape datasets/3dshape/Lshape3d.obj \
        --out   testing_data/3dshape/Lshape3d_env3 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Lshape3d_env3
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Lshape3d_env3 \
      --out ./results/output_3d/Lshape3d_env3
   ```

#### Fshape3d_env3

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env3.obj \
        --shape datasets/3dshape/Fshape3d.obj \
        --out   datasets/3dshape/Fshape3d_env3 \
        --num_samples 800000 \
        --visualize \
        --batch_size 2000 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env3.obj \
        --shape datasets/3dshape/Fshape3d.obj \
        --out   testing_data/3dshape/Fshape3d_env3 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Fshape3d_env3
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Fshape3d_env3 \
      --out ./results/output_3d/Fshape3d_env3
   ```

#### Ashape3d_env3

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env3.obj \
        --shape datasets/3dshape/Ashape3d.obj \
        --out   datasets/3dshape/Ashape3d_env3 \
        --num_samples 800000 \
        --visualize \
        --batch_size 2000 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env3.obj \
        --shape datasets/3dshape/Ashape3d.obj \
        --out   testing_data/3dshape/Ashape3d_env3 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Ashape3d_env3
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Ashape3d_env3 \
      --out ./results/output_3d/Ashape3d_env3
   ```

#### Vshape3d_env3

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env3.obj \
        --shape datasets/3dshape/Vshape3d.obj \
        --out   datasets/3dshape/Vshape3d_env3 \
        --num_samples 800000 \
        --visualize \
        --batch_size 2000 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env3.obj \
        --shape datasets/3dshape/Vshape3d.obj \
        --out   testing_data/3dshape/Vshape3d_env3 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Vshape3d_env3
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Vshape3d_env3 \
      --out ./results/output_3d/Vshape3d_env3
   ```

#### 4shape3d_env3

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env3.obj \
        --shape datasets/3dshape/4shape3d.obj \
        --out   datasets/3dshape/4shape3d_env3 \
        --num_samples 800000 \
        --visualize \
        --batch_size 2000 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env3.obj \
        --shape datasets/3dshape/4shape3d.obj \
        --out   testing_data/3dshape/4shape3d_env3 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/4shape3d_env3
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/4shape3d_env3 \
      --out ./results/output_3d/4shape3d_env3
   ```


### env4

#### rectangle_env4

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env4.obj \
        --shape datasets/3dshape/rectangle.obj \
        --out   datasets/3dshape/rectangle_env4 \
        --num_samples 800000 \
        --visualize \
        --batch_size 2000 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env4.obj \
        --shape datasets/3dshape/rectangle.obj \
        --out   testing_data/3dshape/rectangle_env4 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/rectangle_env4
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/rectangle_env4 \
      --out ./results/output_3d/rectangle_env4
   ```

#### Lshape3d_env4

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env4.obj \
        --shape datasets/3dshape/Lshape3d.obj \
        --out   datasets/3dshape/Lshape3d_env4 \
        --num_samples 800000 \
        --visualize \
        --batch_size 2000 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env4.obj \
        --shape datasets/3dshape/Lshape3d.obj \
        --out   testing_data/3dshape/Lshape3d_env4 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Lshape3d_env4
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Lshape3d_env4 \
      --out ./results/output_3d/Lshape3d_env4
   ```

#### Fshape3d_env4

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env4.obj \
        --shape datasets/3dshape/Fshape3d.obj \
        --out   datasets/3dshape/Fshape3d_env4 \
        --num_samples 800000 \
        --visualize \
        --batch_size 2000 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env4.obj \
        --shape datasets/3dshape/Fshape3d.obj \
        --out   testing_data/3dshape/Fshape3d_env4 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Fshape3d_env4
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Fshape3d_env4 \
      --out ./results/output_3d/Fshape3d_env4
   ```

#### Ashape3d_env4

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env4.obj \
        --shape datasets/3dshape/Ashape3d.obj \
        --out   datasets/3dshape/Ashape3d_env4 \
        --num_samples 800000 \
        --visualize \
        --batch_size 2000 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env4.obj \
        --shape datasets/3dshape/Ashape3d.obj \
        --out   testing_data/3dshape/Ashape3d_env4 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Ashape3d_env4
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Ashape3d_env4 \
      --out ./results/output_3d/Ashape3d_env4
   ```

#### Vshape3d_env4

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env4.obj \
        --shape datasets/3dshape/Vshape3d.obj \
        --out   datasets/3dshape/Vshape3d_env4 \
        --num_samples 800000 \
        --visualize \
        --batch_size 2000 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env4.obj \
        --shape datasets/3dshape/Vshape3d.obj \
        --out   testing_data/3dshape/Vshape3d_env4 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Vshape3d_env4
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Vshape3d_env4 \
      --out ./results/output_3d/Vshape3d_env4
   ```

#### 4shape3d_env4

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env4.obj \
        --shape datasets/3dshape/4shape3d.obj \
        --out   datasets/3dshape/4shape3d_env4 \
        --num_samples 800000 \
        --visualize \
        --batch_size 2000 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env4.obj \
        --shape datasets/3dshape/4shape3d.obj \
        --out   testing_data/3dshape/4shape3d_env4 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/4shape3d_env4
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/4shape3d_env4 \
      --out ./results/output_3d/4shape3d_env4
   ```


### Corozal

#### Lcouch_Corozal

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/Corozal.obj \
        --shape datasets/3dshape/Lcouch.obj \
        --out   datasets/3dshape/Lcouch_Corozal \
        --num_samples 800000 \
        --visualize \
        --batch_size 2000 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/Corozal.obj \
        --shape datasets/3dshape/Lcouch.obj \
        --out   testing_data/3dshape/Lcouch_Corozal \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Lcouch_Corozal
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Lcouch_Corozal \
      --out ./results/output_3d/Lcouch_Corozal
   ```


## 2-D shape pipeline (OBJ, `--2d`)

Same network and same trainer as the 3-D pipeline -- the configuration space is just
restricted to the `(x, y, rz)` slice by passing `--2d` to the preprocessor and to the
evaluator (the data is still stored in the 6-D SE(3) layout with `z`, `rx`, `ry` pinned
to 0, so `train/train_3dshape.py` is used unchanged).  Both meshes must be z-up.

Four environments, seven shapes each.  `2denv4_zup.obj` is the 8-body maze; the
denser 12-body `2denv1_zup.obj` and 13-body `2denv2_zup.obj` are the same
350 x 350 footprint with more obstacles, and `2denv3_zup.obj` is the sparsest of
the set at 6 bodies.  Datasets are named `<shape>_<env>` with the env tags `2denv4`, `2denv1`,
`2denv2` and `2denv3`, which keeps them clear of the 3-D `<shape>_env4` /
`<shape>_env1` / `<shape>_env2` / `<shape>_env3` datasets built from `env4.obj` /
`env1.obj` / `env2.obj` / `env3.obj`.

### Mesh preparation (z-up)

Every mesh in `datasets/3dshape` is extruded along its own **+Y**: the shapes are 10
units thick in y, `2denv4.obj`, `2denv1.obj`, `2denv2.obj` and `2denv3.obj` are
350 x 30 x 350.  `--2d`, however,
defines the planar slice as **z = 0** -- it flattens the environment onto `z=0` and
squashes the shape's z extent.  Pointing `--2d` at a Y-up mesh therefore collapses one
of the two *in-plane* axes, no placement is ever collision-free, and the rejection loop
in `generate_valid_pairs` spins forever.  Rotate every input once:

```
python dataprocessing/obj_yup_to_zup.py \
    datasets/3dshape/2denv1.obj \
    datasets/3dshape/2denv2.obj \
    datasets/3dshape/2denv3.obj \
    datasets/3dshape/2denv4.obj \
    datasets/3dshape/rectangle.obj \
    datasets/3dshape/Lshape3d.obj \
    datasets/3dshape/Fshape3d.obj \
    datasets/3dshape/Ashape3d.obj \
    datasets/3dshape/Vshape3d.obj \
    datasets/3dshape/4shape3d.obj \
    datasets/3dshape/Tshape3d.obj
```

which writes `<name>_zup.obj` beside each input.

### A note on `--batch_size`

Use a much smaller batch here than in 3-D.  Flattening the environment onto `z=0`
defeats the broad-phase env cull in `evaluate_placements` (every env point ends up
within `min_center_dist + 2*R_shape` of the placement), so the `(B, kept env points, F)`
clearance tensor stays dense and peak memory scales with the shape's boundary-triangle
count.  The 3-D `--batch_size 2000` needs over 23 GiB on a planar env and OOMs a 24 GiB
card on its own for the larger shapes; `--batch_size 500` peaks near 9.5 GiB and is what
the `2denv4` / `2denv1` / `2denv2` commands below use.  Batch size only changes how the
rejection sampler is chunked, never the sampled distribution, so it is safe to retune
per sweep: the only failure mode is an OOM, and the dataset that comes out is the same
either way.  The `2denv3` commands are written at `--batch_size 1000` on that basis --
roughly double the peak, still inside a 24 GiB card for the smaller shapes; drop it back
to 500 for `Ashape3d` / `4shape3d` / `Fshape3d` if the sweep OOMs.  `2denv3` is also
the sparsest environment (6 bodies against `2denv4`'s 8 and `2denv2`'s 13), so its
clearance tensor is the smallest of the four to begin with.

### 2denv4 (`2denv4_zup.obj`)

#### rectangle_2denv4

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv4_zup.obj \
        --shape datasets/3dshape/rectangle_zup.obj \
        --out   datasets/3dshape/rectangle_2denv4 \
        --num_samples 800000 \
        --2d \
        --visualize \
        --batch_size 500 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv4_zup.obj \
        --shape datasets/3dshape/rectangle_zup.obj \
        --out   testing_data/3dshape/rectangle_2denv4 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --2d \
        --batch_size 500 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/rectangle_2denv4 \
      --modelPath ./Experiments/3dshape_2d --name rectangle_2denv4
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/rectangle_2denv4 \
      --out ./results/output_3d/rectangle_2denv4 \
      --checkpoint ./Experiments/3dshape_2d/rectangle_2denv4/latest.pt \
      --2d
   ```

#### Lshape3d_2denv4

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv4_zup.obj \
        --shape datasets/3dshape/Lshape3d_zup.obj \
        --out   datasets/3dshape/Lshape3d_2denv4 \
        --num_samples 800000 \
        --2d \
        --visualize \
        --batch_size 500 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv4_zup.obj \
        --shape datasets/3dshape/Lshape3d_zup.obj \
        --out   testing_data/3dshape/Lshape3d_2denv4 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --2d \
        --batch_size 500 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Lshape3d_2denv4 \
      --modelPath ./Experiments/3dshape_2d --name Lshape3d_2denv4
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Lshape3d_2denv4 \
      --out ./results/output_3d/Lshape3d_2denv4 \
      --checkpoint ./Experiments/3dshape_2d/Lshape3d_2denv4/latest.pt \
      --2d
   ```

#### Fshape3d_2denv4

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv4_zup.obj \
        --shape datasets/3dshape/Fshape3d_zup.obj \
        --out   datasets/3dshape/Fshape3d_2denv4 \
        --num_samples 800000 \
        --2d \
        --visualize \
        --batch_size 500 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv4_zup.obj \
        --shape datasets/3dshape/Fshape3d_zup.obj \
        --out   testing_data/3dshape/Fshape3d_2denv4 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --2d \
        --batch_size 500 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Fshape3d_2denv4 \
      --modelPath ./Experiments/3dshape_2d --name Fshape3d_2denv4
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Fshape3d_2denv4 \
      --out ./results/output_3d/Fshape3d_2denv4 \
      --checkpoint ./Experiments/3dshape_2d/Fshape3d_2denv4/latest.pt \
      --2d
   ```

#### Ashape3d_2denv4

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv4_zup.obj \
        --shape datasets/3dshape/Ashape3d_zup.obj \
        --out   datasets/3dshape/Ashape3d_2denv4 \
        --num_samples 800000 \
        --2d \
        --visualize \
        --batch_size 500 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv4_zup.obj \
        --shape datasets/3dshape/Ashape3d_zup.obj \
        --out   testing_data/3dshape/Ashape3d_2denv4 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --2d \
        --batch_size 500 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Ashape3d_2denv4 \
      --modelPath ./Experiments/3dshape_2d --name Ashape3d_2denv4
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Ashape3d_2denv4 \
      --out ./results/output_3d/Ashape3d_2denv4 \
      --checkpoint ./Experiments/3dshape_2d/Ashape3d_2denv4/latest.pt \
      --2d
   ```

#### Vshape3d_2denv4

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv4_zup.obj \
        --shape datasets/3dshape/Vshape3d_zup.obj \
        --out   datasets/3dshape/Vshape3d_2denv4 \
        --num_samples 800000 \
        --2d \
        --visualize \
        --batch_size 500 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv4_zup.obj \
        --shape datasets/3dshape/Vshape3d_zup.obj \
        --out   testing_data/3dshape/Vshape3d_2denv4 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --2d \
        --batch_size 500 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Vshape3d_2denv4 \
      --modelPath ./Experiments/3dshape_2d --name Vshape3d_2denv4
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Vshape3d_2denv4 \
      --out ./results/output_3d/Vshape3d_2denv4 \
      --checkpoint ./Experiments/3dshape_2d/Vshape3d_2denv4/latest.pt \
      --2d
   ```

#### 4shape3d_2denv4

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv4_zup.obj \
        --shape datasets/3dshape/4shape3d_zup.obj \
        --out   datasets/3dshape/4shape3d_2denv4 \
        --num_samples 800000 \
        --2d \
        --visualize \
        --batch_size 500 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv4_zup.obj \
        --shape datasets/3dshape/4shape3d_zup.obj \
        --out   testing_data/3dshape/4shape3d_2denv4 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --2d \
        --batch_size 500 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/4shape3d_2denv4 \
      --modelPath ./Experiments/3dshape_2d --name 4shape3d_2denv4
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/4shape3d_2denv4 \
      --out ./results/output_3d/4shape3d_2denv4 \
      --checkpoint ./Experiments/3dshape_2d/4shape3d_2denv4/latest.pt \
      --2d
   ```

#### Tshape3d_2denv4

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv4_zup.obj \
        --shape datasets/3dshape/Tshape3d_zup.obj \
        --out   datasets/3dshape/Tshape3d_2denv4 \
        --num_samples 800000 \
        --2d \
        --visualize \
        --batch_size 500 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv4_zup.obj \
        --shape datasets/3dshape/Tshape3d_zup.obj \
        --out   testing_data/3dshape/Tshape3d_2denv4 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --2d \
        --batch_size 500 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Tshape3d_2denv4 \
      --modelPath ./Experiments/3dshape_2d --name Tshape3d_2denv4
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Tshape3d_2denv4 \
      --out ./results/output_3d/Tshape3d_2denv4 \
      --checkpoint ./Experiments/3dshape_2d/Tshape3d_2denv4/latest.pt \
      --2d
   ```


### 2denv1 (`2denv1_zup.obj`)

#### rectangle_2denv1

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv1_zup.obj \
        --shape datasets/3dshape/rectangle_zup.obj \
        --out   datasets/3dshape/rectangle_2denv1 \
        --num_samples 800000 \
        --2d \
        --visualize \
        --batch_size 500 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv1_zup.obj \
        --shape datasets/3dshape/rectangle_zup.obj \
        --out   testing_data/3dshape/rectangle_2denv1 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --2d \
        --batch_size 500 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/rectangle_2denv1 \
      --modelPath ./Experiments/3dshape_2d --name rectangle_2denv1
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/rectangle_2denv1 \
      --out ./results/output_3d/rectangle_2denv1 \
      --checkpoint ./Experiments/3dshape_2d/rectangle_2denv1/latest.pt \
      --2d
   ```

#### Lshape3d_2denv1

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv1_zup.obj \
        --shape datasets/3dshape/Lshape3d_zup.obj \
        --out   datasets/3dshape/Lshape3d_2denv1 \
        --num_samples 800000 \
        --2d \
        --visualize \
        --batch_size 500 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv1_zup.obj \
        --shape datasets/3dshape/Lshape3d_zup.obj \
        --out   testing_data/3dshape/Lshape3d_2denv1 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --2d \
        --batch_size 500 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Lshape3d_2denv1 \
      --modelPath ./Experiments/3dshape_2d --name Lshape3d_2denv1
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Lshape3d_2denv1 \
      --out ./results/output_3d/Lshape3d_2denv1 \
      --checkpoint ./Experiments/3dshape_2d/Lshape3d_2denv1/latest.pt \
      --2d
   ```

#### Fshape3d_2denv1

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv1_zup.obj \
        --shape datasets/3dshape/Fshape3d_zup.obj \
        --out   datasets/3dshape/Fshape3d_2denv1 \
        --num_samples 800000 \
        --2d \
        --visualize \
        --batch_size 500 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv1_zup.obj \
        --shape datasets/3dshape/Fshape3d_zup.obj \
        --out   testing_data/3dshape/Fshape3d_2denv1 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --2d \
        --batch_size 500 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Fshape3d_2denv1 \
      --modelPath ./Experiments/3dshape_2d --name Fshape3d_2denv1
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Fshape3d_2denv1 \
      --out ./results/output_3d/Fshape3d_2denv1 \
      --checkpoint ./Experiments/3dshape_2d/Fshape3d_2denv1/latest.pt \
      --2d
   ```

#### Ashape3d_2denv1

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv1_zup.obj \
        --shape datasets/3dshape/Ashape3d_zup.obj \
        --out   datasets/3dshape/Ashape3d_2denv1 \
        --num_samples 800000 \
        --2d \
        --visualize \
        --batch_size 500 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv1_zup.obj \
        --shape datasets/3dshape/Ashape3d_zup.obj \
        --out   testing_data/3dshape/Ashape3d_2denv1 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --2d \
        --batch_size 500 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Ashape3d_2denv1 \
      --modelPath ./Experiments/3dshape_2d --name Ashape3d_2denv1
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Ashape3d_2denv1 \
      --out ./results/output_3d/Ashape3d_2denv1 \
      --checkpoint ./Experiments/3dshape_2d/Ashape3d_2denv1/latest.pt \
      --2d
   ```

#### Vshape3d_2denv1

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv1_zup.obj \
        --shape datasets/3dshape/Vshape3d_zup.obj \
        --out   datasets/3dshape/Vshape3d_2denv1 \
        --num_samples 800000 \
        --2d \
        --visualize \
        --batch_size 500 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv1_zup.obj \
        --shape datasets/3dshape/Vshape3d_zup.obj \
        --out   testing_data/3dshape/Vshape3d_2denv1 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --2d \
        --batch_size 500 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Vshape3d_2denv1 \
      --modelPath ./Experiments/3dshape_2d --name Vshape3d_2denv1
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Vshape3d_2denv1 \
      --out ./results/output_3d/Vshape3d_2denv1 \
      --checkpoint ./Experiments/3dshape_2d/Vshape3d_2denv1/latest.pt \
      --2d
   ```

#### 4shape3d_2denv1

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv1_zup.obj \
        --shape datasets/3dshape/4shape3d_zup.obj \
        --out   datasets/3dshape/4shape3d_2denv1 \
        --num_samples 800000 \
        --2d \
        --visualize \
        --batch_size 500 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv1_zup.obj \
        --shape datasets/3dshape/4shape3d_zup.obj \
        --out   testing_data/3dshape/4shape3d_2denv1 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --2d \
        --batch_size 500 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/4shape3d_2denv1 \
      --modelPath ./Experiments/3dshape_2d --name 4shape3d_2denv1
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/4shape3d_2denv1 \
      --out ./results/output_3d/4shape3d_2denv1 \
      --checkpoint ./Experiments/3dshape_2d/4shape3d_2denv1/latest.pt \
      --2d
   ```

#### Tshape3d_2denv1

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv1_zup.obj \
        --shape datasets/3dshape/Tshape3d_zup.obj \
        --out   datasets/3dshape/Tshape3d_2denv1 \
        --num_samples 800000 \
        --2d \
        --visualize \
        --batch_size 500 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv1_zup.obj \
        --shape datasets/3dshape/Tshape3d_zup.obj \
        --out   testing_data/3dshape/Tshape3d_2denv1 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --2d \
        --batch_size 500 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Tshape3d_2denv1 \
      --modelPath ./Experiments/3dshape_2d --name Tshape3d_2denv1
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Tshape3d_2denv1 \
      --out ./results/output_3d/Tshape3d_2denv1 \
      --checkpoint ./Experiments/3dshape_2d/Tshape3d_2denv1/latest.pt \
      --2d
   ```



### 2denv2 (`2denv2_zup.obj`)

#### rectangle_2denv2

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv2_zup.obj \
        --shape datasets/3dshape/rectangle_zup.obj \
        --out   datasets/3dshape/rectangle_2denv2 \
        --num_samples 800000 \
        --2d \
        --visualize \
        --batch_size 500 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv2_zup.obj \
        --shape datasets/3dshape/rectangle_zup.obj \
        --out   testing_data/3dshape/rectangle_2denv2 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --2d \
        --batch_size 500 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/rectangle_2denv2 \
      --modelPath ./Experiments/3dshape_2d --name rectangle_2denv2
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/rectangle_2denv2 \
      --out ./results/output_3d/rectangle_2denv2 \
      --checkpoint ./Experiments/3dshape_2d/rectangle_2denv2/latest.pt \
      --2d
   ```

#### Lshape3d_2denv2

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv2_zup.obj \
        --shape datasets/3dshape/Lshape3d_zup.obj \
        --out   datasets/3dshape/Lshape3d_2denv2 \
        --num_samples 800000 \
        --2d \
        --visualize \
        --batch_size 500 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv2_zup.obj \
        --shape datasets/3dshape/Lshape3d_zup.obj \
        --out   testing_data/3dshape/Lshape3d_2denv2 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --2d \
        --batch_size 500 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Lshape3d_2denv2 \
      --modelPath ./Experiments/3dshape_2d --name Lshape3d_2denv2
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Lshape3d_2denv2 \
      --out ./results/output_3d/Lshape3d_2denv2 \
      --checkpoint ./Experiments/3dshape_2d/Lshape3d_2denv2/latest.pt \
      --2d
   ```

#### Fshape3d_2denv2

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv2_zup.obj \
        --shape datasets/3dshape/Fshape3d_zup.obj \
        --out   datasets/3dshape/Fshape3d_2denv2 \
        --num_samples 800000 \
        --2d \
        --visualize \
        --batch_size 500 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv2_zup.obj \
        --shape datasets/3dshape/Fshape3d_zup.obj \
        --out   testing_data/3dshape/Fshape3d_2denv2 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --2d \
        --batch_size 500 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Fshape3d_2denv2 \
      --modelPath ./Experiments/3dshape_2d --name Fshape3d_2denv2
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Fshape3d_2denv2 \
      --out ./results/output_3d/Fshape3d_2denv2 \
      --checkpoint ./Experiments/3dshape_2d/Fshape3d_2denv2/latest.pt \
      --2d
   ```

#### Ashape3d_2denv2

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv2_zup.obj \
        --shape datasets/3dshape/Ashape3d_zup.obj \
        --out   datasets/3dshape/Ashape3d_2denv2 \
        --num_samples 800000 \
        --2d \
        --visualize \
        --batch_size 500 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv2_zup.obj \
        --shape datasets/3dshape/Ashape3d_zup.obj \
        --out   testing_data/3dshape/Ashape3d_2denv2 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --2d \
        --batch_size 500 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Ashape3d_2denv2 \
      --modelPath ./Experiments/3dshape_2d --name Ashape3d_2denv2
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Ashape3d_2denv2 \
      --out ./results/output_3d/Ashape3d_2denv2 \
      --checkpoint ./Experiments/3dshape_2d/Ashape3d_2denv2/latest.pt \
      --2d
   ```

#### Vshape3d_2denv2

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv2_zup.obj \
        --shape datasets/3dshape/Vshape3d_zup.obj \
        --out   datasets/3dshape/Vshape3d_2denv2 \
        --num_samples 800000 \
        --2d \
        --visualize \
        --batch_size 500 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv2_zup.obj \
        --shape datasets/3dshape/Vshape3d_zup.obj \
        --out   testing_data/3dshape/Vshape3d_2denv2 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --2d \
        --batch_size 500 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Vshape3d_2denv2 \
      --modelPath ./Experiments/3dshape_2d --name Vshape3d_2denv2
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Vshape3d_2denv2 \
      --out ./results/output_3d/Vshape3d_2denv2 \
      --checkpoint ./Experiments/3dshape_2d/Vshape3d_2denv2/latest.pt \
      --2d
   ```

#### 4shape3d_2denv2

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv2_zup.obj \
        --shape datasets/3dshape/4shape3d_zup.obj \
        --out   datasets/3dshape/4shape3d_2denv2 \
        --num_samples 800000 \
        --2d \
        --visualize \
        --batch_size 500 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv2_zup.obj \
        --shape datasets/3dshape/4shape3d_zup.obj \
        --out   testing_data/3dshape/4shape3d_2denv2 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --2d \
        --batch_size 500 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/4shape3d_2denv2 \
      --modelPath ./Experiments/3dshape_2d --name 4shape3d_2denv2
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/4shape3d_2denv2 \
      --out ./results/output_3d/4shape3d_2denv2 \
      --checkpoint ./Experiments/3dshape_2d/4shape3d_2denv2/latest.pt \
      --2d
   ```

#### Tshape3d_2denv2

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv2_zup.obj \
        --shape datasets/3dshape/Tshape3d_zup.obj \
        --out   datasets/3dshape/Tshape3d_2denv2 \
        --num_samples 800000 \
        --2d \
        --visualize \
        --batch_size 500 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv2_zup.obj \
        --shape datasets/3dshape/Tshape3d_zup.obj \
        --out   testing_data/3dshape/Tshape3d_2denv2 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --2d \
        --batch_size 500 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Tshape3d_2denv2 \
      --modelPath ./Experiments/3dshape_2d --name Tshape3d_2denv2
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Tshape3d_2denv2 \
      --out ./results/output_3d/Tshape3d_2denv2 \
      --checkpoint ./Experiments/3dshape_2d/Tshape3d_2denv2/latest.pt \
      --2d
   ```



### 2denv3 (`2denv3_zup.obj`)

#### rectangle_2denv3

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv3_zup.obj \
        --shape datasets/3dshape/rectangle_zup.obj \
        --out   datasets/3dshape/rectangle_2denv3 \
        --num_samples 800000 \
        --2d \
        --visualize \
        --batch_size 1000 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv3_zup.obj \
        --shape datasets/3dshape/rectangle_zup.obj \
        --out   testing_data/3dshape/rectangle_2denv3 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --2d \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/rectangle_2denv3 \
      --modelPath ./Experiments/3dshape_2d --name rectangle_2denv3
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/rectangle_2denv3 \
      --out ./results/output_3d/rectangle_2denv3 \
      --checkpoint ./Experiments/3dshape_2d/rectangle_2denv3/latest.pt \
      --2d
   ```

#### Lshape3d_2denv3

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv3_zup.obj \
        --shape datasets/3dshape/Lshape3d_zup.obj \
        --out   datasets/3dshape/Lshape3d_2denv3 \
        --num_samples 800000 \
        --2d \
        --visualize \
        --batch_size 1000 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv3_zup.obj \
        --shape datasets/3dshape/Lshape3d_zup.obj \
        --out   testing_data/3dshape/Lshape3d_2denv3 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --2d \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Lshape3d_2denv3 \
      --modelPath ./Experiments/3dshape_2d --name Lshape3d_2denv3
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Lshape3d_2denv3 \
      --out ./results/output_3d/Lshape3d_2denv3 \
      --checkpoint ./Experiments/3dshape_2d/Lshape3d_2denv3/latest.pt \
      --2d
   ```

#### Fshape3d_2denv3

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv3_zup.obj \
        --shape datasets/3dshape/Fshape3d_zup.obj \
        --out   datasets/3dshape/Fshape3d_2denv3 \
        --num_samples 800000 \
        --2d \
        --visualize \
        --batch_size 1000 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv3_zup.obj \
        --shape datasets/3dshape/Fshape3d_zup.obj \
        --out   testing_data/3dshape/Fshape3d_2denv3 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --2d \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Fshape3d_2denv3 \
      --modelPath ./Experiments/3dshape_2d --name Fshape3d_2denv3
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Fshape3d_2denv3 \
      --out ./results/output_3d/Fshape3d_2denv3 \
      --checkpoint ./Experiments/3dshape_2d/Fshape3d_2denv3/latest.pt \
      --2d
   ```

#### Ashape3d_2denv3

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv3_zup.obj \
        --shape datasets/3dshape/Ashape3d_zup.obj \
        --out   datasets/3dshape/Ashape3d_2denv3 \
        --num_samples 800000 \
        --2d \
        --visualize \
        --batch_size 1000 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv3_zup.obj \
        --shape datasets/3dshape/Ashape3d_zup.obj \
        --out   testing_data/3dshape/Ashape3d_2denv3 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --2d \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Ashape3d_2denv3 \
      --modelPath ./Experiments/3dshape_2d --name Ashape3d_2denv3
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Ashape3d_2denv3 \
      --out ./results/output_3d/Ashape3d_2denv3 \
      --checkpoint ./Experiments/3dshape_2d/Ashape3d_2denv3/latest.pt \
      --2d
   ```

#### Vshape3d_2denv3

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv3_zup.obj \
        --shape datasets/3dshape/Vshape3d_zup.obj \
        --out   datasets/3dshape/Vshape3d_2denv3 \
        --num_samples 800000 \
        --2d \
        --visualize \
        --batch_size 1000 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv3_zup.obj \
        --shape datasets/3dshape/Vshape3d_zup.obj \
        --out   testing_data/3dshape/Vshape3d_2denv3 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --2d \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Vshape3d_2denv3 \
      --modelPath ./Experiments/3dshape_2d --name Vshape3d_2denv3
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Vshape3d_2denv3 \
      --out ./results/output_3d/Vshape3d_2denv3 \
      --checkpoint ./Experiments/3dshape_2d/Vshape3d_2denv3/latest.pt \
      --2d
   ```

#### 4shape3d_2denv3

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv3_zup.obj \
        --shape datasets/3dshape/4shape3d_zup.obj \
        --out   datasets/3dshape/4shape3d_2denv3 \
        --num_samples 800000 \
        --2d \
        --visualize \
        --batch_size 1000 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv3_zup.obj \
        --shape datasets/3dshape/4shape3d_zup.obj \
        --out   testing_data/3dshape/4shape3d_2denv3 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --2d \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/4shape3d_2denv3 \
      --modelPath ./Experiments/3dshape_2d --name 4shape3d_2denv3
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/4shape3d_2denv3 \
      --out ./results/output_3d/4shape3d_2denv3 \
      --checkpoint ./Experiments/3dshape_2d/4shape3d_2denv3/latest.pt \
      --2d
   ```

#### Tshape3d_2denv3

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv3_zup.obj \
        --shape datasets/3dshape/Tshape3d_zup.obj \
        --out   datasets/3dshape/Tshape3d_2denv3 \
        --num_samples 800000 \
        --2d \
        --visualize \
        --batch_size 1000 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv3_zup.obj \
        --shape datasets/3dshape/Tshape3d_zup.obj \
        --out   testing_data/3dshape/Tshape3d_2denv3 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --2d \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Tshape3d_2denv3 \
      --modelPath ./Experiments/3dshape_2d --name Tshape3d_2denv3
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Tshape3d_2denv3 \
      --out ./results/output_3d/Tshape3d_2denv3 \
      --checkpoint ./Experiments/3dshape_2d/Tshape3d_2denv3/latest.pt \
      --2d
   ```



### Recovered (June-3) pipeline

The same environments run through the frozen June-3 generator and network, so the
current pipeline can be measured against the one that produced
`pretrained/baseline_rectangle_env1.pt`.  `dataprocessing/preprocess_obj_june03.py`
carries the era's defaults (`--margin 0.1`, `--offset 0.01`, 400k samples, 8000 env
points) and `models/metric_june03` is a different architecture -- a single embedding
route rather than the current split translational/rotational one -- so its checkpoints
only load under `--models metric_june03`.

Only the *training* set is regenerated: evaluation reuses the test set built above, so
both pipelines are scored on exactly the same start/goal queries.

1. **Preprocess** (`<shape>` in `rectangle`, `Lshape3d`, `Fshape3d`, `Ashape3d`, `Vshape3d`, `4shape3d`, `Tshape3d`, `<env obj>` = `2denv4_zup.obj`,
   `2denv1_zup.obj`, `2denv2_zup.obj` or `2denv3_zup.obj`, `<env>` = `2denv4`,
   `2denv1`, `2denv2` or `2denv3`)
   ```
    python dataprocessing/preprocess_obj_june03.py \
        --env   datasets/3dshape/<env obj> \
        --shape datasets/3dshape/<shape>_zup.obj \
        --out   datasets/3dshape/<shape>_<env>_june03 \
        --num_samples 400000 \
        --margin 0.1 \
        --offset 0.01 \
        --2d \
        --visualize \
        --batch_size 256 \
        --device cuda:2
   ```
2. **Train**:
   ```
   python train/train_3dshape_june03.py \
      --dataPath ./datasets/3dshape/<shape>_<env>_june03 \
      --modelPath ./Experiments/3dshape_2d_june03 --name <shape>_<env>
   ```
3. **Eval**:
   ```
   python evaluate_training_3d_batched.py \
      --dataPath testing_data/3dshape/<shape>_<env> \
      --out ./results/output_3d/<shape>_<env>_june03 \
      --models metric_june03 \
      --checkpoint ./Experiments/3dshape_2d_june03/<shape>_<env>/latest.pt \
      --2d
   ```

Worked example for `rectangle_2denv4`:

```
python dataprocessing/preprocess_obj_june03.py \
    --env   datasets/3dshape/2denv4_zup.obj \
    --shape datasets/3dshape/rectangle_zup.obj \
    --out   datasets/3dshape/rectangle_2denv4_june03 \
    --num_samples 400000 --margin 0.1 --offset 0.01 --2d --visualize \
    --batch_size 256 --device cuda:2
python train/train_3dshape_june03.py \
    --dataPath ./datasets/3dshape/rectangle_2denv4_june03 \
    --modelPath ./Experiments/3dshape_2d_june03 --name rectangle_2denv4
python evaluate_training_3d_batched.py \
    --dataPath testing_data/3dshape/rectangle_2denv4 \
    --out ./results/output_3d/rectangle_2denv4_june03 \
    --models metric_june03 \
    --checkpoint ./Experiments/3dshape_2d_june03/rectangle_2denv4/latest.pt --2d
```

### Running the whole sweep

Preprocessing is driven one environment at a time by
`_run_2denv4_preprocess.sh`, `_run_2denv1_preprocess.sh` and
`_run_2denv2_preprocess.sh` -- current pipeline only, all seven shapes, one job
per GPU round-robin over the three GPUs.  `SHAPES` runs a subset (used to resume
after an interrupted run); `_run_2denv2_preprocess.sh` also takes `GPU_LIST` and
`NWORKERS` to pin a sweep to a subset of the cards, e.g. one shape at a time on a
single free GPU while another sweep holds the other two:

```
bash _run_2denv2_preprocess.sh                                # all 7 shapes, 3 GPUs
SHAPES="Tshape3d rectangle" bash _run_2denv1_preprocess.sh    # just those two
GPU_LIST="cuda:2" NWORKERS=1 bash _run_2denv2_preprocess.sh   # serial, cuda:2 only
```

`_run_2d_preprocess.sh` is the older combined driver that also fans out the
June-3 pipeline; it still `cd`s one level below the repo root and maps `2denv1`
to a mesh name that no longer exists, so prefer the per-env scripts above.
`_run_2d_train.sh` and `_run_2d_eval.sh` drive training and evaluation; results
are tabulated in `../../experiments_ours.md`, per-cell generation times in
`../../2d_gen_times.md`, and the overall status board is
`../../MASTER_EXPERIMENTS_README.md`.

```
bash _run_2d_train.sh        # 5000 epochs each
bash _run_2d_eval.sh         # evaluations on the shared test sets
```

### Legacy dataset

`datasets/3dshape/Tshape3d_env4` is the original 2-D dataset for the
(`Tshape3d`, `2denv4`) pair, generated with these same settings before the env-tag
naming above existed.  `Tshape3d_2denv4` supersedes it; the old row is kept in
`../../experiments_ours.md` for continuity.
