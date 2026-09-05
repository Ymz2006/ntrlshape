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

### Tshape_env4

1. **Preprocess** 
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv4_zup.obj \
        --shape datasets/3dshape/Tshape3d_zup.obj \
        --out   datasets/3dshape/Tshape3d_env4 \
        --num_samples 800000 \
        --2d \
        --visualize \
        --batch_size 2000 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/2denv4_zup.obj \
        --shape datasets/3dshape/Tshape3d_zup.obj \
        --out   testing_data/3dshape/Tshape3d_env4 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --2d \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Tshape3d_env4
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Tshape3d_env4 \
      --out ./results/output_3d/Tshape3d_env4 \
      --2d
   ```
