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

Seven shapes (`rectangle`, `Lshape3d`, `Fshape3d`, `Ashape3d`, `Vshape3d`, `4shape3d`,
`Tshape3d`) across `env1..env4`.  The `Tshape3d_env*` cells were added on 2026-09-10; they
are driven end to end by `_run_3d_tshape.sh` (see "Running the 3-D T-shape cells" below)
and their commands are written out per env like every other cell.

**Epochs.**  Every 3-D row in `../../experiments_ours.md` was trained for **10000 epochs**,
the trainer default at the time those runs were made.  The default is now 5000
(`models/metric/model_train_metric.py`), so a new 3-D cell that should be comparable to
the table passes `--epochs 10000` explicitly, as the `Tshape3d_env*` entries do.  The
`--modelPath ./Experiments/3dshape --name <cell>` pair pins the run folder to
`Experiments/3dshape/<cell>/` instead of the `3dshape_<timestamp>` name the older rows
carry; either form is loaded the same way by `--checkpoint`.

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

#### Tshape3d_env1

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env1.obj \
        --shape datasets/3dshape/Tshape3d.obj \
        --out   datasets/3dshape/Tshape3d_env1 \
        --num_samples 800000 \
        --visualize \
        --batch_size 500 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env1.obj \
        --shape datasets/3dshape/Tshape3d.obj \
        --out   testing_data/3dshape/Tshape3d_env1 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 500 \
        --visualize
   ```
2. **Train** (10000 epochs, the setting of every other 3-D row -- see "Epochs" above):
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Tshape3d_env1 \
      --modelPath ./Experiments/3dshape --name Tshape3d_env1 --epochs 10000
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Tshape3d_env1 \
      --out ./results/output_3d/Tshape3d_env1 \
      --checkpoint ./Experiments/3dshape/Tshape3d_env1/latest.pt
   ```

   An earlier `Tshape3d_env1` (dataset, test set at `--offset 0.01`, 100-case result,
   built 2026-08-18/19 before the cell was in this README) is kept as
   `*/Tshape3d_env1_aug18`; the entry above regenerated everything at the settings
   shared by the other 3-D cells.


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

#### Tshape3d_env2

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env2.obj \
        --shape datasets/3dshape/Tshape3d.obj \
        --out   datasets/3dshape/Tshape3d_env2 \
        --num_samples 800000 \
        --visualize \
        --batch_size 500 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env2.obj \
        --shape datasets/3dshape/Tshape3d.obj \
        --out   testing_data/3dshape/Tshape3d_env2 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 500 \
        --visualize
   ```
2. **Train** (10000 epochs, the setting of every other 3-D row -- see "Epochs" above):
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Tshape3d_env2 \
      --modelPath ./Experiments/3dshape --name Tshape3d_env2 --epochs 10000
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Tshape3d_env2 \
      --out ./results/output_3d/Tshape3d_env2 \
      --checkpoint ./Experiments/3dshape/Tshape3d_env2/latest.pt
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

#### Tshape3d_env3

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env3.obj \
        --shape datasets/3dshape/Tshape3d.obj \
        --out   datasets/3dshape/Tshape3d_env3 \
        --num_samples 800000 \
        --visualize \
        --batch_size 500 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env3.obj \
        --shape datasets/3dshape/Tshape3d.obj \
        --out   testing_data/3dshape/Tshape3d_env3 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 500 \
        --visualize
   ```
2. **Train** (10000 epochs, the setting of every other 3-D row -- see "Epochs" above):
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Tshape3d_env3 \
      --modelPath ./Experiments/3dshape --name Tshape3d_env3 --epochs 10000
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Tshape3d_env3 \
      --out ./results/output_3d/Tshape3d_env3 \
      --checkpoint ./Experiments/3dshape/Tshape3d_env3/latest.pt
   ```

   **`Tshape3d_env3/latest.pt` is the epoch-5500 model, not 10000.**  It was evaluated
   mid-run on 2026-09-12 and already sat inside the band of the finished env3 rows, so
   the run was stopped at epoch 7500 and `Model_Epoch_05500_*.pt` was installed as
   `latest.pt`; the eval / SD results under the regular `results/output_3d/Tshape3d_env3`
   and `results/spread_sd/Tshape3d_env3` are that model's (their `checkpoint:` line names
   `Tshape3d_env3_early/latest.pt`, the byte-identical frozen copy the eval ran from).
   The folder keeps the per-500-epoch checkpoints up to 7500.


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

#### Tshape3d_env4

Not to be confused with the legacy planar dataset that used to sit under this name --
see "Legacy dataset" at the end of the file.  This is the SE(3) cell: `Tshape3d.obj`
(not `_zup`) against `env4.obj`, no `--2d`.

1. **Preprocess**
   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env4.obj \
        --shape datasets/3dshape/Tshape3d.obj \
        --out   datasets/3dshape/Tshape3d_env4 \
        --num_samples 800000 \
        --visualize \
        --batch_size 500 \
        --device cuda:2
   ```

   ```
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env4.obj \
        --shape datasets/3dshape/Tshape3d.obj \
        --out   testing_data/3dshape/Tshape3d_env4 \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 500 \
        --visualize
   ```
2. **Train** (10000 epochs, the setting of every other 3-D row -- see "Epochs" above):
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/Tshape3d_env4 \
      --modelPath ./Experiments/3dshape --name Tshape3d_env4 --epochs 10000
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Tshape3d_env4 \
      --out ./results/output_3d/Tshape3d_env4 \
      --checkpoint ./Experiments/3dshape/Tshape3d_env4/latest.pt
   ```

### Running the 3-D T-shape cells

`_run_3d_tshape.sh` runs the four `Tshape3d_env*` entries above exactly as written --
preprocess (training set + test set), train, evaluate, then the SD planner
(`_spread_planner.py --steer-bias 0.5 --no-gate`, the `SR Alternate Bellman Horizon SD`
column) -- one env per worker, round-robin over `SLOTS`.  Each stage skips a cell whose
output already exists, so it resumes after an interruption.  Logs go to
`.preplogs_3d_tshape/`, `Experiments/3dshape/logs/` and `.evallogs_3d_tshape/`.

```
bash _run_3d_tshape.sh                                  # all four envs, all stages
STAGES="train evaluate spread" bash _run_3d_tshape.sh   # data already generated
ENVS="env2" SLOTS="cuda:1" bash _run_3d_tshape.sh       # one cell on one card
BS=250 bash _run_3d_tshape.sh                           # smaller sampler chunks
```

The T entries above are written at `--batch_size 500`, not the `2000` the other 3-D
cells use: in 3-D the T shape peaks near 10 GiB at 500 and needs more than 18 GiB at
2000, which OOMs a 24 GiB card that is shared with anything else.  As with the 2-D
sweeps, batch size only chunks the rejection sampler -- it never changes the sampled
distribution -- so `BS` is free to retune per machine.  The 2026-09-11 sweep that
produced the table rows ran at `BS=250`, four workers, sharing GPUs 1 and 2 with the
`_run_3d_eval_1k.sh` sweep.

Run it from this directory inside the `pytorchserver` container.  After the sweep,
`python _make_experiments_3d.py` rewrites the 3-D table of `../../experiments_ours.md`
from `results/output_3d/` and `results/spread_sd/` (SD column included).  Do NOT follow it
with `_make_spread_sd.py`: that script rewrites the SD column of *every* table in the
file from `results/spread_sd/`, which clobbers the `3D_1k_valid` table (its SD values
come from `results/spread_sd_1k/`); re-run `_make_3d_1k_table.py` if that happens.

### Scoring the 3-D cells on the tight test sets

`_run_3d_eval_tight.sh` is `_run_3d_eval_1k.sh` with `--dataPath` pointed at
`testing_data/3dshape/<cell>_tight` -- the sets `dataprocessing/make_tight_testing_data.py`
samples at `--offset 0.005` instead of `0.02`, so start/goal poses may sit four times
closer to the obstacles.  Same 28 checkpoints, same two stages (`evaluate_training_3d_batched.py`
then `_spread_planner.py --steer-bias 0.5 --no-gate`), same 1000 cases; outputs go to
`results/output_3d_tight/` and `results/spread_sd_tight/`, logs to `.evallogs_3d_tight/`.
A `gen` stage runs first and preprocesses any cell whose `_tight` set is missing
(`--batch_size $BS`, default 500 -- the T-shape limit above); on 2026-09-12 that was
`Tshape3d_env2..env4`, which postdate the tight sweep.  Every stage skips a cell whose
output exists.

```
bash _run_3d_eval_tight.sh                                   # gen, main, spread
STAGES="main spread" bash _run_3d_eval_tight.sh              # tight sets already on disk
STAGES=gen SLOT_LIST="cuda:1 cuda:2" bash _run_3d_eval_tight.sh   # one sampler per card
```

Afterwards `python _make_3d_tight_table.py` rewrites the `3D_tight` section of
`../../experiments_ours.md` (between `3D_1k_valid` and the 2-D table).  The same
`_make_spread_sd.py` caveat applies: it would overwrite this table's SD column from
`results/spread_sd/`.


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

#### Scoring Lcouch_Corozal on an RRT-verified set (the "Gibson" table)

The `testing_data/` set above is collision-free only; the tabulated numbers
(`../../experiments_ours.md`, "Gibson (Lcouch_Corozal)") are scored on
`testing_data_1k_complete/3dshape/Lcouch_Corozal`, built with the same
RRT-Connect-verified generator as the 1k sets but with **500 pairs** (the
Corozal scan is 85k triangles, so each RRT call is slow).  The generator needs
OMPL, so it runs in `ntrl_mpnet`; the evaluation runs in the pytorchserver
container (`ntrl_baselines_train`), with `main` on one card and `spread` /
`er_opt` on another.

```
# in ntrl_mpnet, cwd = ntrl-demo/ntrl-demo
python dataprocessing/generate_testing_data_rrt.py \
    --shape datasets/3dshape/Lcouch.obj --env datasets/3dshape/Corozal.obj \
    --out testing_data_1k_complete/3dshape/Lcouch_Corozal \
    --offset 0.02 --num_samples 500 --rrt_time 180 --workers 20 \
    --batch_size 500 --device cuda:0

# in ntrl_baselines_train, cwd = ntrl-demo/ntrl-demo
DEV_MAIN=cuda:0 DEV_SD=cuda:2 DEV_ER=cuda:2 bash _run_gibson_eval_1k.sh   # CASES=500
python _make_gibson_table.py                                          # -> experiments_ours.md
```

The checkpoint is `Experiments/3dshape/3dshape_08_31_13_53/latest.pt` (10000
epochs, trained 2026-08-31; `3dshape_08_31_14_01` is a duplicate run of the
same dataset).  Outputs land in `results/output_3d_1k/`, `results/spread_sd_1k/`
and `results/er_opt_1k/` next to the SE(3) 1k cells, logs in `.evallogs_3d_1k/`.


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



### Metric NTFields (recovered June-3) pipeline

The same four planar mazes and seven shapes run through the frozen June-3 network,
so the current pipeline can be measured against the one that produced
`pretrained/baseline_rectangle_env1.pt`.  `models/metric_june03` is a different
architecture -- a single embedding route rather than the current split
translational/rotational one, and a Fourier `B` initialized at std 0.86 rather than
0.2 -- so its checkpoints only load under `--models metric_june03`.

**These runs train on the shared current-pipeline datasets**, the same
`datasets/3dshape/<shape>_<env>` dirs that `models/metric` and the NTFields baseline
consume, rather than on regenerated `_june03` data.  That is possible because
`models/metric_june03/data_mlp.py` loads only `sampled_points.npy`, `speed.npy` and
`normal.npy` -- the three arrays every dataset here has -- and its training loop
unpacks `points[:2*dim] | speed[2] | normal[2*dim]`, which is exactly the leading
26 columns of those files.  The extra `speed_dists` / `speed_angles` / `trans_n` /
`rot_n` arrays the current preprocessor also writes are simply never read.

Sharing the data is deliberate: with the generator held fixed, a June-3 vs current
difference isolates the **network and trainer** (embedding route, `B` scale, 5000
epochs) instead of confounding it with the generator's `--margin` / `--offset` /
sample-count changes.  It also matches how the NTFields baseline is set up, so all
three pipelines in `../../experiments_ours.md` and `../../experiments_ntfields.md`
see identical training pairs and identical test queries.

`dataprocessing/preprocess_obj_june03.py` still exists and still carries the era's
own defaults (`--margin 0.1`, `--offset 0.01`, 400k samples, 8000 env points).  Run
it only if you specifically want the generator ablation as well; nothing below
needs it, and no 2-D `_june03` dataset is generated.

Checkpoints land in `./Experiments/3dshape_2d_june03/<shape>_<env>`.  Evaluation
reuses the shared test sets, so both pipelines are scored on exactly the same
start/goal queries.

All 28 cells are driven by `_run_2d_june03_train.sh` -- see
`### Running the whole sweep`.

#### June-3 2denv1 (`2denv1_zup.obj`)

##### rectangle_2denv1_june03

1. **Train** (no preprocess step -- reuses the shared dataset):
   ```
   python train/train_3dshape_june03.py \
      --dataPath ./datasets/3dshape/rectangle_2denv1 \
      --modelPath ./Experiments/3dshape_2d_june03 --name rectangle_2denv1 \
      --epochs 5000 \
      --device cuda:2
   ```

2. **Eval**:
   ```
   python evaluate_training_3d_batched.py \
      --dataPath testing_data/3dshape/rectangle_2denv1 \
      --out ./results/output_3d/rectangle_2denv1_june03 \
      --models metric_june03 \
      --checkpoint ./Experiments/3dshape_2d_june03/rectangle_2denv1/latest.pt \
      --2d
   ```

##### Lshape3d_2denv1_june03

1. **Train** (no preprocess step -- reuses the shared dataset):
   ```
   python train/train_3dshape_june03.py \
      --dataPath ./datasets/3dshape/Lshape3d_2denv1 \
      --modelPath ./Experiments/3dshape_2d_june03 --name Lshape3d_2denv1 \
      --epochs 5000 \
      --device cuda:2
   ```

2. **Eval**:
   ```
   python evaluate_training_3d_batched.py \
      --dataPath testing_data/3dshape/Lshape3d_2denv1 \
      --out ./results/output_3d/Lshape3d_2denv1_june03 \
      --models metric_june03 \
      --checkpoint ./Experiments/3dshape_2d_june03/Lshape3d_2denv1/latest.pt \
      --2d
   ```

##### Fshape3d_2denv1_june03

1. **Train** (no preprocess step -- reuses the shared dataset):
   ```
   python train/train_3dshape_june03.py \
      --dataPath ./datasets/3dshape/Fshape3d_2denv1 \
      --modelPath ./Experiments/3dshape_2d_june03 --name Fshape3d_2denv1 \
      --epochs 5000 \
      --device cuda:2
   ```

2. **Eval**:
   ```
   python evaluate_training_3d_batched.py \
      --dataPath testing_data/3dshape/Fshape3d_2denv1 \
      --out ./results/output_3d/Fshape3d_2denv1_june03 \
      --models metric_june03 \
      --checkpoint ./Experiments/3dshape_2d_june03/Fshape3d_2denv1/latest.pt \
      --2d
   ```

##### Ashape3d_2denv1_june03

1. **Train** (no preprocess step -- reuses the shared dataset):
   ```
   python train/train_3dshape_june03.py \
      --dataPath ./datasets/3dshape/Ashape3d_2denv1 \
      --modelPath ./Experiments/3dshape_2d_june03 --name Ashape3d_2denv1 \
      --epochs 5000 \
      --device cuda:2
   ```

2. **Eval**:
   ```
   python evaluate_training_3d_batched.py \
      --dataPath testing_data/3dshape/Ashape3d_2denv1 \
      --out ./results/output_3d/Ashape3d_2denv1_june03 \
      --models metric_june03 \
      --checkpoint ./Experiments/3dshape_2d_june03/Ashape3d_2denv1/latest.pt \
      --2d
   ```

##### Vshape3d_2denv1_june03

1. **Train** (no preprocess step -- reuses the shared dataset):
   ```
   python train/train_3dshape_june03.py \
      --dataPath ./datasets/3dshape/Vshape3d_2denv1 \
      --modelPath ./Experiments/3dshape_2d_june03 --name Vshape3d_2denv1 \
      --epochs 5000 \
      --device cuda:2
   ```

2. **Eval**:
   ```
   python evaluate_training_3d_batched.py \
      --dataPath testing_data/3dshape/Vshape3d_2denv1 \
      --out ./results/output_3d/Vshape3d_2denv1_june03 \
      --models metric_june03 \
      --checkpoint ./Experiments/3dshape_2d_june03/Vshape3d_2denv1/latest.pt \
      --2d
   ```

##### 4shape3d_2denv1_june03

1. **Train** (no preprocess step -- reuses the shared dataset):
   ```
   python train/train_3dshape_june03.py \
      --dataPath ./datasets/3dshape/4shape3d_2denv1 \
      --modelPath ./Experiments/3dshape_2d_june03 --name 4shape3d_2denv1 \
      --epochs 5000 \
      --device cuda:2
   ```

2. **Eval**:
   ```
   python evaluate_training_3d_batched.py \
      --dataPath testing_data/3dshape/4shape3d_2denv1 \
      --out ./results/output_3d/4shape3d_2denv1_june03 \
      --models metric_june03 \
      --checkpoint ./Experiments/3dshape_2d_june03/4shape3d_2denv1/latest.pt \
      --2d
   ```

##### Tshape3d_2denv1_june03

1. **Train** (no preprocess step -- reuses the shared dataset):
   ```
   python train/train_3dshape_june03.py \
      --dataPath ./datasets/3dshape/Tshape3d_2denv1 \
      --modelPath ./Experiments/3dshape_2d_june03 --name Tshape3d_2denv1 \
      --epochs 5000 \
      --device cuda:2
   ```

2. **Eval**:
   ```
   python evaluate_training_3d_batched.py \
      --dataPath testing_data/3dshape/Tshape3d_2denv1 \
      --out ./results/output_3d/Tshape3d_2denv1_june03 \
      --models metric_june03 \
      --checkpoint ./Experiments/3dshape_2d_june03/Tshape3d_2denv1/latest.pt \
      --2d
   ```

#### June-3 2denv2 (`2denv2_zup.obj`)

##### rectangle_2denv2_june03

1. **Train** (no preprocess step -- reuses the shared dataset):
   ```
   python train/train_3dshape_june03.py \
      --dataPath ./datasets/3dshape/rectangle_2denv2 \
      --modelPath ./Experiments/3dshape_2d_june03 --name rectangle_2denv2 \
      --epochs 5000 \
      --device cuda:2
   ```

2. **Eval**:
   ```
   python evaluate_training_3d_batched.py \
      --dataPath testing_data/3dshape/rectangle_2denv2 \
      --out ./results/output_3d/rectangle_2denv2_june03 \
      --models metric_june03 \
      --checkpoint ./Experiments/3dshape_2d_june03/rectangle_2denv2/latest.pt \
      --2d
   ```

##### Lshape3d_2denv2_june03

1. **Train** (no preprocess step -- reuses the shared dataset):
   ```
   python train/train_3dshape_june03.py \
      --dataPath ./datasets/3dshape/Lshape3d_2denv2 \
      --modelPath ./Experiments/3dshape_2d_june03 --name Lshape3d_2denv2 \
      --epochs 5000 \
      --device cuda:2
   ```

2. **Eval**:
   ```
   python evaluate_training_3d_batched.py \
      --dataPath testing_data/3dshape/Lshape3d_2denv2 \
      --out ./results/output_3d/Lshape3d_2denv2_june03 \
      --models metric_june03 \
      --checkpoint ./Experiments/3dshape_2d_june03/Lshape3d_2denv2/latest.pt \
      --2d
   ```

##### Fshape3d_2denv2_june03

1. **Train** (no preprocess step -- reuses the shared dataset):
   ```
   python train/train_3dshape_june03.py \
      --dataPath ./datasets/3dshape/Fshape3d_2denv2 \
      --modelPath ./Experiments/3dshape_2d_june03 --name Fshape3d_2denv2 \
      --epochs 5000 \
      --device cuda:2
   ```

2. **Eval**:
   ```
   python evaluate_training_3d_batched.py \
      --dataPath testing_data/3dshape/Fshape3d_2denv2 \
      --out ./results/output_3d/Fshape3d_2denv2_june03 \
      --models metric_june03 \
      --checkpoint ./Experiments/3dshape_2d_june03/Fshape3d_2denv2/latest.pt \
      --2d
   ```

##### Ashape3d_2denv2_june03

1. **Train** (no preprocess step -- reuses the shared dataset):
   ```
   python train/train_3dshape_june03.py \
      --dataPath ./datasets/3dshape/Ashape3d_2denv2 \
      --modelPath ./Experiments/3dshape_2d_june03 --name Ashape3d_2denv2 \
      --epochs 5000 \
      --device cuda:2
   ```

2. **Eval**:
   ```
   python evaluate_training_3d_batched.py \
      --dataPath testing_data/3dshape/Ashape3d_2denv2 \
      --out ./results/output_3d/Ashape3d_2denv2_june03 \
      --models metric_june03 \
      --checkpoint ./Experiments/3dshape_2d_june03/Ashape3d_2denv2/latest.pt \
      --2d
   ```

##### Vshape3d_2denv2_june03

1. **Train** (no preprocess step -- reuses the shared dataset):
   ```
   python train/train_3dshape_june03.py \
      --dataPath ./datasets/3dshape/Vshape3d_2denv2 \
      --modelPath ./Experiments/3dshape_2d_june03 --name Vshape3d_2denv2 \
      --epochs 5000 \
      --device cuda:2
   ```

2. **Eval**:
   ```
   python evaluate_training_3d_batched.py \
      --dataPath testing_data/3dshape/Vshape3d_2denv2 \
      --out ./results/output_3d/Vshape3d_2denv2_june03 \
      --models metric_june03 \
      --checkpoint ./Experiments/3dshape_2d_june03/Vshape3d_2denv2/latest.pt \
      --2d
   ```

##### 4shape3d_2denv2_june03

1. **Train** (no preprocess step -- reuses the shared dataset):
   ```
   python train/train_3dshape_june03.py \
      --dataPath ./datasets/3dshape/4shape3d_2denv2 \
      --modelPath ./Experiments/3dshape_2d_june03 --name 4shape3d_2denv2 \
      --epochs 5000 \
      --device cuda:2
   ```

2. **Eval**:
   ```
   python evaluate_training_3d_batched.py \
      --dataPath testing_data/3dshape/4shape3d_2denv2 \
      --out ./results/output_3d/4shape3d_2denv2_june03 \
      --models metric_june03 \
      --checkpoint ./Experiments/3dshape_2d_june03/4shape3d_2denv2/latest.pt \
      --2d
   ```

##### Tshape3d_2denv2_june03

1. **Train** (no preprocess step -- reuses the shared dataset):
   ```
   python train/train_3dshape_june03.py \
      --dataPath ./datasets/3dshape/Tshape3d_2denv2 \
      --modelPath ./Experiments/3dshape_2d_june03 --name Tshape3d_2denv2 \
      --epochs 5000 \
      --device cuda:2
   ```

2. **Eval**:
   ```
   python evaluate_training_3d_batched.py \
      --dataPath testing_data/3dshape/Tshape3d_2denv2 \
      --out ./results/output_3d/Tshape3d_2denv2_june03 \
      --models metric_june03 \
      --checkpoint ./Experiments/3dshape_2d_june03/Tshape3d_2denv2/latest.pt \
      --2d
   ```

#### June-3 2denv3 (`2denv3_zup.obj`)

##### rectangle_2denv3_june03

1. **Train** (no preprocess step -- reuses the shared dataset):
   ```
   python train/train_3dshape_june03.py \
      --dataPath ./datasets/3dshape/rectangle_2denv3 \
      --modelPath ./Experiments/3dshape_2d_june03 --name rectangle_2denv3 \
      --epochs 5000 \
      --device cuda:2
   ```

2. **Eval**:
   ```
   python evaluate_training_3d_batched.py \
      --dataPath testing_data/3dshape/rectangle_2denv3 \
      --out ./results/output_3d/rectangle_2denv3_june03 \
      --models metric_june03 \
      --checkpoint ./Experiments/3dshape_2d_june03/rectangle_2denv3/latest.pt \
      --2d
   ```

##### Lshape3d_2denv3_june03

1. **Train** (no preprocess step -- reuses the shared dataset):
   ```
   python train/train_3dshape_june03.py \
      --dataPath ./datasets/3dshape/Lshape3d_2denv3 \
      --modelPath ./Experiments/3dshape_2d_june03 --name Lshape3d_2denv3 \
      --epochs 5000 \
      --device cuda:2
   ```

2. **Eval**:
   ```
   python evaluate_training_3d_batched.py \
      --dataPath testing_data/3dshape/Lshape3d_2denv3 \
      --out ./results/output_3d/Lshape3d_2denv3_june03 \
      --models metric_june03 \
      --checkpoint ./Experiments/3dshape_2d_june03/Lshape3d_2denv3/latest.pt \
      --2d
   ```

##### Fshape3d_2denv3_june03

1. **Train** (no preprocess step -- reuses the shared dataset):
   ```
   python train/train_3dshape_june03.py \
      --dataPath ./datasets/3dshape/Fshape3d_2denv3 \
      --modelPath ./Experiments/3dshape_2d_june03 --name Fshape3d_2denv3 \
      --epochs 5000 \
      --device cuda:2
   ```

2. **Eval**:
   ```
   python evaluate_training_3d_batched.py \
      --dataPath testing_data/3dshape/Fshape3d_2denv3 \
      --out ./results/output_3d/Fshape3d_2denv3_june03 \
      --models metric_june03 \
      --checkpoint ./Experiments/3dshape_2d_june03/Fshape3d_2denv3/latest.pt \
      --2d
   ```

##### Ashape3d_2denv3_june03

1. **Train** (no preprocess step -- reuses the shared dataset):
   ```
   python train/train_3dshape_june03.py \
      --dataPath ./datasets/3dshape/Ashape3d_2denv3 \
      --modelPath ./Experiments/3dshape_2d_june03 --name Ashape3d_2denv3 \
      --epochs 5000 \
      --device cuda:2
   ```

2. **Eval**:
   ```
   python evaluate_training_3d_batched.py \
      --dataPath testing_data/3dshape/Ashape3d_2denv3 \
      --out ./results/output_3d/Ashape3d_2denv3_june03 \
      --models metric_june03 \
      --checkpoint ./Experiments/3dshape_2d_june03/Ashape3d_2denv3/latest.pt \
      --2d
   ```

##### Vshape3d_2denv3_june03

1. **Train** (no preprocess step -- reuses the shared dataset):
   ```
   python train/train_3dshape_june03.py \
      --dataPath ./datasets/3dshape/Vshape3d_2denv3 \
      --modelPath ./Experiments/3dshape_2d_june03 --name Vshape3d_2denv3 \
      --epochs 5000 \
      --device cuda:2
   ```

2. **Eval**:
   ```
   python evaluate_training_3d_batched.py \
      --dataPath testing_data/3dshape/Vshape3d_2denv3 \
      --out ./results/output_3d/Vshape3d_2denv3_june03 \
      --models metric_june03 \
      --checkpoint ./Experiments/3dshape_2d_june03/Vshape3d_2denv3/latest.pt \
      --2d
   ```

##### 4shape3d_2denv3_june03

1. **Train** (no preprocess step -- reuses the shared dataset):
   ```
   python train/train_3dshape_june03.py \
      --dataPath ./datasets/3dshape/4shape3d_2denv3 \
      --modelPath ./Experiments/3dshape_2d_june03 --name 4shape3d_2denv3 \
      --epochs 5000 \
      --device cuda:2
   ```

2. **Eval**:
   ```
   python evaluate_training_3d_batched.py \
      --dataPath testing_data/3dshape/4shape3d_2denv3 \
      --out ./results/output_3d/4shape3d_2denv3_june03 \
      --models metric_june03 \
      --checkpoint ./Experiments/3dshape_2d_june03/4shape3d_2denv3/latest.pt \
      --2d
   ```

##### Tshape3d_2denv3_june03

1. **Train** (no preprocess step -- reuses the shared dataset):
   ```
   python train/train_3dshape_june03.py \
      --dataPath ./datasets/3dshape/Tshape3d_2denv3 \
      --modelPath ./Experiments/3dshape_2d_june03 --name Tshape3d_2denv3 \
      --epochs 5000 \
      --device cuda:2
   ```

2. **Eval**:
   ```
   python evaluate_training_3d_batched.py \
      --dataPath testing_data/3dshape/Tshape3d_2denv3 \
      --out ./results/output_3d/Tshape3d_2denv3_june03 \
      --models metric_june03 \
      --checkpoint ./Experiments/3dshape_2d_june03/Tshape3d_2denv3/latest.pt \
      --2d
   ```

#### June-3 2denv4 (`2denv4_zup.obj`)

##### rectangle_2denv4_june03

1. **Train** (no preprocess step -- reuses the shared dataset):
   ```
   python train/train_3dshape_june03.py \
      --dataPath ./datasets/3dshape/rectangle_2denv4 \
      --modelPath ./Experiments/3dshape_2d_june03 --name rectangle_2denv4 \
      --epochs 5000 \
      --device cuda:2
   ```

2. **Eval**:
   ```
   python evaluate_training_3d_batched.py \
      --dataPath testing_data/3dshape/rectangle_2denv4 \
      --out ./results/output_3d/rectangle_2denv4_june03 \
      --models metric_june03 \
      --checkpoint ./Experiments/3dshape_2d_june03/rectangle_2denv4/latest.pt \
      --2d
   ```

##### Lshape3d_2denv4_june03

1. **Train** (no preprocess step -- reuses the shared dataset):
   ```
   python train/train_3dshape_june03.py \
      --dataPath ./datasets/3dshape/Lshape3d_2denv4 \
      --modelPath ./Experiments/3dshape_2d_june03 --name Lshape3d_2denv4 \
      --epochs 5000 \
      --device cuda:2
   ```

2. **Eval**:
   ```
   python evaluate_training_3d_batched.py \
      --dataPath testing_data/3dshape/Lshape3d_2denv4 \
      --out ./results/output_3d/Lshape3d_2denv4_june03 \
      --models metric_june03 \
      --checkpoint ./Experiments/3dshape_2d_june03/Lshape3d_2denv4/latest.pt \
      --2d
   ```

##### Fshape3d_2denv4_june03

1. **Train** (no preprocess step -- reuses the shared dataset):
   ```
   python train/train_3dshape_june03.py \
      --dataPath ./datasets/3dshape/Fshape3d_2denv4 \
      --modelPath ./Experiments/3dshape_2d_june03 --name Fshape3d_2denv4 \
      --epochs 5000 \
      --device cuda:2
   ```

2. **Eval**:
   ```
   python evaluate_training_3d_batched.py \
      --dataPath testing_data/3dshape/Fshape3d_2denv4 \
      --out ./results/output_3d/Fshape3d_2denv4_june03 \
      --models metric_june03 \
      --checkpoint ./Experiments/3dshape_2d_june03/Fshape3d_2denv4/latest.pt \
      --2d
   ```

##### Ashape3d_2denv4_june03

1. **Train** (no preprocess step -- reuses the shared dataset):
   ```
   python train/train_3dshape_june03.py \
      --dataPath ./datasets/3dshape/Ashape3d_2denv4 \
      --modelPath ./Experiments/3dshape_2d_june03 --name Ashape3d_2denv4 \
      --epochs 5000 \
      --device cuda:2
   ```

2. **Eval**:
   ```
   python evaluate_training_3d_batched.py \
      --dataPath testing_data/3dshape/Ashape3d_2denv4 \
      --out ./results/output_3d/Ashape3d_2denv4_june03 \
      --models metric_june03 \
      --checkpoint ./Experiments/3dshape_2d_june03/Ashape3d_2denv4/latest.pt \
      --2d
   ```

##### Vshape3d_2denv4_june03

1. **Train** (no preprocess step -- reuses the shared dataset):
   ```
   python train/train_3dshape_june03.py \
      --dataPath ./datasets/3dshape/Vshape3d_2denv4 \
      --modelPath ./Experiments/3dshape_2d_june03 --name Vshape3d_2denv4 \
      --epochs 5000 \
      --device cuda:2
   ```

2. **Eval**:
   ```
   python evaluate_training_3d_batched.py \
      --dataPath testing_data/3dshape/Vshape3d_2denv4 \
      --out ./results/output_3d/Vshape3d_2denv4_june03 \
      --models metric_june03 \
      --checkpoint ./Experiments/3dshape_2d_june03/Vshape3d_2denv4/latest.pt \
      --2d
   ```

##### 4shape3d_2denv4_june03

1. **Train** (no preprocess step -- reuses the shared dataset):
   ```
   python train/train_3dshape_june03.py \
      --dataPath ./datasets/3dshape/4shape3d_2denv4 \
      --modelPath ./Experiments/3dshape_2d_june03 --name 4shape3d_2denv4 \
      --epochs 5000 \
      --device cuda:2
   ```

2. **Eval**:
   ```
   python evaluate_training_3d_batched.py \
      --dataPath testing_data/3dshape/4shape3d_2denv4 \
      --out ./results/output_3d/4shape3d_2denv4_june03 \
      --models metric_june03 \
      --checkpoint ./Experiments/3dshape_2d_june03/4shape3d_2denv4/latest.pt \
      --2d
   ```

##### Tshape3d_2denv4_june03

1. **Train** (no preprocess step -- reuses the shared dataset):
   ```
   python train/train_3dshape_june03.py \
      --dataPath ./datasets/3dshape/Tshape3d_2denv4 \
      --modelPath ./Experiments/3dshape_2d_june03 --name Tshape3d_2denv4 \
      --epochs 5000 \
      --device cuda:2
   ```

   The dataset was built under the pre-env-tag name `Tshape3d_env4` and renamed to
   `Tshape3d_2denv4` on 2026-09-10 (see "Legacy dataset" below), so dataset and run
   folder now share the env-tagged name like every other cell.

2. **Eval**:
   ```
   python evaluate_training_3d_batched.py \
      --dataPath testing_data/3dshape/Tshape3d_2denv4 \
      --out ./results/output_3d/Tshape3d_2denv4_june03 \
      --models metric_june03 \
      --checkpoint ./Experiments/3dshape_2d_june03/Tshape3d_2denv4/latest.pt \
      --2d
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

The recovered June-3 network is driven separately by `_run_2d_june03_train.sh`
(all 28 2-D cells, 5000 epochs, round-robin over `SLOTS`).  It has **no
preprocessing step**: it reuses the shared `datasets/3dshape/<shape>_<env>` dirs,
because `models/metric_june03/data_mlp.py` reads only the `sampled_points` /
`speed` / `normal` arrays those already contain.  Cells with an existing
`latest.pt` are skipped, so it resumes after an interruption; `FORCE=1` retrains.

```
bash _run_2d_june03_train.sh                              # all 28 cells
SHAPES="rectangle Tshape3d" SLOTS="cuda:1" bash _run_2d_june03_train.sh
```

### Legacy dataset

`datasets/3dshape/Tshape3d_2denv4` is the original 2-D dataset for the
(`Tshape3d`, `2denv4`) pair, generated 2026-08-18 with these same settings before the
env-tag naming above existed, under the name `Tshape3d_env4`.  It was renamed on
2026-09-10 so that the SE(3) `Tshape3d_env4` cell (section "env4" above) could take
the name the `<shape>_<env>` convention gives it.  The other artefacts of that early
planar run moved with it and now carry a `_legacy` suffix:

| was | now |
| --- | --- |
| `datasets/3dshape/Tshape3d_env4` | `datasets/3dshape/Tshape3d_2denv4` (still the training set of the current `Tshape3d_2denv4` row) |
| `testing_data/3dshape/Tshape3d_env4` (+ `_tight`) | `testing_data/3dshape/Tshape3d_2denv4_legacy` (+ `_tight`) -- the current row scores on `Tshape3d_2denv4`, generated 2026-09-07 |
| `results/output_3d/Tshape3d_env4` | `results/output_3d/Tshape3d_2denv4_legacy` |
| `results/spread_sd/Tshape3d_env4` | `results/spread_sd/Tshape3d_2denv4_legacy` |
| `Experiments/3dshape/3dshape_08_19_12_31` | unchanged (timestamped run folder) |

The old row is kept in `../../experiments_ours.md` as `Tshape3d_2denv4_legacy (2D)`
for continuity.  Baseline trees under `../../baselines/` still refer to their own
`Tshape3d_env4` run folders (`Experiments/3dshape_metric/Tshape3d_env4_*` etc.);
those folder names are unaffected, but any baseline script that reads
`datasets/3dshape/Tshape3d_env4` or `testing_data/3dshape/Tshape3d_env4` now sees
the SE(3) cell, not the planar one.

Likewise `*/Tshape3d_env1_aug18` (+ `_tight`) is the 2026-08-18/19 3-D T-shape data
that predates the README entry (test set at `--offset 0.01`, 100-case result); the
`Tshape3d_env1` cell was regenerated at the shared settings on 2026-09-11.

## Gibson "Aloha" scan (`datasets/gibson/preprocess_obj_gibson.py`)

`datasets/gibson/preprocess_obj_gibson.py` is the Gibson variant of
`dataprocessing/preprocess_obj.py`: it defaults to `--env Aloha.obj` and keeps the
inside-mesh (winding-number) filter **on**, so every placement must lie inside the
building (pass `--no_inside_check` only for the open box envs).  The shape OBJs
(`drone.obj`, `lamp.obj`, `toycar.obj`, and `drone_small.obj` -- the drone cube at
half size, 0.15 a side) are simple boxes that sit next to the script.  Run it from `datasets/gibson` (inside the docker container) with the same
parameters as the 3-D cells above, except `--batch_size 1000` for the training
set: the Aloha surface samples to ~44k env points (vs ~10k for the box envs), so
the `(B, M, F, 3)` clearance tensor in `calculate_dist` is ~4x larger per config and
`--batch_size 2000` OOMs a 24 GB card (~14 GB at 1000).  Smoke-tested 2026-09-13
on `drone.obj` (200 pairs, ~5 s on one GPU); the `lamp_Aloha` / `toycar_Aloha`
sets were generated 2026-09-13 with the commands below.

### drone_Aloha

1. **Preprocess**
   ```
    cd datasets/gibson
    python preprocess_obj_gibson.py \
        --env   Aloha.obj \
        --shape drone.obj \
        --out   ../3dshape/drone_Aloha \
        --num_samples 800000 \
        --visualize \
        --batch_size 1000 \
        --device cuda:2
   ```

   ```
    cd datasets/gibson
    python preprocess_obj_gibson.py \
        --env   Aloha.obj \
        --shape drone.obj \
        --out   ../../testing_data/3dshape/drone_Aloha \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/drone_Aloha
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/drone_Aloha \
      --out ./results/output_3d/drone_Aloha
   ```

### lamp_Aloha

1. **Preprocess**
   ```
    cd datasets/gibson
    python preprocess_obj_gibson.py \
        --env   Aloha.obj \
        --shape lamp.obj \
        --out   ../3dshape/lamp_Aloha \
        --num_samples 800000 \
        --visualize \
        --batch_size 1000 \
        --device cuda:2
   ```

   ```
    cd datasets/gibson
    python preprocess_obj_gibson.py \
        --env   Aloha.obj \
        --shape lamp.obj \
        --out   ../../testing_data/3dshape/lamp_Aloha \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/lamp_Aloha
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/lamp_Aloha \
      --out ./results/output_3d/lamp_Aloha
   ```

### toycar_Aloha

1. **Preprocess**
   ```
    cd datasets/gibson
    python preprocess_obj_gibson.py \
        --env   Aloha.obj \
        --shape toycar.obj \
        --out   ../3dshape/toycar_Aloha \
        --num_samples 800000 \
        --visualize \
        --batch_size 1000 \
        --device cuda:2
   ```

   ```
    cd datasets/gibson
    python preprocess_obj_gibson.py \
        --env   Aloha.obj \
        --shape toycar.obj \
        --out   ../../testing_data/3dshape/toycar_Aloha \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/toycar_Aloha
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/toycar_Aloha \
      --out ./results/output_3d/toycar_Aloha
   ```

### drone_small_Aloha

`drone_small.obj` is `drone.obj` scaled by 1/2: a 0.15 x 0.15 x 0.15 cube (0.015 a
side in env-normalized units, the Aloha bbox being 10.0 wide).  Same commands as
`drone_Aloha` with the shape swapped, and the training set generated with
`--margin 0.02 --offset 0.0005` (defaults 0.05 / 0.001).

1. **Preprocess**
   ```
    cd datasets/gibson
    python preprocess_obj_gibson.py \
        --env   Aloha.obj \
        --shape drone_small.obj \
        --out   ../3dshape/drone_small_Aloha \
        --num_samples 800000 \
        --margin 0.02 \
        --offset 0.0005 \
        --visualize \
        --batch_size 1000 \
        --device cuda:2
   ```

   ```
    cd datasets/gibson
    python preprocess_obj_gibson.py \
        --env   Aloha.obj \
        --shape drone_small.obj \
        --out   ../../testing_data/3dshape/drone_small_Aloha \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/drone_small_Aloha
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/drone_small_Aloha \
      --out ./results/output_3d/drone_small_Aloha
   ```

### tabletop_Aloha

`datasets/gibson/tabletop.obj` is a 0.05 x 0.6 x 0.6 box (a thin plate, 0.005 x 0.06 x
0.06 in env-normalized units), same face list as the other box shapes.  Training set
at `--margin 0.015 --offset 0.001` (band [0.067, 1]).  Low acceptance (the plate only
fits standing in open floor or lying flat): 3k previews ran at ~8 / 23 / 29 pairs/s
for margin 0.01 / 0.015 / 0.02 on one card, i.e. ~10 h for 800k at 0.015.

1. **Preprocess**
   ```
    cd datasets/gibson
    python preprocess_obj_gibson.py \
        --env   Aloha.obj \
        --shape tabletop.obj \
        --out   ../3dshape/tabletop_Aloha \
        --num_samples 800000 \
        --margin 0.015 \
        --offset 0.001 \
        --visualize \
        --batch_size 1000 \
        --device cuda:1
   ```

   ```
    cd datasets/gibson
    python preprocess_obj_gibson.py \
        --env   Aloha.obj \
        --shape tabletop.obj \
        --out   ../../testing_data/3dshape/tabletop_Aloha \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/tabletop_Aloha
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/tabletop_Aloha \
      --out ./results/output_3d/tabletop_Aloha
   ```

### tabletop_Aloha-y

`datasets/gibson/Aloha-y.obj` is the Aloha scan with everything at `y > 0` removed
(the y in [-5, 0] half, 5.9 x 5.0 x 2.0 m), made by `datasets/gibson/_make_cut_mesh.py`:
triangles straddling the plane are clipped at it and the opening is capped with a
2 cm grid of inward-facing quads over the interior cross-section (group `o cap`), so
the half-house stays closed for the inside-mesh gate and the cap is a real wall for
the clearance queries.  Longest extent is now 5.0 m, so 1 env-normalized unit = 5 m
(`--margin 0.015` = 7.5 cm here vs 15 cm on Aloha).

```
cd datasets/gibson
python _make_cut_mesh.py Aloha.obj Aloha-y.obj --axis y --max 0 --cap_res 0.02
```

Same tabletop and settings as `tabletop_Aloha`; a 3k preview ran at ~10 pairs/s.

1. **Preprocess**
   ```
    cd datasets/gibson
    python preprocess_obj_gibson.py \
        --env   Aloha-y.obj \
        --shape tabletop.obj \
        --out   ../3dshape/tabletop_Aloha-y \
        --num_samples 800000 \
        --margin 0.015 \
        --offset 0.001 \
        --visualize \
        --batch_size 1000 \
        --device cuda:1
   ```

   ```
    cd datasets/gibson
    python preprocess_obj_gibson.py \
        --env   Aloha-y.obj \
        --shape tabletop.obj \
        --out   ../../testing_data/3dshape/tabletop_Aloha-y \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 1000 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/tabletop_Aloha-y
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/tabletop_Aloha-y \
      --out ./results/output_3d/tabletop_Aloha-y
   ```

## Gibson "Badger" scan

`datasets/gibson/Badger.obj` is a second Gibson house (1.15M triangles / 572k
vertices, bbox 5.63 x 10.0 x 1.78 m -- 13x the triangle count of Aloha).  Same
script and settings as the Aloha cells, with two differences forced by the mesh
size: `sample_surface_points` seeds the env cloud with every mesh vertex, so the
cloud is ~570k points and the per-placement broad-phase cull (`(B, E, 3)`) is 13x
bigger per config than on Aloha -- the training set therefore runs at
`--batch_size 250` (`1000` does not fit a 24 GB card); and the inside-mesh gate
uses the cached winding-number grid (`--inside_grid 0.002`, the default: nodes
evaluated once with `igl.fast_winding_number`, ~20 s, then a trilinear gather per
batch with an exact re-check in wall-crossing cells).  The brute-force gate
(`--inside_grid 0`) is ~5x slower on this mesh (~28 h vs ~6 h for 800k pairs).

### drone_small_Badger

Training set at `--margin 0.02 --offset 0.0005`, as for `drone_small_Aloha`.

1. **Preprocess**
   ```
    cd datasets/gibson
    python preprocess_obj_gibson.py \
        --env   Badger.obj \
        --shape drone_small.obj \
        --out   ../3dshape/drone_small_Badger \
        --num_samples 800000 \
        --margin 0.02 \
        --offset 0.0005 \
        --visualize \
        --batch_size 250 \
        --device cuda:0
   ```

   ```
    cd datasets/gibson
    python preprocess_obj_gibson.py \
        --env   Badger.obj \
        --shape drone_small.obj \
        --out   ../../testing_data/3dshape/drone_small_Badger \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 250 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/drone_small_Badger
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/drone_small_Badger \
      --out ./results/output_3d/drone_small_Badger
   ```

### pole_Badger

`datasets/gibson/pole.obj` is a 0.05 x 0.05 x 0.7 box (a pole standing along z;
0.005 x 0.005 x 0.07 in env-normalized units), same face list as the other box
shapes.  Same commands and settings as `drone_small_Badger` with the shape swapped
and the training set generated at `--margin 0.01 --offset 0.001`.

1. **Preprocess**
   ```
    cd datasets/gibson
    python preprocess_obj_gibson.py \
        --env   Badger.obj \
        --shape pole.obj \
        --out   ../3dshape/pole_Badger \
        --num_samples 800000 \
        --margin 0.01 \
        --offset 0.001 \
        --visualize \
        --batch_size 250 \
        --device cuda:2
   ```

   ```
    cd datasets/gibson
    python preprocess_obj_gibson.py \
        --env   Badger.obj \
        --shape pole.obj \
        --out   ../../testing_data/3dshape/pole_Badger \
        --num_samples 1000 \
        --testing_data \
        --offset 0.02 \
        --batch_size 250 \
        --visualize
   ```
2. **Train**:
   ```
   python train/train_3dshape.py --dataPath datasets/3dshape/pole_Badger
   ```

3. **Eval**:
   ```
   python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/pole_Badger \
      --out ./results/output_3d/pole_Badger
   ```
