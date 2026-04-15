

Docker run
```
cd ~/ntrlshape/n/Eikonal_Planning/ntrl-demo/ntrl-demo

docker build -f Dockerfile.new -t newpytorch .

sudo docker run --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume="/home/jeffrey/ntrlshape/n/Eikonal_Planning/ntrl-demo:/workspace" --volume="/usr/lib/x86_64-linux-gnu/:/glu" --volume="/home/jeffrey/ntrlshape/n/.local:/.local" --env="QT_X11_NO_MITSHM=1"  --gpus all -ti --rm newpytorch 
```

cd in 2dshape
Preprocess
```
# change the out_path variable to create two different files
python preprocess2d_gpu.py
```

Train
```
# change dataPath
python start2dtrain.py

```

Evaluate
```
# change pt to the model you want to use
python evaluate_training_batched.py
```

