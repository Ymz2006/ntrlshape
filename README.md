Docker run
```
sudo docker run --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume="/home/jeffrey/ntrlshape/n/Eikonal_Planning/ntrl-demo:/workspace" --volume="/usr/lib/x86_64-linux-gnu/:/glu" --volume="/home/jeffrey/ntrlshape/n/.local:/.local" --env="QT_X11_NO_MITSHM=1"  --gpus all -ti --rm newpytorch

```


Preprocess
```
python dataprocessing/preprocess.py --config configs/gibson.txt 
```

Train
```
python train/train_gib.py
```

Path plan
```
python test/3d_visualize.py
```

Visualize (Run outside of container)
```
python test/Visual3d.py
```

