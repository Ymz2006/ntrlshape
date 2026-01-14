Docker run
```
docker run --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume="/media/corallab-s1/4tbhdd/Jeffrey/ntrl-shape/n/Eikonal_Planning/ntrl-demo:/workspace" --volume="/usr/lib/x86_64-linux-gnu/:/glu" --volume="/media/corallab-s1/4tbhdd/Jeffrey/ntrl-shape/n/.local:/.local" --env="QT_X11_NO_MITSHM=1"  --gpus all -ti --rm ntrl:demo
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

