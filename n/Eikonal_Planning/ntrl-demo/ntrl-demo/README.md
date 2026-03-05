## About
This is a minimal example. 

## Setup
1. git clone this repo
2. run `docker build -f Dockerfile.new -t newpytorch.` under the root directory of this repo, once you built the docker image, you don't need to build it again unless you change the dockerfile.
3. run `sudo docker run --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume="/home/jeffrey/ntrlshape/n/Eikonal_Planning/ntrl-demo:/workspace" --volume="/usr/lib/x86_64-linux-gnu/:/glu" --volume="/home/jeffrey/ntrlshape/n/.local:/.local" --env="QT_X11_NO_MITSHM=1"  --gpus all -ti --rm newpytorch` to start the docker container.
