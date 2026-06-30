#!/bin/bash


# Allow connections to X server from local machine's root user
#xhost +si:localuser:root


# Run the Docker container
docker run \
   -ti --rm \
   --network host \
   --gpus all \
   --ipc=host \
   --env="DISPLAY" \
   --env="QT_X11_NO_MITSHM=1" \
   --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
   --volume="$HOME/.Xauthority:/home/.Xauthority:rw" \
   --volume="$(pwd):/home/DFUDA" \
   dfuda:v1.0



