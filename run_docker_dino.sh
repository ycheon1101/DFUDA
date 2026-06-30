#!/bin/bash


# Run the DINOv3 container
docker run \
   -ti --rm \
   --network host \
   --gpus all \
   --ipc=host \
   --volume="$(pwd):/home/DFUDA" \
   dfuda-dinov3:v1.0


