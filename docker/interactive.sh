#!/bin/bash

docker run -it --rm --gpus all --shm-size=256g \
  -v $PWD:/workspace/v2a \
  -v /media/AI:/media/AI \
  v2a-$(id -un) bash
