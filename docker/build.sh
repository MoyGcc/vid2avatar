#!/bin/bash

docker build . --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg UNAME=$(id -un) --rm -t v2a-$(id -un)
