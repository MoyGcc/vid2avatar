ARG FROM_IMAGE_NAME=amazevr/23.06:cuda11.7-py3.8-torch2.0.1-pytorch3d0.7.2-kaolin0.14.0
FROM ${FROM_IMAGE_NAME}

ADD . /workspace/v2a
WORKDIR /workspace/v2a

#RUN python -m pip install --no-cache-dir .

ARG UNAME
ARG UID
ARG GID
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
USER $UNAME
