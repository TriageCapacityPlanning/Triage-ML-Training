FROM tensorflow/tensorflow:2.0.0-gpu-py3

# Tensorboard
EXPOSE 6006

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y wget libsm6 libxext6 libfontconfig1 libxrender1 python3-tk

RUN apt-get install -y vim

COPY ml-training /opt/ml-training

RUN cd /opt/ml-training && \
    pip install .

CMD cd /opt/ml-training && /bin/bash



