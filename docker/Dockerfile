ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.12-py3

FROM ${FROM_IMAGE_NAME}

RUN update-alternatives --install /usr/bin/python python $(which python3) 1

RUN pip3 install --upgrade pip && pip3 install \
    torch==1.9.1+cu111 \
    torchaudio \
    -f https://download.pytorch.org/whl/cu111/torch_stable.html

RUN pip3 install --upgrade pip && pip3 install \
    lmdb \
    torch-scatter==2.0.9 \
    -f https://pytorch-geometric.com/whl/torch-1.9.1+cu111.html
RUN pip3 install --upgrade pip && pip3 install \
    torch-sparse==0.6.12 \
    -f https://pytorch-geometric.com/whl/torch-1.9.1+cu111.html
RUN pip3 install --upgrade pip && pip3 install torch-geometric==1.7.2

RUN git clone https://github.com/argoai/argoverse-api
RUN wget https://s3.amazonaws.com/argoai-argoverse/hd_maps.tar.gz -P argoverse-api
RUN cd argoverse-api; tar xvf hd_maps.tar.gz
RUN pip3 install argoverse-api/
# Otherwise I'm getting an error; probably has to do with the fact pip3 is a conda pip
RUN cd argoverse-api; cp -r map_files/ /opt/conda/lib/python3.8/site-packages/

RUN pip3 uninstall -y opencv-python && pip3 install opencv-python
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6
RUN pip3 install \
    pandas==1.2.4 \
    tensorboard-plugin-wit==1.8.1 \
    tensorboard==2.10.0 \
    torch-tb-profiler==0.4.0
