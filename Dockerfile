FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
## Note that there is not a PyTorch 2.0.1 image with CUDA 11.8

LABEL authors="Colby T. Ford <colby@tuple.xyz>"

## Set environment variables
ENV MPLCONFIGDIR /data/MPL_Config
ENV TORCH_HOME /data/Torch_Home
ENV TORCH_EXTENSIONS_DIR /data/Torch_Extensions
ENV DEBIAN_FRONTEND noninteractive

## Install system requirements
RUN apt update && \
    apt-get install -y --reinstall \
        ca-certificates && \
    apt install -y \
        git \
        vim \
        wget \
        libxml2 \
        libgl-dev \
        libgl1

## Make directories
RUN mkdir -p /software/
WORKDIR /software/

## Install dependencies from Conda/Mamba
COPY environment.yaml /software/environment.yaml
RUN conda env create -f environment.yaml
RUN conda init bash && \
    echo "conda activate GCDM-SBDD" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

## Install GCDM-SBDD
RUN git clone https://github.com/BioinfoMachineLearning/GCDM-SBDD && \
    cd GCDM-SBDD && \
    pip install -e .
WORKDIR /software/GCDM-SBDD/

CMD /bin/bash