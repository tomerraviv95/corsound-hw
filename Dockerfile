# Use the official Python 3.7 image as the base image
FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

# Install base utilities
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# Set the working directory
WORKDIR /python-code

# Copy your Python code into the container
COPY . /python-code

# Install packages from list of requirements
RUN conda env create -f environment.yml