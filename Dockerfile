FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
USER root
RUN apt-get update && apt-get upgrade -y\
    && apt-get install -y --no-install-recommends openssh-client\
      git \
      curl \
      build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
