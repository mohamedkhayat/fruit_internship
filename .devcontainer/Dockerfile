FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git \
      curl \
      build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
