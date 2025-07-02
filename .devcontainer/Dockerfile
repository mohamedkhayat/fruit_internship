FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER root
RUN apt-get update && apt-get upgrade\
 && apt-get install -y --no-install-recommends openssh-client\
      git \
      curl \
      build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

USER $USERNAME
