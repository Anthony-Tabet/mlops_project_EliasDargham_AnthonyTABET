FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update\
  && apt-get upgrade -y\
  && apt-get install -y python3.12 python3-pip python3.12-venv

# Create a virtual environment
RUN python3.12 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Activate the virtual environment
RUN python3 -m venv $VIRTUAL_ENV

RUN pip install --no-cache-dir pipx poetry

# Set the working directory
RUN mkdir /opt/app
WORKDIR /opt/app

# Copy the current directory contents into the container at /opt/app
COPY . /opt/app

RUN poetry lock --no-update
RUN poetry install --no-root