# Running gcloud
FROM gcr.io/cloud-builders/gsutil

# Dockerfile-gpu
FROM nvidia/cuda:10.1-cudnn7-runtime

RUN apt update && \
    apt install --no-install-recommends -y build-essential software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.8 python3-pip python3-setuptools python3-distutils && \
    apt clean && rm -rf /var/lib/apt/lists/*

# git is needed to run DVC as we use git for version control
RUN apt-get update && apt-get install -y git
WORKDIR /mlops_g27

# Installing basic requirements
ADD docker/requirements_gpu.txt .
RUN python3.8 -m pip install --upgrade pip
RUN python3.8 -m pip install cython
RUN python3.8 -m pip install -r requirements_gpu.txt --no-cache-dir
RUN python3.8 -m pip install dvc[gs]
RUN python3.8 -m pip install antlr4-python3-runtime

# Importing nessecary folders 
COPY src/ src/
COPY setup.py setup.py
COPY .dvc .dvc
COPY data.dvc data.dvc
COPY .git .git
COPY data_path.py data_path.py
COPY setup.py setup.py

RUN git config user.email "jonpo@dtu.dk"
RUN git config user.name "jonpodtu"

# Pull data into folder: data
RUN dvc pull

# Due to wandbLogger - oev
RUN python3.8 -m pip install setuptools==59.5.0
RUN python3.8 -m pip install wandb

# python package
RUN python3.8 -m pip install -e .

# Entrypoint: The application we want to run when the image is being executed
ENTRYPOINT ["python3.8", "-u", "src/models/train_model.py"]