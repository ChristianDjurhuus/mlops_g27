# Running gcloud
FROM gcr.io/cloud-builders/gsutil

ARG KEY_FILE_CONTENT
RUN gcloud auth activate-service-account g27-bucket@mlops-g27.iam.gserviceaccount.com --key-file=$KEY_FILE_CONTENT
RUN echo finished login to gcloud

# Base image
FROM python:3.7-slim

# A basic python installation.
# "Same procedure as last year James!"
RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt clean && rm -rf /var/lib/apt/lists/*

# dvc
RUN apt-get update && apt-get install -y git
WORKDIR /mlops_g27

ADD docker/requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install dvc[gs]

# Prøv at tilføje .git
COPY src/models/ src/models/
COPY setup.py setup.py
COPY .dvc .dvc
COPY data.dvc data.dvc
COPY .git .git
COPY data_path.py data_path.py
COPY setup.py setup.py

# dvc
RUN git config user.email "jonpo@dtu.dk"
RUN git config user.name "jonpodtu"
RUN dvc remote modify --local remote_storage \
        credentialpath $KEY_FILE_CONTENT
RUN dvc pull

# python package
RUN pip install -e .

# Entrypoint: The application we want to run when the image is being executed
ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
