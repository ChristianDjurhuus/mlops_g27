# Running gcloud
FROM gcr.io/cloud-builders/gsutil

# Base image
FROM python:3.7-slim

# A basic python installation.
# "Same procedure as last year James!"
RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt clean && rm -rf /var/lib/apt/lists/*

# git is needed to run DVC as we use git for version control
RUN apt-get update && apt-get install -y git
WORKDIR /mlops_g27


# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

ADD docker/requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install dvc[gs]



# Importing nessecary folders 
COPY src/models/ src/models/
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

# python package
RUN pip install -e .

# Entrypoint: The application we want to run when the image is being executed
ENTRYPOINT ["python", "-u", "src/models/fine_tuninng_with_PyTorch_Lightning.py"]
