# Base image
FROM python:3.7-slim

# A basic python installation.
# "Same procedure as last year James!"
RUN apt update && \
apt install --no-install recommends \
apt clean && rm -rf /var/lib/apt/*

# Copying essetial files to the VM
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/models/ src/models/

# Pull data with dvc
COPY data/processed data/processed

# Install dependencies. 
# Note: --no-cache-dir will reduce image size
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

# Entrypoint: The application we want to run when the image is being executed
ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
