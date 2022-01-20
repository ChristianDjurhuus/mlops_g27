import os

# Torch Imports
import torch
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning import Trainer, seed_everything

# Experiment tracking and setup
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import wandb
from google.cloud import storage

from src.data.load_data import ImdbDataModule
from src.models.model import ImdbTransformer

MODEL_FILE_NAME = 'model.pt'

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )


@hydra.main(config_path="config", config_name="default_config.yaml")
def main(cfg: DictConfig):
    # Hyperparmeters
    wandb.init(mode="disabled")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HYDRA_FULL_ERROR"] = "1"
    cfg = cfg.experiment
    lr = cfg.hyper_param["lr"]
    epochs = cfg.hyper_param["epochs"]
    batch_size = cfg.hyper_param["batch_size"]
    
    # Seed and model-type
    seed = cfg.seed
    model = cfg.model
    
    # DEBUG mode: Use small datasets instead of whole
    debug_toggle = cfg.debug_mode

    # Return wheither there is 1 or 0 GPUs
    if cfg.force_CPU == True:
        GPUs = 0
    else:
        GPUs = torch.cuda.device_count()


    """
        # FOR HPC
        GPUs = min(1, torch.cuda.device_count())
        if GPUs == 1:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    """

    # Fix random seed
    seed_everything(seed)

    # Load Data
    dm = ImdbDataModule(model_name=model, 
                        data_path=to_absolute_path(cfg.data_path), 
                        batch_size=batch_size,
                        debug = debug_toggle,
                        seed = seed)
    dm.prepare_data()
    dm.setup("fit")

    # Import Model
    model = ImdbTransformer(
        model_name=model,
        learning_rate=lr,
        batch_size=batch_size,
    )

    # Directing the hyperparameters to wandb
    config = {
        "model": model,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
        "seed": seed,
    }
    wandb_logger = WandbLogger(project=cfg.wandb.model_dir, entity=cfg.wandb.entity, config=config)

    # Train Model
    trainer = Trainer(max_epochs=epochs, 
                      precision=16,
                      gpus=GPUs, 
                      logger=wandb_logger, 
                      default_root_dir=to_absolute_path(cfg.wandb.model_dir))
    trainer.fit(model, datamodule=dm)

    # If local path given, assume to save it locally
    if not os.path.exists(cfg.local_path):
        os.makedirs(cfg.local_path)
    tmp_file_name = os.path.join(cfg.local_path, MODEL_FILE_NAME)    
    torch.save(model.state_dict(), tmp_file_name)
    if cfg.google_bucket_path != None:
        upload_blob(cfg.google_bucket_path, tmp_file_name, "model1.pt")



if __name__ == "__main__":
    main()
