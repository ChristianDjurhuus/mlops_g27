import os

# Torch Imports
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything

# Experiment tracking and setup
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.data.load_data import ImdbDataModule
from src.models.model import ImdbTransformer

@hydra.main(config_path="config", config_name="default_config.yaml")
def main(cfg: DictConfig):
    # Hyperparmeters
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
        GPUs = min(1, torch.cuda.device_count())
        if GPUs == 1:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

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
    wandb_logger = WandbLogger(project="dtu_mlops_g27", entity="dtu_mlops_g27", config=config)

    # Train Model
    trainer = Trainer(max_epochs=epochs, gpus=GPUs, logger=wandb_logger)
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
