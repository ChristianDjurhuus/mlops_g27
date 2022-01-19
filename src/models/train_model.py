import os

# Torch Imports
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer,
                               seed_everything)
from torch.utils.data import DataLoader

# Experiment tracking and setup
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

# Transformers
from transformers import (AdamW, AutoModelForSequenceClassification,
                          AutoTokenizer, get_scheduler)
import datasets

class ImdbDataModule(LightningDataModule):
    """A Pytorch-Lightning DataModule"""

    def __init__(
        self,
        model_name: str = "bert-base-cased",
        data_path: str = "data/processed",
        batch_size: int = 32,
        debug = False
    ):
        super().__init__()
        self.model_name = model_name
        self.data_path = data_path
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.debug=debug

    def setup(self, stage=None):
        self.datasets = datasets.load_from_disk(self.data_path)
        self.tokenized_datasets = self.datasets.map(
            self.convert_to_features, batched=True
        )
        self.tokenized_datasets = self.tokenized_datasets.remove_columns(["text"])
        self.tokenized_datasets = self.tokenized_datasets.rename_column(
            "label", "labels"
        )
        self.tokenized_datasets.set_format("torch")

    def prepare_data(self):
        datasets.load_from_disk(self.data_path)
        AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

    def train_dataloader(self):
        if self.debug:
            return DataLoader(
                self.tokenized_datasets["train"].shuffle(seed=42).select(range(4)),
                self.batch_size,
            )
        else:
            return DataLoader(self.tokenized_datasets["train"], self.batch_size)

    def val_dataloader(self):
        if self.debug:
            return DataLoader(
                self.tokenized_datasets["valid"].shuffle(seed=42).select(range(2)),
                self.batch_size,
            )
        return DataLoader(self.tokenized_datasets["valid"], self.batch_size)

    def convert_to_features(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)


class ImdbTransformer(LightningModule):
    """A Pytorch-Lightning DataModule"""

    # The different parameters are initialized and
    # utilized through save_hyperparmeters() function
    def __init__(
        self,
        model_name: str = "bert-base-cased",
        learning_rate: float = 5e-5,
        batch_size: int = 32
    ):
        super().__init__()
        # save all hyperparameters
        self.save_hyperparameters()
        # define the model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        self.metric = datasets.load_metric("accuracy")

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        # This parameter——batch_idx——cannot be deleted arbitrarily
        outputs = self(**batch)
        loss = outputs.loss  # or outputs[0]
        return loss

    def training_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train_loss", loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]  # Or outputs.loss & outputs.logits
        preds = torch.argmax(logits, axis=1)
        labels = batch["labels"]
        return {"loss": val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("valid_loss", loss, prog_bar=True)
        self.log_dict(
            self.metric.compute(predictions=preds, references=labels), prog_bar=True
        )
        return loss

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.trainer.datamodule.train_dataloader()

        # Calculate total steps
        tb_size = self.hparams.batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear)"""
        model = self.model
        optimizer = AdamW(model.parameters(), lr=self.hparams.learning_rate)

        scheduler = get_scheduler(
            "linear", optimizer, num_warmup_steps=0, num_training_steps=self.total_steps
        )

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


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
                        debug = debug_toggle)
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
