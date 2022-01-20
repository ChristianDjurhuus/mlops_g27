# Torch Imports
import torch
from pytorch_lightning import LightningModule

# Transformers
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler
import datasets


class ImdbTransformer(LightningModule):
    """A Pytorch-Lightning DataModule"""

    # The different parameters are initialized and
    # utilized through save_hyperparmeters() function
    def __init__(
        self,
        model_name: str = "bert-base-cased",
        learning_rate: float = 5e-5,
        batch_size: int = 32,
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
