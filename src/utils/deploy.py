import io

import datasets
import torch
from datasets import Dataset
from google.cloud import storage
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler


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


BUCKET_NAME = "g27-models"
MODEL_FILE = "model1.pt"
client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)
blob = bucket.get_blob(MODEL_FILE)
classification = {0: "Negative", 1: "Positive"}

model = ImdbTransformer(
    model_name="bert-base-cased", learning_rate=0.01, batch_size=24,
)
print("model defined")
data = io.BytesIO(blob.download_as_string())
print("io")
model.load_state_dict(torch.load(data, map_location=torch.device("cpu")))
print("model loaded")
model.eval()

predictions = []


def tokenize_function(sample_data):
    return tokenizer(sample_data["text"], padding="max_length", truncation=True)


def bert_predicter(request):
    print("im in bert :)")
    requst_json = request.get_json()
    if request_json and "input_data" in requet_json:
        sample_data = request_json["input_data"]
        tokenized_sample = Dataset.from_dict(sample_data).map(
            tokenize_function, batched=True
        )
        tokenized_sample = tokenized_sample.remove_columns(["text"])
        tokenized_sample = tokenized_sample.rename_column("label", "labels")
        tokenized_sample.set_format("torch")
        dataloader = DataLoader(tokenized_sample)
        with torch.no_grad():
            for batch in dataloader:
                outputs = model(**batch)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                predictions.append((classification[predictions.data[0].item()]))
        return f"Sentiment predictions of string(s): {predictions}"
    else:
        return "No input data received"
