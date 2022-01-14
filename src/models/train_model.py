# Model related
from transformers import AutoTokenizer
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from transformers import get_scheduler
import torch
from transformers import AdamW

# Config related
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
import logging

from data_path import get_data_path

log = logging.getLogger(__name__)


def fetch_data(cfg: DictConfig):
    path = get_data_path("mlops_g27/data/processed")
    processed_datasets = load_from_disk(path)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = processed_datasets.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    small_train_dataset = (
        tokenized_datasets["train"].shuffle(seed=cfg.seed).select(range(1000))
    )

    train_dataloader = DataLoader(
        small_train_dataset, shuffle=True, batch_size=cfg.batch_size
    )

    return train_dataloader


@hydra.main(config_path="config", config_name="config.yaml")
def train(cfg: DictConfig):
    # Model hyperparameters
    log.info(f"configuration: \n {OmegaConf.to_yaml(cfg)}")
    cfg = cfg.experiment

    torch.manual_seed(cfg.seed)

    # getting data from disk
    train_dataloader = fetch_data(cfg)

    # Defining model
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model, num_labels=2)
    optimizer = AdamW(model.parameters(), lr=cfg.lr)

    # Implementing learning rate scheduler
    num_epochs = cfg.epochs
    num_training_steps = num_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # Checking for gpu's
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    model.to(device)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        batch_length = 0.0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            batch_length += len(batch)

            log.info("[%d] loss: %.3f" % (epoch + 1, running_loss / batch_length))
            running_loss = 0.0

    torch.save(model.state_dict(), 'trained_model.pt')

if __name__=="__main__":
    train()







