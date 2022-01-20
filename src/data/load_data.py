# Torch Imports
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

# Transformers
from transformers import AutoTokenizer
import datasets
import multiprocessing


class ImdbDataModule(LightningDataModule):
    """A Pytorch-Lightning DataModule"""

    def __init__(
        self,
        model_name: str = "bert-base-cased",
        data_path: str = "data/processed",
        batch_size: int = 32,
        debug = False,
        seed = None
    ):
        super().__init__()
        self.model_name = model_name
        self.data_path = data_path
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.debug = debug
        self.seed = seed
        self.n_workers = multiprocessing.cpu_count()*2

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
                self.tokenized_datasets["train"].shuffle(seed=self.seed).select(range(4)),
                self.batch_size,
            )
        else:
            return DataLoader(self.tokenized_datasets["train"], batch_size=self.batch_size, num_workers=self.n_workers)

    def val_dataloader(self):
        if self.debug:
            return DataLoader(
                self.tokenized_datasets["valid"].shuffle(seed=self.seed).select(range(2)),
                self.batch_size,
            )
        return DataLoader(self.tokenized_datasets["valid"], batch_size=self.batch_size, num_workers=self.n_workers)

    def convert_to_features(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)