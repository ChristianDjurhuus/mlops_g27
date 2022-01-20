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
        debug=False,
        seed=None
    ):
        super().__init__()
        self.model_name = model_name
        self.data_path = data_path
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.debug = debug
        self.seed = seed
        max_pos_workers = multiprocessing.cpu_count() * 2

        # No more than 8 workers are recommended
        self.n_workers = 8 if(max_pos_workers >= 8) else max_pos_workers

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
                self.tokenized_datasets["train"].shuffle(seed=self.seed).select(range(7500)),
                self.batch_size,
                num_workers=self.n_workers
            )
        else:
            return DataLoader(self.tokenized_datasets["train"],
                              batch_size=self.batch_size,
                              num_workers=self.n_workers)

    def val_dataloader(self):
        if self.debug:
            return DataLoader(
                self.tokenized_datasets["valid"].shuffle(seed=self.seed).select(range(2500)),
                self.batch_size,
                num_workers=self.n_workers
            )
        return DataLoader(self.tokenized_datasets["valid"],
                          batch_size=self.batch_size,
                          num_workers=self.n_workers)

    def test_dataloader(self):
        if self.debug:
            return DataLoader(
                self.tokenized_datasets["test"].shuffle(seed=self.seed).select(range(2500)),
                self.batch_size,
                num_workers=self.n_workers
            )
        return DataLoader(self.tokenized_datasets["test"],
                          batch_size=self.batch_size,
                          num_workers=self.n_workers)

    def convert_to_features(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)
