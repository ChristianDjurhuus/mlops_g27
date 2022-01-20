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

    def setup(self, stage='fit'):
        if stage == 'fit':
            # if fitting model, we only tokenize train and validation data
            self.datasets = datasets.load_from_disk(self.data_path)
            self.tokenized_datasets = self.datasets.map(
                self.convert_to_features, batched=True
            )
            self.tokenized_datasets = self.tokenized_datasets.remove_columns(["text"])
            self.tokenized_datasets = self.tokenized_datasets.rename_column(
                "label", "labels"
            )
            self.tokenized_datasets.set_format("torch")
        else:
            # if test model or do inference, we only tokenize the test data
            self.dataset = datasets.load_from_disk(self.data_path)['test']
            self.tokenized_dataset = self.dataset.map(
                self.convert_to_features, batched=True
            )
            self.tokenized_dataset = self.tokenized_dataset.remove_columns(["text"])
            self.tokenized_dataset = self.tokenized_dataset.rename_column(
                "label", "labels"
            )
            self.tokenized_dataset.set_format("torch")

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
        # if we set the stage in setup() is 'fit', do not call this func.
        # this only for test model or inference
        if self.debug:
            return DataLoader(
                self.tokenized_dataset.shuffle(seed=self.seed).select(range(2500)),
                self.batch_size,
                num_workers=self.n_workers
            )
        return DataLoader(self.tokenized_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.n_workers)

    def convert_to_features(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)
