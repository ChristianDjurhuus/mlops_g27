import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def dataload():
    # Fetching data
    processed_dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = processed_dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    return tokenized_datasets


class TestClass:
    tokenized_datasets = dataload()
    full_train_dataset = tokenized_datasets["train"]
    full_eval_dataset = tokenized_datasets["test"]
    N_train = 25000
    N_test = 25000

    # Testing trainingdata
    def test_traindata(self):
        # Number of documents in train corpus
        labels = self.full_train_dataset["labels"]
        assert (
            len(self.full_train_dataset) == self.N_train
        ), "Train data did not have the correct number of documents"

        # Labels
        assert (
            len(labels) == self.N_train
        ), "Train data did not have the correct number of labels"
        assert all(
            i in torch.unique(labels) for i in range(1)
        ), "At least one train data label wasn't correct."

    def test_testdata(self):
        # Image structure
        labels = self.full_eval_dataset["labels"]
        assert len(self.full_eval_dataset) == self.N_test, (
            f"Test data did not have the correct number of documents, "
            f"but had: {len(self.full_eval_dataset)}"
        )

        # Labels
        assert (
            len(labels) == self.N_test
        ), "Test data did not have the correct number of labels"
        assert all(
            i in torch.unique(labels) for i in range(1)
        ), "At least one test data label wasn't correct."
