import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
)
from src.models.model import ImdbTransformer

# Ideas for testing
# https://thenerdstation.medium.com/how-to-unit-test-machine-learning-code-57cf6fd81765

# Source code:
# https://github.com/suriyadeepan/torchtest/blob/master/torchtest/torchtest.py


# Temporary load of data
raw_datasets = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=4)

# Clearing up memory
del small_train_dataset
del tokenized_datasets


class testClassTraining:
    def testing_optimizer(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-cased", num_labels=2
        )
        optimizer = AdamW(model.parameters(), lr=0.001)

        params = [x for x in model.named_parameters() if x[1].requires_grad]
        init_params = [(name, p.clone()) for (name, p) in params]

        # Running a single training step
        num_epochs = 1
        num_training_steps = num_epochs * len(train_dataloader)

        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        # Checking for gpu's
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        model.to(device)

        model.train()
        for _ in range(num_epochs):
            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Check if params have changed
                for (_, p0), (name, p1) in zip(init_params, params):
                    assert not torch.equal(
                        p0.to(device), p1.to(device)
                    ), "Parameters are not being optimized"
                break


    def test_loss(self):
        """
        Test that loss is not 0
        """
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-cased", num_labels=2
        )
        optimizer = AdamW(model.parameters(), lr=0.001)

        # Running a single training step
        num_epochs = 1
        num_training_steps = num_epochs * len(train_dataloader)

        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        # Checking for gpu's
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        model.to(device)

        model.train()
        for _ in range(num_epochs):
            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                assert loss.item() != 0, "Loss is stuck at zero"
                break
