import torch
from transformers import AutoTokenizer
from datasets import load_from_disk
from torch.utils.data import DataLoader
from datasets import load_metric
from transformers import AutoModelForSequenceClassification
from data_path import get_data_path


def fetch_data():
    path = get_data_path("mlops_g27/data/processed")
    processed_datasets = load_from_disk(path)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = processed_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    small_test_dataset = (
        tokenized_datasets["test"].shuffle(seed=1984).select(range(1000))
    )
    test_dataloader = DataLoader(
        small_test_dataset, shuffle=True, batch_size=4
    )
    return test_dataloader


def evaluate():
    print("Evaluating")
    metric = load_metric("accuracy")
    # initializing wandb logging
    test_dataloader = fetch_data()
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=2
    )
    model.load_state_dict(torch.load("trained_model.pt"))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
            print(f'Accuracy: {metric.compute().get("accuracy")*100}%')


if __name__ == "__main__":
    evaluate()
