import torch
import argparse
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from datasets import load_metric
from transformers import AutoModelForSequenceClassification

#################################### Temporaily importing data ###########################################################
raw_datasets = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
full_train_dataset = tokenized_datasets["train"]
full_eval_dataset = tokenized_datasets["test"]

tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)


########################################################################################################################

def evaluate(self):
    print("Evaluating")
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('load_model_from', default="")
    args = parser.par_args(sys.argv[2:])
    print(args)
    metric = load_metric("accuracy")
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model, num_labels=2)
    state_dict =torch.load(args.load_model_from)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        ### NEED TO DEFINE EVAL_DATALOADER ###
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
            print(f'Accuracy: {metric.compute()}')
if __name__=="__main__":
    evaluate()