from datasets import Dataset
from transformers import AutoTokenizer

# A test data
sample_data = {
    "text": [
        "This is a good movie, my family are all like it!",
        "I don't like this film, The story is too slow.",
    ],
    "label": [1, 0],
}

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_sample = Dataset.from_dict(sample_data).map(tokenize_function, batched=True)

assert list(dict(tokenized_sample.features).keys()) == [
    "attention_mask",
    "input_ids",
    "label",
    "text",
    "token_type_ids",
]
assert tokenized_sample.num_rows == len(sample_data["text"])
