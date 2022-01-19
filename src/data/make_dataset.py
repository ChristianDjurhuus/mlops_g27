import os

from datasets import DatasetDict, load_from_disk
from download_dataset import download_dataset

# os.getcwd()

# Download the raw datasets and save to "data/raw"
download_dataset(name="imdb", save_path="data/raw")

# Load the downloaded raw data
dataset_dict = load_from_disk("data/raw")
train_set = dataset_dict["train"]
test_set = dataset_dict["test"]
unsupervised_set = dataset_dict["unsupervised"]

# Making our processed dataset, we donnot need the unsupervised dataset.
# so we keep the raw train data as the processed train data,
# and split the raw test data in two halves as the test data and validation data.
test_valid = test_set.train_test_split(0.5, 0.5)
train_test_valid_dataset = DatasetDict(
    {"train": train_set, "test": test_valid["test"], "valid": test_valid["train"]}
)

# Save the processed dataset dict
if not os.path.exists("data/processed"):
    os.makedirs("data/processed")
train_test_valid_dataset.save_to_disk("data/processed")

# Load the processed dataset dict
dataset_dict = load_from_disk("data/processed")
train_set = dataset_dict["train"]
test_set = dataset_dict["test"]
valid_set = dataset_dict["valid"]
print("This is our training dataset:\t", type(train_set), len(train_set))
print("This is our test dataset:\t", type(test_set), len(test_set))
print("This is our validation dataset:\t", type(valid_set), len(valid_set))
