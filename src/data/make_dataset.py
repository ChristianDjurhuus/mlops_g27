# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import subprocess
import zipfile
import glob
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import datasets

max_len = 512

class IMDBDataset(Dataset):
  def __init__(self, reviews, targets, tokenizer, max_len):
    self.reviews = reviews
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.reviews)

  # __getitem__ helps us to get a review out of all reviews
  def __getitem__(self, item):
    review = str(self.reviews[item])
    target = self.targets[item]

    encoding = self.tokenizer.encode_plus(
      review,
      add_special_tokens=True,
      max_length=self.max_len,
      truncation = True,
      return_token_type_ids=False,
      padding='max_length',
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'review_text': review,
      'input_ids': encoding['input_ids'].flatten(),         # flatten() flattens a continguous range of dims in a tensor
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Unzip raw data
    if not os.path.exists(os.path.join(input_filepath, 'aclImdb')):
        # TODO: Why don't ospath work?
        extract('data/raw/aclImdb_v1.tar.gz', input_filepath, extension='tar.gz')
    else:
        print('There should be a test here')
    
    train_data = imdb_dataset(train = True, test = False)
    test_data = imdb_dataset(train = False, test = True)
    
    raw_datasets = datasets.DatasetDict({"train":datasets.DatasetBuilder.as_dataset(train_data),"test":datasets.DatasetBuilder.as_dataset(train_data)})
    print(raw_datasets.keys())
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    print('Done tokenize')
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000)) 
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000)) 
    full_train_dataset = tokenized_datasets["train"]
    full_eval_dataset = tokenized_datasets["test"]

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    print('Loading data....')
    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

    print(train_dataloader)

# Source: https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/download.html#download_file_maybe_extract
def extract(compressed_filename, directory, extension=None):
    """ Extract a compressed file to ``directory``.

    Args:
        compressed_filename (str): Compressed file.
        directory (str): Extract to directory.
        extension (str, optional): Extension of the file; Otherwise, attempts to extract extension
            from the filename.
    """
    logger = logging.getLogger(__name__)
    logger.info('Extracting {}'.format(compressed_filename))

    if extension is None:
        basename = os.path.basename(compressed_filename)
        extension = basename.split('.', 1)[1]

    if 'zip' in extension:
        with zipfile.ZipFile(compressed_filename, "r") as zip_:
            zip_.extractall(directory)
    elif 'tar.gz' in extension or 'tgz' in extension:
        # `tar` is much faster than python's `tarfile` implementation
        subprocess.call(['tar', '-C', directory, '-zxvf', compressed_filename])
    elif 'tar' in extension:
        subprocess.call(['tar', '-C', directory, '-xvf', compressed_filename])

    logger.info('Extracted {}'.format(compressed_filename))

# Source: https://analyticsindiamag.com/guide-to-imdb-movie-dataset-with-python-implementation/
# Note to self: Slightly modified
def imdb_dataset(directory=os.path.join('data', 'raw'),
                 train=False,
                 test=False,
                 train_directory='train',
                 test_directory='test',
                 extracted_name='aclImdb',
                 sentiments=['pos', 'neg']):
    x= []
    splits = [
        dir_ for (requested, dir_) in [(train, train_directory), (test, test_directory)]
        if requested
    ]
    for split_directory in splits:
        full_path = os.path.join(directory, extracted_name, split_directory)
        examples = []
        for sentiment in sentiments:
            for filename in glob.iglob(os.path.join(full_path, sentiment, '*.txt')):
                with open(filename, 'r', encoding="utf-8") as f:
                    textnew = f.readline()
                examples.append({
                    'text': textnew ,
                    'sentiment': sentiment,
                })
        x.append(examples)
    if len(x) == 1:
        return x[0]
    else:
        return tuple(x)
        

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
