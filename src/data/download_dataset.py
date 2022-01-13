import os
from datasets import load_dataset


# Download the raw datasets and save to "data/raw"
def download_dataset(name='imdb', save_path='data/raw'):
    if os.path.exists(os.path.join(save_path, 'dataset_dict.json')):
        print("Already downloaded!")
    else:
        dataset_dict = load_dataset(name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        dataset_dict.save_to_disk(save_path)

if __name__ == "__main__":
    download_dataset('imdb', 'data/raw')