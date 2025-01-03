import pandas as pd
from datasets import load_dataset
import re
import os

def get_dataset_path(dataset_name="preprocessed_recipenlg.csv", data_dir="dataset"):
    file_path = os.path.join(data_dir, dataset_name)

    return file_path


def load_preprocessed_dataset(dataset_name="preprocessed_recipenlg.csv", data_dir="dataset"):
    file_path = os.path.join(data_dir, dataset_name)

    return pd.read_csv(file_path)

if __name__ == "__main__":
    base_directory = "dataset"
    os.makedirs(base_directory, exist_ok=True)

    raw_data_directory = os.path.join(base_directory, "full_data")
    os.makedirs(raw_data_directory, exist_ok=True)

    save_file = os.path.join(base_directory, "cleaned_recipenlg.csv")
