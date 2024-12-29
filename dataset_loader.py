import pandas as pd
from datasets import load_dataset
import re
import os


def clean_text(text):
    if not isinstance(text, str):
        if isinstance(text, (list, tuple)):
            text = " ".join(text)
        else:
            text = str(text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip().lower()
    return text


def get_dataset_path(dataset_name="preprocessed_recipenlg.csv", data_dir="dataset"):
    file_path = os.path.join(data_dir, dataset_name)

    return file_path


def load_preprocessed_dataset(dataset_name="preprocessed_recipenlg.csv", data_dir="dataset"):
    file_path = os.path.join(data_dir, dataset_name)

    return pd.read_csv(file_path)


def load_and_prepare_dataset(dataset_name="recipe_nlg", data_dir="~/full_data", save_path="cleaned_recipenlg.csv"):
    """
    Loads the manually downloaded RecipeNLG dataset, preprocesses it, and saves the cleaned dataset.
    """
    print("Loading dataset...")
    dataset = load_dataset(dataset_name, data_dir=data_dir)

    data = dataset["train"].to_pandas()

    print(f"Available columns: {list(data.columns)}")

    print("Cleaning data...")
    if "ingredients" in data.columns:
        print("Cleaning ingredients...")
        data["ingredients"] = data["ingredients"].apply(clean_text)

    if "directions" in data.columns:
        print("Cleaning directions...")
        data["instructions"] = data["directions"].apply(clean_text)
        data.drop(columns=["directions"], inplace=True)

    if "title" in data.columns:
        print("Cleaning title...")
        data["title"] = data["title"].apply(clean_text)

    print(f"Saving cleaned dataset to {save_path}...")
    data.to_csv(save_path, index=False)

    print("Dataset preparation completed!")


if __name__ == "__main__":
    base_directory = "dataset"
    os.makedirs(base_directory, exist_ok=True)

    raw_data_directory = os.path.join(base_directory, "full_data")
    os.makedirs(raw_data_directory, exist_ok=True)

    save_file = os.path.join(base_directory, "cleaned_recipenlg.csv")

    load_and_prepare_dataset(data_dir=raw_data_directory, save_path=save_file)
