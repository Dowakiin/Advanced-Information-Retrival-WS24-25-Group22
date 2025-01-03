import pandas as pd

import re
import os


def clean_text(text):
    if not isinstance(text, str):
        if isinstance(text, (list, tuple)):
            text = " ".join(text)
        else:
            text = str(text)
    text = re.sub(r"[\[\]{}()]", "", text)
    return text


def preprocess_dataset(dataset_name="full_dataset.csv", data_dir="~/full_data", save_path="preprocessed_recipenlg.csv"):
    dataset_path = os.path.join(data_dir, dataset_name)

    data = pd.read_csv(dataset_path)

    columns_to_drop = ['link', 'source', 'NER', 'Unnamed: 0']
    print(data.columns)
    data = data.drop(columns=columns_to_drop)

    data['Merged'] = data.apply(
        lambda row: f"Title: {row['title']}; Ingredients: {row['ingredients']}; Directions: {row['directions']}",
        axis=1)
    # data['Merged'] = data.apply(lambda row: f"Title: {row['title']} {', '.join(map(str, row['ingredients']))}", axis=1)
    # data['Merged'] = data.apply(clean_text)

    columns_to_drop = ['title', 'ingredients', 'directions']
    data = data.drop(columns=columns_to_drop)

    data.to_csv(save_path, index=False)

    return data


def create_mini_dataset(data: pd.DataFrame, save_path, nr_rows=100_000):
    data = data.head(nr_rows)
    data.to_csv(save_path, index=False)


if __name__ == "__main__":
    base_directory = "dataset"
    os.makedirs(base_directory, exist_ok=True)

    raw_data_directory = os.path.join(base_directory, "full_data")
    os.makedirs(raw_data_directory, exist_ok=True)

    save_file = os.path.join(base_directory, "preprocessed_recipenlg.csv")
    save_file_mini = os.path.join(base_directory, "preprocessed_mini_recipenlg.csv")

    data = preprocess_dataset(data_dir=raw_data_directory, save_path=save_file)
    create_mini_dataset(data, save_path=save_file_mini)
