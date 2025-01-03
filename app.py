from tfidf_processing import TfidfPipeline
from word2vec_processing import Word2VecPipeline, create_word2vec_model
from bert_processing import BertPipeline, create_bert_processing

from dataset_loader import load_preprocessed_dataset, get_dataset_path
from dataset_preprocesor import create_mini_dataset, preprocess_dataset, create_mini_dataset

from query_pipeline import RecipeQueryPipeline

import numpy as np
import sys
import os

import torch as th
from transformers import set_seed


def prototyping():
    """
    Function for testing various configurations, models and queries by hand
    There are also functions to save the fitted models to save computation time later
    """

    # set to True to use a smaller data set
    use_mini = False
    if use_mini:
        data_set_name = 'preprocessed_mini_recipenlg.csv'
        data = load_preprocessed_dataset(dataset_name=data_set_name)
    else:
        data = load_preprocessed_dataset()

    tfid_path = './models/tfidf.pkl'
    bert_path = './models/bert.pkl'
    word2vec_path = './models/word2vec.pkl'

    if os.path.exists(tfid_path):
        print('Loading existing tfidf model')
        tfidf = TfidfPipeline.load(tfid_path)
    else:
        tfidf = TfidfPipeline()
        tfidf.fit(data)
        tfidf.save(tfid_path)

    # query_emb , db_emb = create_bert_processing(data, 'Mix sugar, butter and peanut butter.", "Roll into balls and place on cookie sheet.", "Set in freezer for at least 30 minutes. Melt chocolate chips and paraffin in double boiler.", "Using a toothpick, dip balls 3/4 of way into chocolate chip and paraffin mixture to make them look like buckeyes."')

    if os.path.exists(word2vec_path):
        print('Loading existing w2v model')
        word2vec = Word2VecPipeline.load(word2vec_path)
    else:
        print('Creating w2v model')
        word2vec = Word2VecPipeline()
        word2vec.fit(data)
        word2vec.save(word2vec_path)

    if os.path.exists(bert_path):
        print('Loading existing bert model')
        bert = BertPipeline.load(bert_path)
    else:
        print('Creating bert model')
        bert = BertPipeline()
        bert.fit(data)
        bert.save(bert_path)

    query = 'cookies'

    print("-" * 30)
    tfidf.query(query)
    print("-" * 30)

    print("-" * 30)
    word2vec.query(query)
    print("-" * 30)

    bert.query(query)
    print("-" * 30)


def create_dataset(sub_set_size: int):
    """
    Creates a subset of the original dataset with 100,000 entries by default (saves computation time).
    """
    if not os.path.exists('./dataset/preprocessed_recipenlg.csv'):
        print("Preprocessing Dataset")
        data = preprocess_dataset(dataset_name="full_dataset.csv", data_dir="./dataset/full_data", save_path="./dataset/preprocessed_recipenlg.csv")
        data = load_preprocessed_dataset("preprocessed_recipenlg.csv", "./dataset")
        print("Creating mini dataset")
        create_mini_dataset(data, './dataset/preprocessed_final_recipenlg.csv', sub_set_size)


def perform_queries():
    """
    Sets all data paths and starts the query pipeline
    """
    data_set_name = 'preprocessed_final_recipenlg.csv'
    dataset_path = get_dataset_path(dataset_name=data_set_name)

    model_dir = f"./models/"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    config = {
        "tfidf_path": f"{model_dir}tfidf.pkl",
        "bert_path": f"{model_dir}bert.pkl",
        "word2vec_path": f"{model_dir}word2vec.pkl",
        "results_csv_path": "./query_results.csv",
    }

    pipeline = RecipeQueryPipeline(dataset_path, config=config)
    pipeline.query_all('./queries.json')


def main(seed: int = 42, test: bool = False, sub_set_size: int = 100_000):
    """
    Sets the seed value for the random number generator and runs queries against the dataset.
    And starts either prototyping or performs queries.
    :param seed: the seed value for the random number generator
    :param test: flag to run prototyping mode
    :param sub_set_size: sets the size of the subset used
    """
    set_seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    print("Creating Dataset")
    create_dataset(sub_set_size)

    if test:
        prototyping()
    else:
        perform_queries()


if __name__ == '__main__':
    """
    Entry function that handles command line arguments. 0 to 3 arguments can be passed.
    :arg 1: enter any integer to set a custom seed for reproducibility (default 42)
    :arg 2: enter any integer to only use a subset of the original dataset (default 100,000)
    :arg 3: enter 'test' to set the program mode to prototype (default perform queries)
    """
    args = sys.argv

    if len(args) == 2:
        seed_arg: int = int(sys.argv[1])
        main(seed_arg)

    elif len(args) == 3:
        seed_arg: int = int(sys.argv[1])
        subset_arg: int = int(sys.argv[2])
        main(seed_arg, False, subset_arg)

    elif len(args) == 4:
        seed_arg: int = int(sys.argv[1])
        subset_arg: int = int(sys.argv[2])
        test_arg: str = sys.argv[3]
        if test_arg == 'test':
            main(seed_arg, True, subset_arg)
        else:
            main(seed_arg, False, subset_arg)

    else:
        main()
