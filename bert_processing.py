import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, set_seed
from datasets import Dataset
import pickle

import torch as th


class BertPipeline:
    def __init__(self, config: dict = None):
        # model_name = 'google-bert/bert-base-uncased'
        model_name = 'alexdseo/RecipeBERT'
        # self.embedding_pipeline = pipeline('feature-extraction', model=model_name, framework='pt', tokenizer="bert-base-uncased")
        self.embedding_pipeline = pipeline('feature-extraction', model=model_name, framework='pt')
        self.batch_size = config['batch_size'] if config and 'batch_size' in config else 5

    def fit(self, data: pd.DataFrame):
        self.data = data
        dataset = Dataset.from_pandas(data[['Merged']])

        dataset = dataset.map(self.__get_embeddings,
                              batched=True,
                              batch_size=self.batch_size)

        self.embeddings = th.tensor(dataset['embeddings'])

    def query(self, query: str, nr_results: int = 5):
        query_embedding = self.__get_query_embeddings([query])[0].reshape(1, -1)

        similarities = cosine_similarity(query_embedding, self.embeddings).flatten()

        top_indices = similarities.argsort()[-nr_results:][::-1]
        results, scores = self.data.iloc[top_indices], similarities[top_indices]

        for i, (index, score) in enumerate(zip(results.index, scores)):
            print(f"{i + 1}. {results.loc[index, 'Merged']} (Score: {score:.4f})")

        return results, scores

    def save(self, path: str):
        with open(path, 'wb') as file:
            pickle.dump(self, file=file)

    @staticmethod
    def load(path: str):
        with open(path, 'rb') as file:
            obj = pickle.load(file)
        return obj

    # def __get_embeddings(self, text_list):
    #     embeddings = self.embedding_pipeline(
    #         text_list,
    #         return_tensors='pt',
    #         batch_size=self.batch_size,
    #         truncation=True
    #     )
    #     ret = []

    #     for emb in embeddings:
    #         ret.append(emb[0].mean(axis=0))
    #     return ret

    def __get_embeddings(self, batch):
        embeddings = self.embedding_pipeline(
            batch['Merged'],
            return_tensors='pt',
            batch_size=self.batch_size,
            truncation=True
        )
        return {
            "embeddings": [
                emb[0].mean(dim=0).numpy() for emb in embeddings
            ]
        }
        ret = []

        for emb in embeddings:
            print('test')
            ret.append(emb[0].mean(axis=0))
        return ret

    def __get_query_embeddings(self, text_list):
        embeddings = self.embedding_pipeline(
            text_list,
            return_tensors='pt',
            batch_size=self.batch_size,
            truncation=True
        )
        ret = []

        for emb in embeddings:
            ret.append(emb[0].mean(axis=0))
        return ret


def create_bert_processing(data: pd.DataFrame, query_text):
    model = 'alexdseo/RecipeBERT'

    embedding = pipeline('feature-extraction', model=model, framework='pt')

    def get_embeddings(text_list, batch_size=5):
        print(f"list length {len(text_list)}")
        embeddings = embedding(text_list, return_tensors='pt', batch_size=batch_size, truncation=True)
        ret = []

        for emb in embeddings:
            ret.append(emb[0].mean(axis=0))

        return ret

    database_embeddings = get_embeddings(data["Merged"].tolist())
    query_embedding = get_embeddings([query_text])
    print(query_embedding[0].reshape(1, -1))

    similarities = cosine_similarity(query_embedding[0].reshape(1, -1), database_embeddings).flatten()

    top_n = 5
    top_indices = similarities.argsort()[-top_n:][::-1]
    results, scores = data.iloc[top_indices], similarities[top_indices]

    for i, (index, score) in enumerate(zip(results.index, scores)):
        print(f"{i + 1}. {results.loc[index, 'Merged']} (Score: {score:.4f})")

    return query_embedding, database_embeddings
