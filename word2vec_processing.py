import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize

import numpy as np

import spacy

import pickle


def prepare_text(text, model):
    vectors = [model.wv[word] for word in text if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)


class Word2VecPipeline:
    def __init__(self, config: dict = None):
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    def fit(self, data: pd.DataFrame):
        docs = self.nlp.pipe(data['Merged'], n_process=4, batch_size=100)
        tokenized_recipes = [[token.text for token in doc] for doc in docs]
        self.model = Word2Vec(tokenized_recipes, vector_size=256, window=8, min_count=1, workers=4)
        self.embeddings = [prepare_text(recipe, self.model) for recipe in tokenized_recipes]
        self.data = data

    def query(self, query: str, nr_results: int = 5):
        query_tokens = [token.text for token in self.nlp(query)]
        query_vector = prepare_text(query_tokens, self.model)

        similarities = cosine_similarity(query_vector.reshape(1, -1), self.embeddings).flatten()

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


def prepare_text(text, model):
    vectors = [model.wv[word] for word in text if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)


def create_word2vec_model(data: pd.DataFrame, query_text):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    docs = nlp.pipe(data['Merged'], n_process=4, batch_size=100)

    tokenized_recipes = [[token.text for token in doc] for doc in docs]

    model = Word2Vec(tokenized_recipes, vector_size=100, window=5, min_count=5, workers=4)

    recipe_embeddings = [prepare_text(recipe, model) for recipe in tokenized_recipes]

    query_token = word_tokenize(query_text)

    query_vector = prepare_text(query_token, model)
    similarities = cosine_similarity(query_vector.reshape(1, -1), recipe_embeddings).flatten()

    # Find top-n most similar sentences for the first sentence
    top_n = 5
    top_indices = similarities.argsort()[-top_n:][::-1]
    results, scores = data.iloc[top_indices], similarities[top_indices]

    for i, (index, score) in enumerate(zip(results.index, scores)):
        print(f"{i + 1}. {results.loc[index, 'Merged']} (Score: {score:.4f})")
