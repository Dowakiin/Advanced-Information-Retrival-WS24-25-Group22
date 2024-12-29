import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pickle


class TfidfPipeline:
    def __init__(self, config: dict = None):

        if config:
            self.vectorizer = TfidfVectorizer(max_features=config['max_features'],
                                              stop_words=config['stop_words'])
        else:
            self.vectorizer = TfidfVectorizer(max_features=50_000,
                                              stop_words='english')

    def fit(self, data: pd.DataFrame):
        self.embeddings = self.vectorizer.fit_transform(data['Merged'])
        self.data = data

    def query(self, query: str, nr_results: int = 5):
        query_embedding = self.vectorizer.transform([query])
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
