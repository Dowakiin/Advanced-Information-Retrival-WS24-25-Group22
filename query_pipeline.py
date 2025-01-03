import pandas as pd
import os
import json
from tfidf_processing import TfidfPipeline
from bert_processing import BertPipeline
from word2vec_processing import Word2VecPipeline


class RecipeQueryPipeline:
    def __init__(self, dataset_path, config=None):
        self.dataset_path = dataset_path
        self.config = config or {}
        self.tfidf_path = self.config.get("tfidf_path", "./models/tfidf.pkl")
        self.bert_path = self.config.get("bert_path", "./models/bert.pkl")
        self.word2vec_path = self.config.get("word2vec_path", "./models/word2vec.pkl")
        self.results_csv_path = self.config.get("results_csv_path", "./query_results.csv")

        self.tfidf = self.__initialize_pipeline(TfidfPipeline, self.tfidf_path)
        self.word2vec = self.__initialize_pipeline(Word2VecPipeline, self.word2vec_path)
        self.bert = self.__initialize_pipeline(BertPipeline, self.bert_path)

    def __initialize_pipeline(self, pipeline_class, path):
        if os.path.exists(path):
            return pipeline_class.load(path)
        else:
            data = pd.read_csv(self.dataset_path)
            pipeline = pipeline_class()
            pipeline.fit(data)
            pipeline.save(path)
            return pipeline

    def query_all(self, queries_path, top_n=7):
        with open(queries_path, "r") as file:
            queries_data = json.load(file)

        results = []
        for difficulty, queries in queries_data.items():
            for method, pipeline in [("tfidf", self.tfidf), ("word2vec", self.word2vec), ("bert", self.bert)]:
                for query_text in queries:
                    print(f"Processing {difficulty} query for {method}: {query_text}")
                    query_results = self.__run_query(pipeline, query_text, top_n)
                    results.append({
                        "difficulty": difficulty,
                        "method": method,
                        "query": query_text,
                        "results": query_results
                    })

        self.__save_results(results)

    def __run_query(self, pipeline, query_text, top_n):
        results = []
        pipeline_results, scores = pipeline.query(query=query_text, nr_results=top_n)
        for index, row in enumerate(pipeline_results.iterrows()):
            results.append({"recipe": row[1]["Merged"], "score": scores[index]})
        return results

    def __save_results(self, results):
        with open(self.results_csv_path, "w") as file:
            for difficulty in ["easy", "medium", "hard"]:
                for method in ["tfidf", "word2vec", "bert"]:
                    file.write(f"{method} ({difficulty})\n")
                    for result in filter(lambda r: r["difficulty"] == difficulty and r["method"] == method, results):
                        file.write(f"Query: {result['query']}\n")
                        for i, recipe in enumerate(result['results']):
                            file.write(f"Recipe {i + 1}: {recipe['recipe']} (Score: {recipe['score']:.4f})\n")
                        file.write("\n")


if __name__ == "__main__":
    config = {
        "tfidf_path": "./models/tfidf.pkl",
        "bert_path": "./models/bert.pkl",
        "word2vec_path": "./models/word2vec.pkl",
        "results_csv_path": "./query_results.csv",
    }

    pipeline = RecipeQueryPipeline(dataset_path="dataset/cleaned_recipenlg.csv", config=config)
    pipeline.query_all(queries_path="./queries.json", top_n=7)
