# A Comparison of Classical and Modern Information Retrieval Approaches on Recipes
___
This is a student project for the lecture "Advanced Information Retrieval" (Lecture Number: 706.705) taught at the Technical University Graz, winter semester 2024/25.

##### Group Organisation
Members and roles of **group 22** are:
- Markus Auer-Jammerbund
  - Query pipeline setup
  - Evaluation
  - Report
- Thomas Knoll
  - Design Document
  - _Bert_ embeddings
  - Evaluation
  - Report
- Jonas Pfisterer
  - _Word2Vec_ embeddings
  - Evaluation
  - Report
- Thomas Puchleitner
  - Design Document
  - _TF-IDF_ handling
  - Result processing
  - Evaluation
  - Report

##### External Datasets and Models
The used dataset for the recipes is from [Hugging Face](https://huggingface.co): [mbien/recipe_nlg](https://huggingface.co/datasets/mbien/recipe_nlg) \
The _Bert_ model is also from Hugging Face: [alexdseo/RecipeBERT](https://huggingface.co/alexdseo/RecipeBERT)

##### Additional Files
- [Design Document](Design/AirDesignDocumentGroup22.pdf)
- Evaluation (as [.pdf](Evaluation/Evaluation.pdf) and [.csv](Evaluation/Evaluation.csv); or online as [google doc](https://docs.google.com/spreadsheets/d/12DoSQCYWASj7j5J2dj4g4nOa0l75Q9l1k2vbwn6Z5d0/edit?usp=sharing))
- [Report](Report_Group_22.pdf)
- [Presentation Slides]() TODO

### Abstract
___

### Project Structure
___
##### Starting the Program
The program entrypoint is in [app.py](app.py). Therefore, to start the program, navigate to the root directory of this project and type the following command into the terminal:

``` bash
    python3 app.py
```

There are up to three commandline arguments that can be used to modify the programs' behaviour (not all arguments have to be used). 
The first one sets the seed for randomisation. By default, it is set to 42. 
With the second argument it can be chosen how many entries of the dataset should be used. 
Because the dataset is quite large (2 million entries), it might be suitable (especially for testing) to use only a smaller subset of the data. 
Thus, the default value is set to 100,000.
As the third argument, "test" can be written to start the program in prototyping mode. 
This mode was used for various testing and pre modeling during development such that not the entire code had to be rerun every time something new was tested.
For example, if one would want to start the program with the seed "123", and use 1,000,000 entries from the data set, they would have to type the following into the terminal:

``` bash
    python3 app.py 123 1000000
```

If they wanted to test something out with those parameters, the command would be:

``` bash
    python3 app.py 123 1000000 test
```

##### File Structure
As mentioned above, [app.py](app.py) provides the entry point. 
Here, during normal start up so not in prototyping mode, the commandline arguments are handled, the data loaded and the query pipeline initialised and started.
Data preprocessing and loading are done in [dataset_preprocessor.py](dataset_preprocesor.py) and [dataset_loader.py](dataset_loader.py) respectively.
The query pipeline is programmed in [query_pipeline.py](query_pipeline.py). 
There, the given queries are run through the pipelines for _TF-IDF_, _Word2Vec_, and _Bert_ ([tfidf_processing.py](tfidf_processing.py), [word2vec_processing.py](word2vec_processing.py), and [bert_processing.py](bert_processing.py) respectively).
[print_debug_cleaned_data.py](print_debug_cleaned_data.py) only contains a print function for debugging. 