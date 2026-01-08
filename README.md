# NERC
NER classification pipeline for the fall 2023 VU course 'Machine Learning in NLP'.

# ma-ml4nlp-labs

## overview

This repository provides notebooks and scripts for the course 'Machine Learning in NLP'.

IMPORTANT: In order for the notebooks to work, please create the following directories and place them at the same level as the \code directory.

/data 

should contain:

1. conll2003.train.conll
2. conll2003.test.conll
3. empty file named 'conll2003.test.output.conll'

/models 

should be empty. Will have trained models added to it when running the notebooks.

/word-embeddings

should contain: GoogleNews-vectors-negative300

/code

Here you can find all the scripts for the NERC project. It contains the following contents:

Jupyter notebooks

1. data_analysis.ipynb: contains scripts for analysing the contents of the CONLL2003 files and extracting initial insights.

2. preprocessing.ipynb: contains scripts for preprocessing the data. It takes the original train, dev and test sets and processes them to include feature-values in the CONLL columns. It takes conll files from \data and creates new conll files which contain the affix 'preprocesses' in the filename. After running the notebook, the new files will also exist in \data. 

IMPORTANT! this notebook must be run for the following scripts to work. They will not work without it!

3. basic_system.ipynb: trains a LogisticRegression classifier with baseline features and saves it. It then loads the model and classifies the test-set with the pretrained model, and prints evaluation.

4. expanded_system.ipynb: trains three classifiers: LogisticRegression, NaiveBayes and LinearSVC, using an extended feature set, and saves the models. It then loads the models and classifies the test-set with the pretrained models, and prints evaluation.

5. hyperparamter_tuning.ipynb: conducts hyperparamater tuning on the LinearSVC model with the extended feature set, and saves it. It then loads the model and classifies the test-set with the pretrained model, and prints evaluation.

6. word_embeddings.ipynb: creates a word embedding model from the path in /word-embeddings. It then trains two systems: one is a LinearSVC classifier trained on word embeddings as features, extracted using the word-embedding model. The other is a LinearSVC classifier trained on combined word embeddings and one-hot representations of the baseline features. It saves both models, and then loads them, classifies the test-set with the pretrained models, and prints evaluation.


7. feature_ablation.ipynb: conducts feature ablation on the LinearSVC classifier, testing different feature combinations, and trains the LogisticRegression and NaiveBayes on the best feature set. It saves all models to be loaded, used to classify the test-sets and evaluating the results.

8. evaluation.ipynb: a short script that provides my own code for calculating precision, recall and f1-score. This is not used anywhere, this was just for practice. This notebook is executable. 

9. error_analysis.ipynb: contains script for printing classification reports and confusion matrices for the two best-performing systems, as well as some code for extracting candidates for examination while conduction error analysis.

python scripts

1. feature_extraction_util.py: utils for feature extraction. this script is imported in all systems that train a model.

2. classification_util.py: utils for classification. this script is imported in all systems that load a pretrained model and use it to classify a test-set.

3. evalation_util.py: utils for evaluation. this script is imported in all systems that retrieve classification results.
