#!/usr/bin/env python
# coding: utf-8

import feature_extraction_util as extract

from scipy.stats import uniform
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer
from sklearn.experimental import enable_halving_search_cv  # explicitly require this experimental feature - noqa
from sklearn.model_selection import HalvingRandomSearchCV


def hyperparameter_tuning(train_features, train_targets):
        
    # define the parameter distributions
    param_distributions = {
        'C': uniform(0.1, 10),  # regularization parameter
        'loss': ['hinge', 'squared_hinge'],  # loss function
        'tol': [1e-4, 1e-3, 1e-2],  # tolerance for stopping criteria
    }
    
    # convert features to numeric representation
    vec = DictVectorizer()
    features_vectorized = vec.fit_transform(train_features)
    
    svm_model = svm.LinearSVC(max_iter=20000)  # increase number of iterations due to converging fail

    # create the HalvingRandomSearchCV object
    halving_random_search = HalvingRandomSearchCV(svm_model, param_distributions,
                                                  factor=2, resource='n_samples',
                                                  max_resources=200, random_state=42)
    
    # fit the HalvingRandomSearch model with the data
    halving_random_search.fit(features_vectorized, train_targets)

    # get the best hyperparameters
    best_params = halving_random_search.best_params_

    return best_params

def create_classifier(train_features,train_targets,modelname):
    '''
    given modelname, creates a classifier from features represented as vectors and gold labels
    
    :param train_features: a list of feature dictionaries
    :param train_targets: list of gold labels
    :param modelname: name of the classifier
    :type train_features: list of vectors
    :type train_targets: list of strings
    :type modelname: string
    
    '''
    if modelname ==  'logreg':
        model = LogisticRegression(solver='lbfgs',max_iter=600)
    elif modelname == 'NB':
        model = MultinomialNB()
    elif modelname == 'SVM':
        model = svm.LinearSVC()
        
    # transform the feature dictionaries to vector representation
    vec = DictVectorizer()
    features_vectorized = vec.fit_transform(train_features)
    
    # train the classifier 
    model = model.fit(features_vectorized,train_targets)
        
    return model, vec

def create_embedding_classifier(train_features,train_targets):
    '''
    creates an svm classifier from word embeddings and gold labels
    
    :param train_features: list of vector representations of features
    :param train_targets: list of gold labels
    :type train_features: list of vectors
    :type train_targets: list of strings
    
    :return model: trained SVM model
    '''
    # create an instance of an SVM model and train it with the word embedding features and gold labels
    # no need to transform since the word embeddings are already in vector representation
    model = svm.LinearSVC()
    model = model.fit(train_features,train_targets) 
    
    return model


def create_hypertuned_classifier(train_features, train_targets):
        
    # convert features to numeric representation
    vec = DictVectorizer()
    features_vectorized = vec.fit_transform(train_features)
    
    # get best hyperparameters and create a LinearSVC model 
    best_params = hyperparameter_tuning(train_features, train_targets)
    hypertuned_model = svm.LinearSVC(**best_params,max_iter=20000)

    # train the hypertuned model on the training set
    hypertuned_model.fit(features_vectorized, train_targets)

    return hypertuned_model, vec

def classify_data(model,vec,feature_list,inputfile,outputfile):
    '''
    classifies dataset given model, vector and features, and writes to outputfile
    
    :param model: a LogisticRegression, Naive Bayes or SVM model instance
    :param vec: a DictVectorizer() instance
    :param inputdata: path to development set
    :param outputfile: path to existing outputfile
    :type model: sklearn.linear_model._logistic.LogisticRegression or 
                 sklearn.naive_bayes.MultinomialNB or 
                 sklearn.svm._classes.LinearSVC
    :type vec:   sklearn.feature_extraction._dict_vectorizer.DictVectorizer
    :type inputdata: string
    :type outputfile: string 
    '''
    # get features from the development file and transform them into vector representation
    features = extract.features(inputfile,feature_list)
    vectorized_features = vec.transform(features)
    
    # generate predictions with trained model and write predictions to file
    predictions = model.predict(vectorized_features)
    write_predictions_to_outputfile(inputfile,outputfile,predictions)
        
    return model
                
def classify_data_given_features(features,model,inputfile,outputfile):
    '''
    classifies data using either just word embeddings, or combined word embeddings and traditional features
    writes the results into an output file
    
    :param features: list of vector representations of features
    :param classifier: pretrained SVM model
    :param inputfile: path to an input file
    :param outputfile: path to an output file
    :type features: list of vectors
    :type classifier: svm.LinearSVC()
    :type inputfile: string
    :type outputfile: string
    '''
    predictions = model.predict(features)
    write_predictions_to_outputfile(inputfile,outputfile,predictions)
    
def write_predictions_to_outputfile(inputfile,outputfile,predictions):
    '''
    writes predictions to outputfile
    
    :param inputfile: a path to the development file
    :param outputfile: a path to the output file
    :param predictions: the model's predictions
    :type inputfile: string
    :type inputfile: string
    :type predictions: array of predicted labels
    '''
    with open(inputfile,'r') as infile:

        header = infile.readline() # do not iterate over header
        counter = 0
        outfile = open(outputfile,'w')
        
        for line in infile:
            
            if len(line.rstrip('\n').split()) > 0:
                outfile.write(line.rstrip('\n') + '\t' + predictions[counter] + '\n')
                counter += 1

