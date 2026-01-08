#!/usr/bin/env python
# coding: utf-8

import numpy as np

from sklearn.feature_extraction import DictVectorizer


def binary(string):
    '''
    returns 1 or 0 for a string input of TRUE or FALSE
    :param string: any string input
    :type token: string
    
    :returns binary integer 1 or 0 if the string is TRUE or FALSE respectively, otherwise returns the string
    '''
    if string == 'TRUE':
        return 1
    elif string == 'FALSE':
        return 0
    else:
        return string
    
def feature_to_index(feature):
    '''
    gets feature index from training conll file
    
    :param feature: feature name
    :type feature: string
    
    :returns: integer
    '''
    with open('../data/conll2003.train.preprocessed.conll','r',encoding='utf8') as infile:
        
        header = infile.readline() # get header as it contains feature names
        # insert 'Id' in 0 position of to align with post-processing conll file
        features = ['Id'] + header.rstrip('\n').split()
        
        feature_index = features.index(feature)
        
    return feature_index


def get_feature_dict(row,feature_list):
    '''
    extracts feature value pairs from row
    
    :param row: row from conll file
    :param selected_features: list of selected features
    :type row: string
    :type selected_features: list of strings
    
    :returns: dictionary of feature value pairs
    '''
    feature_dict = {}
    
    for feature in feature_list:
        i = feature_to_index(feature)
        feature_dict[feature] = binary(row[i]) # return 1 or 0 for binary features, string otherwise
        
    return feature_dict

def features(inputfile,feature_list):
    '''
    extracts the features from a given development file
    
    :param inputfile: path to the development file
    :param feature_list: list of selected features
    :type inputfile: string
    :type feature_list: list of strings
    
    :returns a list of feature dictionaries for each datapoint in the trainingfile
    '''
    data = []
    
    with open(inputfile,'r',encoding='utf8') as infile:
        
        header = infile.readline() # do not iterate over header
        
        for line in infile:
            components = line.rstrip('\n').split()
            
            if len(components) > 0:
                                
                feature_dict = get_feature_dict(components,feature_list)
                data.append(feature_dict) # one dictionary per one data instance
    
    return data

def features_and_labels(inputfile,feature_list):
    '''
    extracts the features and gold labels from a given training file
    
    :param trainingfile: path to the training file
    :param feature_list: list of selected features
    :type trainingfile: string
    :type feature_list: list of strings
    
    :return data: a list of feature dictionaries for each datapoint in the trainingfile
    :return targets: a list of the gold labels
    '''
    data = []
    targets = []
    
    with open(inputfile,'r',encoding='utf8') as infile:
        header = infile.readline() # do not iterate over header
        
        for line in infile:
            components = line.rstrip('\n').split()
            
            if len(components) > 0:
                
                feature_dict = get_feature_dict(components,feature_list)
                data.append(feature_dict) # one dictionary per one data instance
                                
                #gold is in the 4th column, since this conll format has been modified
                targets.append(components[4])
                
    return data, targets

def extract_word_embedding(token,word_embedding_model):
    '''
    returns the word embedding for a given token out of a distributional semantic model,
    and a 300-dimension vector of 0s otherwise
    
    :param token: the token
    :param word_embedding_model: the distributional semantic model
    :type token: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors
    
    :returns a vector representation of the token
    '''
    if token in word_embedding_model:
        vector = word_embedding_model[token]
    else:
        vector = [0]*300
        
    return vector

def combine_sparse_and_dense_features(dense_vectors,sparse_features):
    '''
    takes sparse and dense feature representations and appends their vector representation
    
    :param dense_vectors: list of dense vector representations
    :param sparse_features: list of sparse vector representations
    :type dense_vector: list of arrays
    :type sparse_features: list of lists
    
    :returns: list of arrays in which sparse and dense vectors are concatenated
    '''
    combined_vectors = []
    sparse_vectors = np.array(sparse_features.toarray())
    
    for index,vector in enumerate(sparse_vectors):
        
        combined_vector = np.concatenate((vector,dense_vectors[index]))
        combined_vectors.append(combined_vector)
        
    return combined_vectors

def embeddings_as_features(inputfile,word_embedding_model,get_gold=True):
    '''
    extracts features using word embeddings from development file
    
    :param inputfile: path to inputfile file
    :param word_embedding_model: a pretrained word embedding model
    :type inputfile: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors
    
    :return features: list of vector representation of tokens
    :return labels: list of gold labels
    '''
    features = []
    labels = []
        
    with open(inputfile,'r',encoding='utf8') as infile:
       
        header = infile.readline() # do not iterate over header
        
        for line in infile:
            row = line.rstrip('\n').split()
            
            if len(row) > 0: # check for cases where empty lines mark sentence boundaries
                
                vector = extract_word_embedding(row[1],word_embedding_model)
                features.append(vector)
                labels.append(row[4]) # gold is in the 4th column
                
    if get_gold:
        return_value = features, labels
    else:
        return_value = features  # when extracting features from testfile, gold is not needed
    
    return return_value
                
def combined_features(inputfile,word_embedding_model,selected_features,
                              get_gold_and_vec=True,vectorizer=None):
    '''
    extracts traditional features as well as embeddings and gold labels
    
    :param inputfile: path to inputfile file
    :param word_embedding_model: a pretrained word embedding model
    :param selected_features: a list of selected features
    :param get_gold_and_vec: if True, returns also vectorizer and gold labels
    :param vectorizer: if None, initializes vectorizer and fits with the traditional training features
    :type inputfile: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors
    :type selected_features: list of strings
    :param get_gold_and_vec: bool
    
    :return features: list of vector representation of tokens
    :return labels: list of gold labels
    '''
    traditional_features = []
    dense_vectors = []
    labels = []
        
    with open(inputfile,'r',encoding='utf8') as infile:
               
        header = infile.readline() # do not iterate over header
        
        for line in infile:
            row = line.rstrip('\n').split()
            
            if len(row) > 0: # check for cases where empty lines mark sentence boundaries
                
                token_vector = extract_word_embedding(row[1],word_embedding_model)
                dense_vectors.append(token_vector)
    
                other_features = get_feature_dict(row,selected_features)
                traditional_features.append(other_features)
            
                labels.append(row[4]) # gold is n the 4th column
                
    # vec not provided when extracting from training data, it therefore needs to be created and fitted
    if vectorizer is None:
        vectorizer = DictVectorizer()
        vectorizer.fit(traditional_features) 
            
    sparse_features = vectorizer.transform(traditional_features)       
    combined_vectors = combine_sparse_and_dense_features(dense_vectors,sparse_features)
        
    if get_gold_and_vec:
        return_value = combined_vectors, labels, vec
    else:
        return_value = combined_vectors # when extracting features from testfile, gold and vec is not needed
    
    return return_value
