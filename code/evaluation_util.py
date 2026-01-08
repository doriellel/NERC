#!/usr/bin/env python
# coding: utf-8

import pandas as pd

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

def get_gold_and_predictions(outputdata,exclude_majority):
    '''
    gets an array of gold labels and system predictions given a classified dataset
    :param outputdata: path to the file containing the classified dataset
    :param exclude_majority: if specified as True, then the majority class 'O' is removed
    :type outputdata: string
    :type exclude_majority: bool

    returns an array of gold labels and an array of system predictions
    '''
    df = pd.read_csv(outputdata,sep='\t',names=['Token','POS','Chunk','Gold','Allcaps',
                                                 'Cap_after_lower','Demonym','Comp_suf',
                                                 'Poss_mark','System labels'])
    df = df.dropna() # drop rows that contains nan values
    
    if exclude_majority:
        df = df[df['Gold'] != 'O'] # to remove category 'O' from evaluation
    
    gold = df['Gold']
    predictions = df['System labels']
    
    return gold, predictions
    
def get_label_set(gold,predictions):
    '''
    creates a sorted, alphabetized list of the unique labels given gold labels and system predictions
    :param gold: a list of gold labels
    :param predictions: a list of prediction labels
    :param exclude_majority: if specified as True, then the majority class 'O' is removed
    :type gold: array
    :type predictions: array
    :type exclude_majority: bool

    returns a list of labels 
    '''
    labels = list(gold) + list(predictions)
    label_set = sorted(set(labels))

    return label_set
                
def get_confusion_matrix(outputdata,exclude_majority):
    '''
    creates a confusion matrix and display given a path to a classified dataset with gold labels and predictions
    :param outputdata: a path to the file containing the classified dataset
    :param exclude_majority: if specified as True, then the majority class 'O' is removed
    :type outputdata: string
    :type exclude_majority: bool

    returns a confusion matrix and a ConfusionMatrixDisplay object 
    '''
    gold, predictions = get_gold_and_predictions(outputdata,exclude_majority)
    label_set = get_label_set(gold,predictions)

    matrix = confusion_matrix(gold,predictions)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix,display_labels=label_set)
    
    return matrix, display

def get_classification_report(outputdata,exclude_majority):
    '''
    creates a classification report given a path to a classified dataset with gold labels and predictions
    
    :param outputdata: a path to the file containing the classified dataset
    :param exclude_majority: if specified as True, then the majority class 'O' is removed
    :type outputdata: string
    :type exclude_majority: bool

    returns a classification report containing precision, recall and f1-score metrics for each label, 
    as well as macro and weighted averages for each metric and an overall accuracy score
    '''
    gold, predictions = get_gold_and_predictions(outputdata,exclude_majority)
    label_set = get_label_set(gold,predictions)

    report = classification_report(gold,predictions,digits=7,target_names=label_set)
    
    return report

def get_confusion_matrix_and_classification_report(outputdata,exclude_majority=False):
    '''
    prints the classification report and the confusion matrix
    
    :param outputdata: a path to the file containing the classified dataset
    :param exclude_majority: if specified as True, then the majority class 'O' is removed
    :type outputdata: string
    :type exclude_majority: bool

    '''
    # display classification report
    report = get_classification_report(outputdata,exclude_majority)
    print(report)

    # display confusion matrix plot
    matrix, display = get_confusion_matrix(outputdata,exclude_majority)
    display.plot()
