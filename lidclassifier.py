#!/usr/bin/python3

# Design of utils file and Corpus class inspired by 
# Columbia University
# COMS 4705W 2017 
# Assignment 4
# Professor Karl Stratos

# Model design as described by Jaech et. al. in
# "A Neural Model for Language Identification in Code-Switched Tweets"


import argparse
import csv
import random
import pickle
import os

from math import ceil

import numpy as np

from keras.models import load_model

import utils

from utils import Corpus, Corpus_Aaron
from classifier import Classifier

def main(train_corpus_filepath, test_corpus_filepath, model, weigh_labels, epochs=20, batch_size=25):


    # Ingest the corpus
    train_corpus = Corpus_Aaron()
    train_corpus.read_corpus(train_corpus_filepath, dl='\t')
    char2idx, idx2char = train_corpus.create_dictionary()

    test_corpus = Corpus_Aaron(char_dictionary=(char2idx, idx2char),
            label_dictionary=(label2idx, idx2label))
    test_corpus.read_corpus(test_corpus_filepath, dl='\t')
    maxsentlen = max(train_corpus.maxsentlen, test_corpus.maxsentlen)
    maxwordlen = max(train_corpus.maxwordlen, test_corpus.maxwordlen)   

    train_sentences, train_labels, train_labels_weights = train_corpus.np_idx_conversion(
        maxsentlen, maxwordlen)

    test_sentences, test_labels, _ = test_corpus.np_idx_conversion(maxsentlen, 
        maxwordlen)

    label2idx = train_corpus.label2idx
    idx2label = train_corpus.idx2label

    utils.print_np_sentence(train_sentences[20], idx2char) 
    utils.print_np_label(train_labels[20], idx2label)
    print(train_labels_weights[20])

    '''
    if weigh_labels == True:
        label_weights = {idx : 1 if idx != 0 else 0 for _, idx in label2idx.items()}
    else: 
        label_weights = None
    '''

    #utils.print_np_sentences_np_labels(train_sentences, train_labels, train_idx2char, train_idx2label)

    num_labels = len(label2idx)

    # Build the model
    classifier = Classifier(char2idx, maxsentlen, maxwordlen, num_labels)
    
    if model != None:
        # Load the model
        classifier.model = load_model(model)
    else:
        # Train the model
        classifier.model.fit(x=train_sentences, y=train_labels,
            epochs=epochs, batch_size=batch_size, validation_split=.1, 
            sample_weight=train_labels_weights)
        # Save the model
        classifier.model.save('lid_classifier_model.h5')

    # Evaluate the model
    #evaluation = classifier.model.evaluate(x=test_sentences, y=test_cs, batch_size=batch_size)
    #print(evaluation)
    print("Testing on sentences of shape: " + str(test_sentences.shape))
    pred_labels = classifier.model.predict(x=test_sentences)
    #pred_labels = classifier.model.predict(x=train_sentences)
    # Print sentences, gold labels, and predicted labels
    utils.print_np_sentences_np_gold_pred_labels(test_sentences, test_labels, pred_labels, idx2char, idx2label)

    # Transform labels to represent category index
    test_cat_labels = np.argmax(test_labels, axis=2)
    #train_cat_labels = np.argmax(train_labels, axis=2)
    pred_cat_labels = np.argmax(pred_labels, axis=2)

    metrics = (utils.compute_accuracy_metrics(
            test_cat_labels, pred_cat_labels, label2idx))
    #metrics = (utils.compute_accuracy_metrics(
    #        train_cat_labels, pred_cat_labels, label2idx))

    for metric in metrics:
        if metric == 'confusion_matrix':
            continue
        print(metric + ": " + str(metrics[metric]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A neural network based'
        + 'classifier for detecting code switching.') 
    parser.add_argument('train_corpus_filepath', metavar='TrC', type=str,
            help='Filepath to the training corpus.')
    parser.add_argument('test_corpus_filepath', metavar='TeC', type=str,
            help='Filepath to the test corpus.')
    parser.add_argument('--model', metavar='M', type=str,
            help='Optional pre-trained model to load into classifier.')
    parser.add_argument('--weigh_labels', action='store_true',
            help='Determines whether to weigh labels')


    args = parser.parse_args()
    #main(args.corpus_filepath)
    main(args.train_corpus_filepath, args.test_corpus_filepath, args.model, 
        args.weigh_labels)
