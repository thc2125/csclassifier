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
from keras.callbacks import ModelCheckpoint

import utils

from utils import Corpus, Corpus_Aaron
from classifier import Classifier

def main(corpus_filepath, model, epochs=50, batch_size=25):
    # Ingest the corpus
    corpus = Corpus()
    corpus.read_corpus(corpus_filepath, dl=',')
    char2idx, idx2char = corpus.create_dictionary()

    train_split = [ceil(9 * len(corpus.sentences)/10)]
    #train_split = [ceil(len(corpus.sentences)/100), 2 * ceil(len(corpus.sentences)/100)]

    maxsentlen = corpus.maxsentlen
    maxwordlen = corpus.maxwordlen

    sentences, labels, labels_weights = corpus.np_idx_conversion(maxsentlen,
        maxwordlen)

    train_sentences, test_sentences = np.split(sentences, train_split)
    train_labels, test_labels = np.split(labels, train_split)
    train_labels_weights, _ = np.split(labels_weights, train_split)

    label2idx = corpus.label2idx
    idx2label = corpus.idx2label

    utils.print_np_sentence(train_sentences[20], idx2char) 
    utils.print_np_label(train_labels[20], idx2label)
    print(train_labels_weights[20])


    num_labels = len(corpus.label2idx)

    # Build the model
    classifier = Classifier(char2idx, maxsentlen, maxwordlen, num_labels)

    if model != None:
        # Load the model
        classifier.model = load_model(model)
    else:
        # Train the model
        checkpoint = ModelCheckpoint(filepath='checkpoint.{epoch:02d}--{val_loss:.2f}.hdf5', monitor='val_loss', mode='min')
        classifier.model.fit(x=train_sentences, y=train_labels,
            epochs=epochs, batch_size=batch_size, validation_split=.1,
            sample_weight=train_labels_weights, callbacks=[checkpoint])
        # Save the model
        classifier.model.save('cs_classifier_model.h5')

    # Evaluate the model
    #evaluation = classifier.model.evaluate(x=test_sentences, y=test_cs, batch_size=batch_size)
    #print(evaluation)
    print("Testing on sentences of shape: " + str(test_sentences.shape))
    pred_labels = classifier.model.predict(x=test_sentences)

    # Transform labels to represent category index
    test_cat_labels = np.argmax(test_labels, axis=2)
    pred_cat_labels = np.argmax(pred_labels, axis=2)

    metrics = (utils.compute_accuracy_metrics(
        test_cat_labels, pred_cat_labels, label2idx))
    for metric in metrics:
        if metric == 'confusion_matrix':
            continue
        print(metric + ": " + str(metrics[metric]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A neural network based'
        + 'classifier for detecting code switching.') 
    parser.add_argument('corpus_filepath', metavar='C', type=str,
            help='Filepath to the corpus.')
    parser.add_argument('--model', metavar='M', type=str,
            help='Optional pre-trained model to load into classifier.')

    args = parser.parse_args()
    #main(args.corpus_filepath)
    main(args.corpus_filepath, args.model)
