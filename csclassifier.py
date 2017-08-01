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

class CSClassifier():
    def __init__(self, maxsentlen, maxwordlen, label2idx, idx2label, char2idx, idx2char):
        self.maxsentlen = maxsentlen
        self.maxwordlen = maxwordlen
        self.label2idx = label2idx
        self.idx2label = idx2label
        self.char2idx = char2idx
        self.idx2char = idx2char

    def generate_model(self, train_corpus, test_langs):
        #train_split = [ceil(9 * len(corpus.sentences)/10)]
        #train_split = [ceil(len(corpus.sentences)/100), 2 * ceil(len(corpus.sentences)/100)]
        if train_corpus.maxsentlen > self.maxsentlen or train_corpus.maxwordlen > self.maxwordlen:
            raise Exception("'train_corpus' has greater maxsentlen or maxwordlen")

        train_sentences, train_labels, train_labels_weights = train_corpus.np_idx_conversion(maxsentlen,
            maxwordlen)

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
            checkpoint = ModelCheckpoint(filepath='checkpoints/checkpoint_'+test_langs+'.{epoch:02d}--{val_loss:.2f}.hdf5', monitor='val_loss', mode='min')
            classifier.model.fit(x=train_sentences, y=train_labels,
                epochs=epochs, batch_size=batch_size, validation_split=.1,
                sample_weight=train_labels_weights, callbacks=[checkpoint])
            # Save the model
            classifier.model.save('cs_classifier_model_' + test_langs + '.h5')
        return classifier

    def evaluate_model(self, test_corpus):
        # Evaluate the model
        #evaluation = classifier.model.evaluate(x=test_sentences, y=test_cs, batch_size=batch_size)
        #print(evaluation)
        maxsentlen = train_corpus.maxsentlen
        maxwordlen = train_corpus.maxwordlen
        test_sentences, test_labels, test_labels_weights = test_corpus.np_idx_conversion(maxsentlen,
            maxwordlen)

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
        return metrics
