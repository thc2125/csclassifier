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
from pathlib import Path, PurePath

import numpy as np
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

import utils

from utils import Corpus, Corpus_Aaron
from classifier import Classifier

class CSClassifier():
    def __init__(self, maxsentlen, maxwordlen, label2idx, idx2label, char2idx, 
        idx2char, epochs, batch_size, patience, train_langs, test_langs):
        self.maxsentlen = maxsentlen
        self.maxwordlen = maxwordlen
        self.label2idx = label2idx
        self.idx2label = idx2label
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
 
        self.train_langs = train_langs
        self.test_langs = test_langs

    def generate_model(self, train_corpus, model=None, output_dirpath=PurePath('.')):
        #train_split = [ceil(9 * len(corpus.sentences)/10)]
        #train_split = [ceil(len(corpus.sentences)/100), 2 * ceil(len(corpus.sentences)/100)]
        if train_corpus.maxsentlen > self.maxsentlen or train_corpus.maxwordlen > self.maxwordlen:
            raise Exception("'train_corpus' has greater maxsentlen or maxwordlen")

        self.train_corpus = train_corpus
        self.test_langs_names = 'ALL' if len(self.test_langs) > 1 else self.test_langs[0]

        train_sentences, train_labels, train_labels_weights = self.train_corpus.np_idx_conversion(self.maxsentlen,
            self.maxwordlen)


        num_labels = len(self.label2idx)

        # Build the model
        self.classifier = Classifier(self.char2idx, self.maxsentlen, self.maxwordlen, num_labels)

        if model != None:
            # Load the model
            self.classifier.model = load_model(model)
        else:
            # Train the model
            alph = 'alph' if self.train_corpus.use_alphabets else ''
            # Create a folder to store checkpoints if one does not exist
            checkpoints_dirpath = Path(output_dirpath, 'checkpoints')
            if not checkpoints_dirpath.exists():
                checkpoints_dirpath.mkdir()
            checkpoint = ModelCheckpoint(
                filepath=str(checkpoints_dirpath/('checkpoint_'+self.test_langs_names+'_'
                    +alph+'.{epoch:02d}--'+'{val_loss:.2f}.hdf5')),
                    monitor='val_loss', mode='min')
            stop_early = EarlyStopping(
                monitor='val_categorical_accuracy',
                patience=self.patience)
            self.history = self.classifier.model.fit(x=train_sentences, y=train_labels,
                epochs=self.epochs, batch_size=self.batch_size, validation_split=.1,
                sample_weight=train_labels_weights, 
                callbacks=[checkpoint, stop_early])
            # Save the model
            self.classifier.model.save(str(output_dirpath / 
                ('cs_classifier_model_' + self.test_langs_names + '_' + alph + '.h5')))
            self.trained_epochs = len(self.history.epoch)
        return self.classifier

    def evaluate_model(self, test_corpus):
        # Evaluate the model
        #evaluation = classifier.model.evaluate(x=test_sentences, y=test_cs, batch_size=batch_size)
        #print(evaluation)
        #TODO: remove the following two commented out lines
        #maxsentlen = train_corpus.maxsentlen
        #maxwordlen = train_corpus.maxwordlen
        self.test_corpus = test_corpus
        test_sentences, test_labels, test_labels_weights = self.test_corpus.np_idx_conversion(self.maxsentlen,
            self.maxwordlen)

        print("Testing on sentences of shape: " + str(test_sentences.shape))
        pred_labels = self.classifier.model.predict(x=test_sentences)

        # Transform labels to represent category index
        test_cat_labels = np.argmax(test_labels, axis=2)
        pred_cat_labels = np.argmax(pred_labels, axis=2)

        metrics = {}

        metrics['word'] = (utils.compute_accuracy_metrics(
            test_cat_labels, pred_cat_labels, self.label2idx))

        # Reduce gold labels and predicted labels to sentence level
        # cs identification
        test_cat_labels_sent = np.maximum.reduce(test_cat_labels,axis=1, keepdims=True)
        pred_cat_labels_sent = np.maximum.reduce(pred_cat_labels,axis=1, keepdims=True)

        metrics['sentence'] = (utils.compute_accuracy_metrics(
            test_cat_labels_sent, pred_cat_labels_sent, self.label2idx))

        return metrics
