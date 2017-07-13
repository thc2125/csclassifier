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
from math import log, ceil

import numpy as np

import keras.backend as K

from keras.layers import Input, Embedding, Conv1D, Dropout, GlobalMaxPooling1D
from keras.layers import Dense, Merge, TimeDistributed, Bidirectional, LSTM
from keras.layers import add, concatenate
from keras.models import Model
from keras.utils import plot_model

from sklearn import metrics
from utils import Corpus

class CSClassifier:
    def __init__(self, char2idx, maxsentlen, maxwordlen):

        self.C = char2idx.values()

        self.Cdim = ceil(log(len(self.C), 2))

        # Hyperparameters tuned by Jaech et. al. 
        self.n_1 = 59
        self.n_2 = 108
        self.lstm_dim = 23
        self.dropout_rate = .25
        self.kernel_size = 3

        # First let's set up Char2Vec
        # TODO: Here's an idea on masking: make another input vector for each 
        # sentence that applies masking yourself. I.e. after the embedding layer,
        # Multiply those values by a vector of [1,1,1,0,0,0] with 1 
        # corresponding to values you want and 0 to values you don't want.
        # This could be created during corpus ingestion and just be another value.
        # Maybe pop this input/merge right after the embedding layer and before 
        # the CNN.
        self.inputs = Input(shape=(maxsentlen, maxwordlen,))

        # Set up the embeddings
        # TODO: Add the masking layer for the padded value
        self.embeddings = TimeDistributed(Embedding(len(self.C),
            self.Cdim))(self.inputs)

        # Make T_1 (1st CNN)
        self.T_1 = TimeDistributed(Conv1D(
            # input shape can be deduced
            # 'filters' - the number of filters i.e. dimensionality of output
            filters=self.n_1,
            # 'kernel_size' - size of the window
            kernel_size=self.kernel_size,
            # 'padding' ensures the output will be the same length as input
            padding='same',
            activation = 'relu'))(self.embeddings)



        # Adding the dropout
        self.T_1_dropout = Dropout(rate=self.dropout_rate)(self.T_1)

        # Adding T_2 (2nd CNN)
        self.T_2_a = TimeDistributed(Conv1D(
            filters=self.n_2,
            kernel_size = (3),
            padding='valid',
            activation='relu'))(self.T_1_dropout)

        self.T_2_b = TimeDistributed(Conv1D(
            filters=self.n_2,
            kernel_size = (4),
            padding='valid',
            activation='relu'))(self.T_1_dropout)

        self.T_2_c = TimeDistributed(Conv1D(
            filters=self.n_2,
            kernel_size = (5),
            # TODO: T_2 uses padding so you end up with a vector for each
            # character right?
            padding='valid',
            activation='relu'))(self.T_1_dropout)



        # Adding y (max-pooling across time)
        self.y_a = TimeDistributed(GlobalMaxPooling1D())(self.T_2_a)
        self.y_b = TimeDistributed(GlobalMaxPooling1D())(self.T_2_b)
        self.y_c = TimeDistributed(GlobalMaxPooling1D())(self.T_2_c)
        self.y = concatenate([self.y_a, self.y_b, self.y_c])

        # Add f_r(y) 
        self.f_r_y = Dense(3 * self.n_2, activation='relu')(self.y)
        self.z = add([self.y, self.f_r_y])

        # The next phase is dealing with all the words in the sentence.     
        # TODO: Are default lstm activation functions okay?
        self.v = Bidirectional(LSTM(self.lstm_dim,
            return_sequences=True, dropout=self.dropout_rate))(self.z)
        self.p = Dense(3, activation='softmax')(self.v)
        self.model = Model(inputs=self.inputs, outputs=self.p)
        # Note that 'adam' has a default learning rate of .001
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', 
            metrics=['categorical_accuracy', self.f1_score])
        plot_model(self.model, show_shapes=True)

    def classify_sentence(self, sentence):
        """TKTK

        Keyword arguments:
        TKTK -- TKTK
        """
        pass

    def test(self):
        """TKTK

        Keyword arguments:
        TKTK -- TKTK
        """
        pass

    def f1_score(self, y_true, y_pred):
        """A Keras custom metric to evaluate f1 score

        Keyword arguments:
        y_true -- A tensor of gold labels
        y_pred -- A tensor of predicted labels
        """
        # Create a tensor of values equal to the predicted category
        y_true_cat = K.argmax(y_true, axis=-1)
        y_pred_cat = K.argmax(y_pred, axis=-1)
        cat_f1 = K.placeholder()
        for i in range(3):
            cat = K.constant(i, shape=y_true_cat.get_shape())
            # Binary tensors representing all the labels equal to a 
            # given category, i.e. get rid of any labels not for the 
            # current cat
            y_true_curr_cat = K.equal(y_true_cat, cat, K.floatx())
            y_pred_curr_cat = K.equal(y_pred_cat, cat, K.floatx())
            # Where do they match?
            true_positives = K.equal(y_true_curr_cat,
                                     y_pred_curr_cat, K.floatx())
            # Where do they not match?
            false_positives = K.not_equal(y_true_curr_cat,
                                          y_pred_curr_cat, K.floatx())
            # Sum the true positives and false positives
            sum_tp = K.sum(true_positives)
            sum_fp = K.sum(false_positives)
            precision = sum_tp / (sum_tp + sum_fp)
            recall = sum_tp / K.sum(y_true_curr_cat)
            f1 = 2 * (precision * recall) / (precision + recall)
            cat_f1 = K.concatenate([cat_f1, f1], axis=-1)
        return cat_f1 
        pass
def main(corpus_filepath, epochs=20, batch_size=25, corpus_bin='corpus.bin'):
    # Ingest the corpus
    corpus = Corpus(corpus_filepath)

    #corpus.print_sentences_langs()

    # train_split = [ceil(9 * len(corpus.cs)/10)]
    train_split = [ceil(len(corpus.cs)/100), 2 * ceil(len(corpus.cs)/100)]

    #train_sentences, test_sentences = np.split(corpus.sentences, train_split)
    train_sentences, test_sentences, _ = np.split(corpus.sentences, train_split)
    train_cs, test_cs, _ = np.split(corpus.cs, train_split)
    print(train_sentences.size)
    print(train_sentences.shape)
    print(train_cs.size)
    print(train_cs.shape)

    # Build the model
    classifier = CSClassifier(corpus.char2idx, corpus.maxsentlen, corpus.maxwordlen)
    
    # Train the model
    thistory = classifier.model.fit(x=train_sentences, y=train_cs,
            epochs=epochs, batch_size=batch_size, validation_split=.1)
    print(thistory)

    # Evaluate the model
    evaluation = classifier.model.evaluate(x=test_sentences, y=test_cs, batch_size=batch_size)
    print(evaluation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A neural network based'
        + 'classifier for detecting code switching.') 
    parser.add_argument('corpus_filepath', metavar='C', type=str,
            help='Filepath to the corpus.')
    '''
    parser.add_argument('--corpus_obj', metavar='CO', type=str, 
        help='A previously created corpus object.')
    '''
    args = parser.parse_args()
    main(args.corpus_filepath)
