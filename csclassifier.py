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

from keras.layers import Input, Embedding, Conv1D, Dropout, GlobalMaxPooling1D
from keras.layers import Dense, Merge, TimeDistributed, Bidirectional, LSTM
from keras.layers import add, concatenate
from keras.models import Model
from keras.utils import plot_model

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
        # 
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
        self.p = Dense(4, activation='softmax')(self.v)
        self.model = Model(inputs=self.inputs, outputs=self.p)
        # Note that 'adam' has a default learning rate of .001
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')
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
def main(corpus_filepath, epochs=20, batch_size=25, corpus_bin='corpus.bin'):
    # Ingest the corpus
    corpus = Corpus(corpus_filepath)

    #corpus.print_sentences_langs()

    #train_split = [ceil(9 * len(corpus.langs)/10)]
    train_split = [ceil(len(corpus.langs)/10)]
    #train_sentences, test_sentences = np.split(corpus.sentences, train_split)
    train_sentences, test_sentences = np.split(corpus.sentences, train_split)
    train_langs, test_langs = np.split(corpus.langs, train_split)
    print(train_sentences.size)
    print(train_sentences.shape)
    print(train_langs.size)
    print(train_langs.shape)

    # Build the model
    classifier = CSClassifier(corpus.char2idx, corpus.maxsentlen, corpus.maxwordlen)
    
    # Train the model
    thistory = classifier.model.fit(x=train_sentences, y=train_langs,
            epochs=epochs, batch_size=batch_size, validation_split=.1)
    print(thistory)

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
