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
from math import log, ceil

from keras.layers import Input, Embedding, Conv1D, Dropout, MaxPooling1D
from keras.layers import Dense, Merge, TimeDistributed, Bidirectional, LSTM
from keras.layers import add, concatenate
from keras.models import Model

from utils import Corpus

class CSClassifier:
    def __init__(self, char2idx):

        self.C = char2idx.values()

        self.Cdim = ceil(log(len(self.C), 2))

        # Hyperparameters tuned by Jaech et. al. 
        self.n_1 = 59
        self.n_2 = 108
        self.lstm_dim = 23
        self.dropout_rate = .25
        self.kernel_size = 3


        # First let's set up Char2Vec
        self.input = Input(shape=(None,))

        # Set up the embeddings
        self.embeddings = Embedding(len(self.C), self.Cdim)(self.input) 

        # Make T_1 (1st CNN)
        self.T_1 = (Conv1D(
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
        self.T_2_a = (Conv1D(
            filters=self.n_2,
            kernel_size = (3),
            padding='same',
            activation='relu'))(self.T_1_dropout)

        self.T_2_b = (Conv1D(
            filters=self.n_2,
            kernel_size = (4),
            padding='same',
            activation='relu'))(self.T_1_dropout)

        self.T_2_c = (Conv1D(
            filters=self.n_2,
            kernel_size = (5),
            # TODO: T_2 uses padding so you end up with a vector for each
            # character right?
            padding='same',
            activation='relu'))(self.T_1_dropout)



        # Adding y (max-pooling across time)
        self.y_a = MaxPooling1D(pool_size=1, padding='same')(self.T_2_a)
        self.y_b = MaxPooling1D(pool_size=1, padding='same')(self.T_2_b)
        self.y_c = MaxPooling1D(pool_size=1, padding='same')(self.T_2_c)
        self.y = concatenate([self.y_a, self.y_b, self.y_c])

        # Add f_r(y) 
        self.f_r_y = Dense(3 * self.n_2, activation='relu')(self.y)
        self.z = add([self.y, self.f_r_y])

        # The next phase is dealing with all the words in the sentence.     
        self.char2vec = Model(inputs=self.input, outputs=self.z)
        self.inputs = Input(shape=(None, None))
        sent2vec = TimeDistributed(self.char2vec)(self.inputs)
        # TODO: Are default lstm activation functions okay?
        self.v = (Bidirectional(LSTM(self.lstm_dim,
            dropout=self.dropout_rate))(self.Char2Vec))
        self.p = Dense(3, activation='softmax')

        self.model = Model(inputs=self.inputs, outputs=self.p)
        # Note that 'adam' has a default learning rate of .001
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')


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
def main(corpus_filepath, epochs=20, batch_size=25):
    corpus = Corpus(corpus_filepath)
    #corpus.print_sentences_langs()
    sentences = corpus.sentences
    langs = corpus.langs
    classifier = CSClassifier(corpus.char2idx)
    thistory = classifier.model.fit(x=sentences, y=langs, epochs=epochs, batch_size=batch_size,
            validation_split=.1)
    print(thistory)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A neural network based'
        + 'classifier for detecting code switching.') 
    parser.add_argument('corpus_filepath', metavar='C', type=str,
            help='Filepath to the corpus.')
    args = parser.parse_args()
    main(args.corpus_filepath)
