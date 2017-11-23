#!/usr/bin/python3

# Design of utils file and Corpus class inspired by 
# Columbia University
# COMS 4705W 2017 
# Assignment 4
# Professor Karl Stratos

# Model design as described by Jaech et. al. in
# "A Neural Model for Language Identification in Code-Switched Tweets"

import classifier
from classifier import Classifier

class CSClassifier(Classifier):

    labels = ['no_cs', 'cs']

    def __init__(self,
                 chars,
                 maxsentlen,
                 maxwordlen,
                 use_alphabets=False,
                 hyper_parameters=classifier.DEFAULT_HYPER_PARAMETERS,
                 model=None):

        Classifier.__init__(self, 
                            chars, 
                            self.labels,
                            maxsentlen, 
                            maxwordlen, 
                            use_alphabets,
                            hyper_parameters,
                            model)

    def _get_corpus_labels(self, corpus):
        return corpus.scslabels
