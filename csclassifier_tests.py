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
import re

from math import ceil
from pathlib import Path


import numpy as np
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

import utils
import csclassifier

from utils import Corpus, Corpus_CS_Langs
from classifier import Classifier
from csclassifier import CSClassifier

corpus_filename_prefix = 'normalized-twitter_cs_'

def main(corpus_folder_filepath, model, epochs=50, batch_size=25):
    corpus_patt = re.compile(corpus_filename_prefix + '.')

    # Ingest the corpora
    c_f_fp = Path(corpus_folder_filepath)
    corpora = {}
    for f in c_f_fp.iterdir():
        if corpus_patt.match(f.name):
            corpus = Corpus_CS_Langs()
            corpus.read_corpus(f.name, dl=',')
            langs = f.name.replace(corpus_filename_prefix,'')
            corpora[langs] = corpus

    metrics = {}
    for test_langs in corpora.keys():
        test_corpus = corpora[test_langs]
        train_corpus = Corpus_CS_Langs(train=True)
        for langs in corpora.keys():
            if langs != test_langs:
                train_corpus = corpus + corpora[langs]

        maxsentlen = max(train_corpus.maxsentlen, test_corpus.maxsentlen)
        maxwordlen = max(train_corpus.maxwordlen, test_corpus.maxwordlen)

        label2idx = train_corpus.label2idx
        idx2label = train_corpus.idx2label

        char2idx, idx2char = train_corpus.create_dictionary()

        csc = CSClassifier(maxsentlen, maxwordlen, label2idx, idx2label, char2idx, idx2char)
       
        csc.generate_model(train_corpus, test_langs)
        metrics[test_langs] = csc.evaluate_model(test_corpus)
        del test_corpus
        del train_corpus

    corpus = Corpus_CS_Langs(train=True)

    with open('csclassifier_test_results', 'w') as results_file:
        results_file.write("CSCLASSIFIER MODEL RESULTS:\n")
        for lang in metrics.keys():
            results_file.write("Training on " + str([train_lang 
                    if train_lang != lang else None for train_lang in metrics.keys()]) + '\n')
            results_file.write("Testing on " + lang + '\n')
            for metric in metrics[lang].keys():
                results_file.write(metric + ": " + metrics[lang][metric] + '\n')


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
