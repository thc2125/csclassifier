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


def main(corpus_folder_filename, output_dirname='.', excluded_corpus_filename=None, 
    corpus_filename_prefix = 'normalized-twitter_cs_', use_alphabets=False, 
    epochs=50, batch_size=25):
    corpus_patt = re.compile(corpus_filename_prefix + '.')

    # Ingest the corpora
    c_f_fp = Path(corpus_folder_filename)
    corpora = {}
    for f in c_f_fp.iterdir():
        if corpus_patt.match(f.name):
            corpus = Corpus_CS_Langs(use_alphabets=use_alphabets)
            corpus.read_corpus(str(f), dl=',')
            langs = f.name.replace(corpus_filename_prefix,'')
            corpora[langs] = corpus

    # Create the test and training corpora
    # If we're testing on a single language pair
    if excluded_corpus_filename:
        test_langs = Path(excluded_corpus_filename).name.replace(corpus_filename_prefix,'')
        test_corpus = corpora[test_langs]
        train_corpus = Corpus_CS_Langs(train=True)
        for langs in corpora.keys():
            if langs != test_langs:
                train_corpus = train_corpus + corpora[langs]

    # Otherwise if we're testing on a randomized split of the data
    else:
        test_langs = "ALL"
        temp_corpus = Corpus_CS_Langs()
        for langs in corpora.keys():
            temp_corpus = temp_corpus + corpora[langs]
        train_corpus, test_corpus = temp_corpus.randomly_split_corpus()

    maxsentlen = max(train_corpus.maxsentlen, test_corpus.maxsentlen)
    maxwordlen = max(train_corpus.maxwordlen, test_corpus.maxwordlen)

    label2idx = train_corpus.label2idx
    idx2label = train_corpus.idx2label

    char2idx, idx2char = train_corpus.create_dictionary()
    csc = CSClassifier(maxsentlen, maxwordlen, label2idx, idx2label, char2idx, 
        idx2char, epochs, batch_size)
    print()
    print("Beginning Training. Excluding " + test_langs)
    print()
       
    csc.generate_model(train_corpus, test_langs, output_dirname=output_dirname)
    metrics = csc.evaluate_model(test_corpus)
    del test_corpus
    del train_corpus

    exp_results_path = Path(output_dirname, 'csclassifier_test_results.txt')
    if not exp_results_path.exists():
        mode = 'w'
    else:
        mode = 'a'

    with exp_results_path.open(mode=mode) as results_file:
        if mode == 'w':
            results_file.write("CSCLASSIFIER MODEL RESULTS:\n")
        results_file.write("Training on " + str([train_langs for train_langs 
                    in corpora.keys() if train_langs != test_langs]) + '\n')
        results_file.write("Testing on " + test_langs + '\n')
        for metric in metrics.keys():
            results_file.write(metric + ": " + str(metrics[metric]) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A neural network based'
        + 'classifier for detecting code switching.') 
    parser.add_argument('corpus_filepath', metavar='C', type=str,
            help='Filepath to the corpus.')
    parser.add_argument('-o', '--output_dir', metavar='O', type=str,
            help='Directory to store checkpoint and model files')
    parser.add_argument('-x', '--excluded_corpus', metavar='X', type=str,
            help='Filename of corpus to be excluded from training and instead'
                + 'tested on')
    parser.add_argument('-a', '--use_alphabets', action='store_true',
            help="Whether to use alphabetically based unknowns.")
    parser.add_argument('-p', '--prefix', metavar='P', type=str,
            help='Corpus filename prefix')
    parser.add_argument('-e', '--epochs', metavar='E', type=int,
            help='Number of epochs to train')


    args = parser.parse_args()
    main_args = {}
    main_args['corpus_folder_filename'] = args.corpus_filepath
    if args.output_dir:
        main_args['output_dirname'] = args.output_dir
    if args.excluded_corpus:
        main_args['excluded_corpus_filename'] = args.excluded_corpus
    if args.use_alphabets:
       main_args['use_alphabets'] = args.use_alphabets
    if args.prefix:
        main_args['corpus_filename_prefix'] = args.prefix
    if args.epochs:
        main_args['epochs'] = args.epochs

       
    main(**main_args)
