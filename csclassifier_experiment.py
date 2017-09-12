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
import time
import datetime

from math import ceil
from pathlib import Path, PurePath


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
    epochs=50, batch_size=25, patience=2):

    start_time = time.process_time()
    start_date = datetime.date.today()
    corpus_patt = re.compile(corpus_filename_prefix + '.')

    # Create the test and training corpora
    # If we're testing on a single language pair
    if excluded_corpus_filename:
        # First ingest all the corpora
        c_f_fp = Path(corpus_folder_filename)
        corpora = {}
        for f in c_f_fp.iterdir():
            if corpus_patt.match(f.name):
                corpus = Corpus_CS_Langs(use_alphabets=use_alphabets)
                corpus.read_corpus(str(f), dl=',')
                langs = f.name.replace(corpus_filename_prefix,'')
                corpora[langs] = corpus

        test_langs = [Path(excluded_corpus_filename).name.replace(corpus_filename_prefix,'')]
        train_langs = []
        test_corpus = corpora[test_langs[0]]
        train_corpus = Corpus_CS_Langs(train=True)
        for langs in corpora.keys():
            if langs != test_langs:
                train_langs += [langs]
                train_corpus = train_corpus + corpora[langs]

    # Otherwise if we're testing on a randomized split of the data
    else:

        all_langs = []
        train_corpus = Corpus_CS_Langs(use_alphabets=use_alphabets, train=True)
        test_corpus = Corpus_CS_Langs(use_alphabets=use_alphabets)
        for f in c_f_fp.iterdir():
            if corpus_patt.match(f.name):
                utils.randomly_read_CS_Langs_Corpus(str(f), train_corpus, 
                    test_corpus, dl=',')
                langs = f.name.replace(corpus_filename_prefix,'')
                all_langs += [langs]

        train_corpus.read_corpus_bookkeeping()
        test_corpus.read_corpus_bookkeeping()

        train_langs, test_langs = all_langs, all_langs

    maxsentlen = max(train_corpus.maxsentlen, test_corpus.maxsentlen)
    maxwordlen = max(train_corpus.maxwordlen, test_corpus.maxwordlen)

    label2idx = train_corpus.label2idx
    idx2label = train_corpus.idx2label

    char2idx, idx2char = train_corpus.create_dictionary()
    csc = CSClassifier(maxsentlen, maxwordlen, label2idx, idx2label, char2idx, 
        idx2char, epochs, batch_size, patience, train_langs, test_langs)

    print()
    print("Beginning Training. Excluding " 
        + ('NONE' if len(test_langs)>1 else test_langs[0]))
    print()
       
    csc.generate_model(train_corpus, output_dirpath=PurePath(output_dirname))
    metrics = csc.evaluate_model(test_corpus)
    print()

    
    end_date = datetime.date.today()
    end_time = time.process_time()
    
    output = ([batch_size, epochs, csc, start_date, end_date, start_time, end_time, use_alphabets,
              metrics])
    return produce_output(*output)
    #del test_corpus
    #del train_corpus

def produce_output(
        batch_size, epochs_expected, csc, start_date, end_date, start_time, 
            end_time, use_alphabets, metrics):

    # Let's start with experiment parameters
    experiment_output = "CSCLASSIFIER MODEL RESULTS:\n\n"
    experiment_output += "MODEL INFORMATION: \n"
    experiment_output += "{:<4}{:<22}{}\n".format("", "Batch-Size:", str(batch_size))
    experiment_output += "{:<4}{:<22}{}\n".format("","Epochs Run:", 
        str(csc.trained_epochs))
    experiment_output += "{:<4}{:<22}{}\n".format("","Epochs Expected:", 
        str(epochs_expected))
    experiment_output += "{:<4}{:<22}{}\n".format("","Patience:",
        str(csc.patience))
    experiment_output += "{:<4}{:<22}{}\n".format("","Start Date:",
        str(start_date))
    experiment_output += "{:<4}{:<22}{}\n".format("","End Date:",
        str(end_date))
    experiment_output += "{:<4}{:<22}{}\n".format("","Start Time:",
        str(start_time))
    experiment_output += "{:<4}{:<22}{}\n".format("","End Time:",
        str(end_time))
    experiment_output += "{:<4}{:<22}{}\n".format("","Total Processing Time:",
        str(end_time-start_time))

    experiment_output += "\n"

    # Now let's add hyperparameters
    experiment_output += "\tHyper-parameters:\n"
    experiment_output += "\t\t{:<26}{}\n".format("CNN T1 Filter Dimensions:",
        str(csc.classifier.n_1))  
    experiment_output += "\t\t{:<26}{}\n".format("CNN T1 Kernel Size:",
        str(csc.classifier.kernel_size))  
    experiment_output += "\t\t{:<26}{}\n".format("CNN T2 Filter Dimensions:",
        str(csc.classifier.n_2))  
    experiment_output += "\t\t{:<26}{}\n".format("LSTM Dimensions:",
        str(csc.classifier.lstm_dim))  
    experiment_output += "\t\t{:<26}{}\n".format("Dropout Rate:",
        str(csc.classifier.dropout_rate))  
    experiment_output += "\t\t{:<26}{}\n".format("Loss Algorithm:",
        str(csc.classifier.loss))  
    experiment_output += "\t\t{:<26}{}\n".format("Loss Optimizer:",
        str(csc.classifier.optimizer_name))  
    experiment_output += "\t\t{:<26}{}\n".format("Learning Rate:",
        str(csc.classifier.learning_rate))  
    experiment_output += "\t\t{:<26}{}\n".format("Decay Rate:",
        str(csc.classifier.decay))

    experiment_output += "\n"

    experiment_output += "{:<13}{}\n".format("Training on: ",str(csc.train_langs))
    experiment_output += "{:<13}{}\n".format("Testing on: ",str(csc.test_langs))

    experiment_output += "\n"

    experiment_output += ("Unknown character vectors associated w/ alphabets: "
        + str(use_alphabets) + "\n")

    experiment_output += "\n"

    experiment_output += "Results:\n"

    experiment_output += "\n"

    experiment_output += "{:<17}{:<14}{}\n".format("","Word-level","Sentence-level")
    experiment_output += "{:<17}{:<14.8}{:.8}\n".format("Accuracy:",
            metrics['word']['accuracy'],metrics['sentence']['accuracy'])

    experiment_output += "{:<17}\n".format("Precision:")
    experiment_output += "{:<4}{:<13}{:<14.8}{:.8}\n".format("","PADDING",
            metrics['word']['precision'][0], 
            metrics['sentence']['precision'][0])
    experiment_output += "{:<4}{:<13}{:<14.8}{:.8}\n".format("","NON-CS",
            metrics['word']['precision'][1], 
            metrics['sentence']['precision'][1])
    experiment_output += "{:<4}{:<13}{:<14.8}{:.8}\n".format("","CS",
            metrics['word']['precision'][2], 
            metrics['sentence']['precision'][2])

    experiment_output += "{:<17}\n".format("Recall:")
    experiment_output += "{:<4}{:<13}{:<14.8}{:.8}\n".format("","PADDING",
            metrics['word']['recall'][0], 
            metrics['sentence']['recall'][0])
    experiment_output += "{:<4}{:<13}{:<14.8}{:.8}\n".format("","NON-CS",
            metrics['word']['recall'][1], 
            metrics['sentence']['recall'][1])
    experiment_output += "{:<4}{:<13}{:<14.8}{:.8}\n".format("","CS",
            metrics['word']['recall'][2], 
            metrics['sentence']['recall'][2])

    experiment_output += "{:<17}\n".format("F-Score:")
    experiment_output += "{:<4}{:<13}{:<14.8}{:.8}\n".format("","PADDING",
            metrics['word']['fscore'][0], 
            metrics['sentence']['fscore'][0])
    experiment_output += "{:<4}{:<13}{:<14.8}{:.8}\n".format("","NON-CS",
            metrics['word']['fscore'][1], 
            metrics['sentence']['fscore'][1])
    experiment_output += "{:<4}{:<13}{:<14.8}{:.8}\n".format("","CS",
            metrics['word']['fscore'][2], 
            metrics['sentence']['fscore'][2])

    experiment_output += "\n"
    experiment_output += "CORPUS COMPOSITION:\n"
    experiment_output += "\n"
    experiment_output += "{:<24}{:<18}{}\n".format("", "Train/Dev", "Test")
    experiment_output += "{:<24}{:<18,}{:<18,}\n".format(
            "Total Sentences:", 
            (len(csc.train_corpus.sentences)),
            (len(csc.test_corpus.sentences)))
    experiment_output += "{:<24}{:<10,}{:<8.3%}{:<10,}{:.3%}\n".format(
            "Monolingual Sentences:",
            (len(csc.train_corpus.sentences)-
                csc.train_corpus.multilingual_sentence_count),
            ((len(csc.train_corpus.sentences)-
                csc.train_corpus.multilingual_sentence_count) 
                / len(csc.train_corpus.sentences)),
            (len(csc.test_corpus.sentences)-
                csc.test_corpus.multilingual_sentence_count),
            ((len(csc.test_corpus.sentences)-
                csc.test_corpus.multilingual_sentence_count) 
                / len(csc.test_corpus.sentences)))

    experiment_output += "{:<24}{:<10,}{:<8.3%}{:<10,}{:.3%}\n".format(
            "Multilingual Sentences:",
            (csc.train_corpus.multilingual_sentence_count),
            (csc.train_corpus.multilingual_sentence_count 
                / len(csc.train_corpus.sentences)),
            (csc.test_corpus.multilingual_sentence_count),
            (csc.test_corpus.multilingual_sentence_count 
                / len(csc.test_corpus.sentences)))

    experiment_output += "\n"

    experiment_output += "{:<46}{:<14,}{:<14,}\n".format(
        "Total # of switches:", 
        (csc.train_corpus.switch_count),
        (csc.test_corpus.switch_count))


    experiment_output += "{:<46}{:<14,.3}{:<14,.3}\n".format(
        "Avg. # of switches per sentence:", 
        (csc.train_corpus.switch_count / len(csc.train_corpus.sentences)),
        (csc.test_corpus.switch_count / len(csc.test_corpus.sentences)))

    '''
    experiment_output += "{:<46}{:<14,.3}{:<14,.3}\n".format(
        "Avg. # of switches per multilingual sentence:", 
        (csc.train_corpus.switch_count 
            / csc.train_corpus.multilingual_sentence_count),
        (csc.test_corpus.switch_count 
            / csc.train_corpus.multilingual_sentence_count))
    '''


    print(experiment_output)
    return experiment_output

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
    parser.add_argument('-t', '--patience', metavar='T', type=int,
            help='Number of epochs to wait for improvement')



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
    if args.patience:
        main_args['patience'] = args.patience
       
    main(**main_args)
