#!/usr/bin/python3

# Design of utils file and Corpus class inspired by 
# Columbia University
# COMS 4705W 2017 
# Assignment 4
# Professor Karl Stratos

# Model design as described by Jaech et. al. in
# "A Neural Model for Language Identification in Code-Switched Tweets"

import argparse
import datetime
import itertools
import json
import time

from collections import OrderedDict
from pathlib import Path, PurePath

import classifier
import utils

from corpus_cs_langs import CorpusCSLangs
from csclassifier import CSClassifier


CORPUS_FILENAMES = {'train' : 'train_corpus.tsv', 'test' : 'test_corpus.tsv'}

def run(corpora_dirpath, 
        output_dirpath=Path('exp' 
                            + "_".join(
                                str(datetime.datetime.utcnow()).replace(
                                    ':','_').replace(
                                    '.','_').split())),
        excluded_langs=None, 
        use_alphabets=False, 
        epochs=50, 
        batch_size=25, 
        patience=2,
        model_parameters=classifier.DEFAULT_HYPER_PARAMETERS):

    if not output_dirpath.exists():
        output_dirpath.mkdir(parents=True)

    experiment_parameters = {'use_alphabets':use_alphabets,
                             'epochs':epochs,
                             'batch_size':batch_size,
                             'patience':patience}
                            
    # Begin the timer.
    start_time = time.process_time()
    start_date = datetime.datetime.now()

    # Create the test and training corpora
    train_corpus, test_corpus = create_corpora(corpora_dirpath)
        
    maxsentlen = max(train_corpus.maxsentlen, test_corpus.maxsentlen)
    maxwordlen = max(train_corpus.maxwordlen, test_corpus.maxwordlen)

    chars = train_corpus.get_chars()
    csc = CSClassifier(chars, maxsentlen, maxwordlen, use_alphabets, model_parameters)

    print()

    print("Beginning Training.") 

    print()
       
    csc.train_model(train_corpus, 
                    output_dirpath=PurePath(output_dirpath), 
                    epochs=epochs,
                    batch_size=batch_size,
                    patience=patience)

    print("Saving Model.")
    model_path, model_history_path = csc.save_model(output_dirpath)
    print(model_path)
    print(model_history_path) 

    print()

    print("Evaluating Model.")
    metrics = csc.evaluate_model(test_corpus)

    print()

    end_date = datetime.datetime.now()
    process_time = time.process_time()

    experiment_parameters['epochs_run'] = csc.trained_epochs
    experiment_parameters['start_date'] = str(start_date)
    experiment_parameters['end_date'] = str(end_date)
    experiment_parameters['total_time'] = str(process_time)

    output_results(output_dirpath, 
                   experiment_parameters, 
                   model_parameters, 
                   metrics,
                   train_corpus, 
                   test_corpus)

def output_results(output_dirpath,
                   experiment_parameters, 
                   model_parameters, 
                   metrics,
                   train_corpus, 
                   test_corpus):
    experiment_output = consolidate_output(experiment_parameters, 
                                           model_parameters,
                                           metrics,
                                           train_corpus, 
                                           test_corpus)

    json_output_path = output_dirpath / 'results.json'

    print(experiment_output)
    with json_output_path.open('w') as json_output_file:
        json.dump(experiment_output, json_output_file, indent=4, sort_keys=True)

    txt_output_path = output_dirpath / 'results.txt'
    with txt_output_path.open('w') as txt_output_file:
        txt_output_file.write(print_output(experiment_output))
    return json_output_path, txt_output_path

def consolidate_output(experiment_parameters, 
                       model_parameters, 
                       metrics,
                       train_corpus, 
                       test_corpus):

    output = {'experiment':experiment_parameters,
              'model':model_parameters,
              'metrics':metrics,
              'corpora':extract_cscorpora_metrics(train_corpus, test_corpus)}
    return output

def extract_cscorpora_metrics(train_corpus, test_corpus):
    return {'train':extract_cscorpus_metrics(train_corpus), 
            'test':extract_cscorpus_metrics(test_corpus)}

def extract_cscorpus_metrics(corpus):
    corpus_metrics = {}
    corpus_metrics['filenames'] = [str(filepath) for filepath in corpus.filepaths]
    corpus_metrics['sentences'] = {}
    corpus_metrics['sentences']['monolingual_sentences'] = (len(corpus.sentences) 
                                               - len(corpus.multilingual_sentences))
    corpus_metrics['sentences']['multilingual_sentences'] = len(corpus.multilingual_sentences)
    corpus_metrics['cs_labels'] = corpus.cslabel_count
    corpus_metrics['cs_types'] = corpus.cstype_count
    corpus_metrics['labels'] = corpus.label_count
    corpus_metrics['langs'] = corpus.lang_count
    corpus_metrics['lang_pairs'] = corpus.lang_pair_count


    #corpus_metrics['chars'] = corpus.char_count
    
    return corpus_metrics


def print_output(experiment_output):

    # Let's start with experiment parameters
    str_experiment_output = "CSCLASSIFIER MODEL RESULTS:\n\n"
    str_experiment_output += "EXPERIMENT INFORMATION: \n"
    experiment_info = experiment_output['experiment']
    str_experiment_output += "{:<4}{:<22}{:d}\n".format("", "Batch-Size:", experiment_info['batch_size'])
    str_experiment_output += "{:<4}{:<22}{:d}\n".format("","Epochs Trained:", 
        experiment_info['epochs_run'])
    str_experiment_output += "{:<4}{:<22}{:d}\n".format("","Epochs Expected:", 
        experiment_info['epochs_run'])
    str_experiment_output += "{:<4}{:<22}{:d}\n".format("","Patience:",
        experiment_info['patience'])
    str_experiment_output += "{:<4}{:<22}{:s}\n".format("","Start Date:",
        experiment_info['start_date'])
    str_experiment_output += "{:<4}{:<22}{:s}\n".format("","End Date:",
        experiment_info['end_date'])
    str_experiment_output += "{:<4}{:<22}{:s}\n".format("","Total Processing Time:",
        experiment_info['total_time'])

    str_experiment_output += "\n"

    # Now let's add hyperparameters
    model_info = experiment_output['model']
    str_experiment_output += "MODEL INFORMATION:\n"
    str_experiment_output += "{:<8}{:<26}{:d}\n".format("","CNN T1 Filter Dimensions:",
        model_info['n_1'])  
    str_experiment_output += "{:<8}{:<26}{:d}\n".format("","CNN T1 Kernel Size:",
        model_info['kernel_size'])  
    str_experiment_output += "{:<8}{:<26}{:d}\n".format("","CNN T2 Filter Dimensions:",
        model_info['n_2'])  
    str_experiment_output += "{:<8}{:<26}{:d}\n".format("","LSTM Dimensions:",
        model_info['lstm_dim'])  
    str_experiment_output += "{:<8}{:<26}{:F}\n".format("","Dropout Rate:",
        model_info['dropout_rate'])  
    str_experiment_output += "{:<8}{:<26}{}\n".format("","Loss Algorithm:",
        model_info['loss'])  
    str_experiment_output += "{:<8}{:<26}{}\n".format("","Loss Optimizer:",
        model_info['optimizer'])  
    str_experiment_output += "{:<8}{:<26}{:F}\n".format("","Learning Rate:",
        model_info['learning_rate'])  
    str_experiment_output += "{:<8}{:<26}{:F}\n".format("","Decay Rate:",
        model_info['decay'])

    str_experiment_output += "\n"

    train_corpus_info = experiment_output['corpora']['train']
    test_corpus_info = experiment_output['corpora']['test']
 
    str_experiment_output += "Training on: \n"
    str_experiment_output += train_corpus_info['filenames'][0]
    for filename in train_corpus_info['filenames'][1:]:
        str_experiment_output += ", " + filename
    str_experiment_output += "\n"

    for lang in list(train_corpus_info['lang_pairs'].keys())[:-1]:
        str_experiment_output += "{:<4}{:s}, ".format("", lang)
    str_experiment_output += "{:s}, ".format(list(train_corpus_info['lang_pairs'].keys())[-1])

    str_experiment_output += "\n"


    str_experiment_output += "Testing on: \n"
    str_experiment_output += test_corpus_info['filenames'][0]
    for filename in test_corpus_info['filenames'][1:]:
        str_experiment_output += ", " + filename
    str_experiment_output += "\n"
    for lang in list(test_corpus_info['lang_pairs'].keys())[:-1]:
        str_experiment_output += "{:<4}{:s}, ".format("", lang)
    str_experiment_output += "{:s}, ".format(list(test_corpus_info['lang_pairs'].keys())[-1])

    str_experiment_output += "\n"

    str_experiment_output += ("Unknown character vectors associated w/ alphabets: "
        + str(experiment_info['use_alphabets']) + "\n")

    str_experiment_output += "\n"

    str_experiment_output += "EXPERIMENT RESULTS:\n"

    str_experiment_output += "\n"

    metric_info = experiment_output['metrics']
    str_experiment_output += "{:<17}{:<14}{}\n".format("","Word-level","Sentence-level")
    str_experiment_output += "{:<17}{:<14.8}{:.8}\n".format("Accuracy:",
            metric_info['word']['accuracy'],metric_info['sentence']['accuracy'])

    str_experiment_output += "{:<17}\n".format("Precision:")
    str_experiment_output += "{:<4}{:<13}{:<14.8}{:.8}\n".format("","PADDING",
            metric_info['word']['precision'][0], 
            metric_info['sentence']['precision'][0])
    str_experiment_output += "{:<4}{:<13}{:<14.8}{:.8}\n".format("","NON-CS",
            metric_info['word']['precision'][1], 
            metric_info['sentence']['precision'][1])
    str_experiment_output += "{:<4}{:<13}{:<14.8}{:.8}\n".format("","CS",
            metric_info['word']['precision'][2], 
            metric_info['sentence']['precision'][2])

    str_experiment_output += "{:<17}\n".format("Recall:")
    str_experiment_output += "{:<4}{:<13}{:<14.8}{:.8}\n".format("","PADDING",
            metric_info['word']['recall'][0], 
            metric_info['sentence']['recall'][0])
    str_experiment_output += "{:<4}{:<13}{:<14.8}{:.8}\n".format("","NON-CS",
            metric_info['word']['recall'][1], 
            metric_info['sentence']['recall'][1])
    str_experiment_output += "{:<4}{:<13}{:<14.8}{:.8}\n".format("","CS",
            metric_info['word']['recall'][2], 
            metric_info['sentence']['recall'][2])

    str_experiment_output += "{:<17}\n".format("F-Score:")
    str_experiment_output += "{:<4}{:<13}{:<14.8}{:.8}\n".format("","PADDING",
            metric_info['word']['fscore'][0], 
            metric_info['sentence']['fscore'][0])
    str_experiment_output += "{:<4}{:<13}{:<14.8}{:.8}\n".format("","NON-CS",
            metric_info['word']['fscore'][1], 
            metric_info['sentence']['fscore'][1])
    str_experiment_output += "{:<4}{:<13}{:<14.8}{:.8}\n".format("","CS",
            metric_info['word']['fscore'][2], 
            metric_info['sentence']['fscore'][2])

    str_experiment_output += "\n"
    str_experiment_output += "CORPUS COMPOSITION:\n"
    str_experiment_output += "\n"
    str_experiment_output += "{:<24}{:<18}{}\n".format("", "Train/Dev", "Test")

    train_sentence_count = sum([count 
                                for count in train_corpus_info['sentences'].values()])
    test_sentence_count = sum([count 
                               for count in test_corpus_info['sentences'].values()])
    print(train_sentence_count)
    print(test_sentence_count)
    str_experiment_output += "{:<24}{:<18,}{:<18,}\n".format(
            "Total Tokens:", 
            sum(train_corpus_info['labels'].values()),
            sum(test_corpus_info['labels'].values()))

    str_experiment_output += "{:<24}{:<18,}{:<18,}\n".format(
            "Total Sentences:", 
            train_sentence_count,
            test_sentence_count)
    str_experiment_output += "{:<24}{:<10,}{:<8.2%}{:<10,}{:8.2%}\n".format(
            "Monolingual Sentences:",
            train_corpus_info['sentences']['monolingual_sentences'],
            train_corpus_info['sentences']['monolingual_sentences'] / train_sentence_count,
            test_corpus_info['sentences']['monolingual_sentences'],
            test_corpus_info['sentences']['monolingual_sentences'] / test_sentence_count)

    str_experiment_output += "{:<24}{:<10,}{:<8.2%}{:<10,}{:8.2%}\n".format(
            "Multilingual Sentences:",
            train_corpus_info['sentences']['multilingual_sentences'],
            train_corpus_info['sentences']['multilingual_sentences'] / train_sentence_count,
            test_corpus_info['sentences']['multilingual_sentences'],
            test_corpus_info['sentences']['multilingual_sentences'] / test_sentence_count)

    str_experiment_output += "\n"

    str_experiment_output += "{:<46}{:<14,}{:<14,}\n".format(
        "Total # of switches:", 
        (train_corpus_info['cs_labels']['cs']),
        (test_corpus_info['cs_labels']['cs']))

    str_experiment_output += "{:<46}{:<14,.3}{:<14,.3}\n".format(
        "Avg. # of switches per sentence:", 
        (train_corpus_info['cs_labels']['cs'] 
         / train_sentence_count),
        (test_corpus_info['cs_labels']['cs'] 
         / test_sentence_count))

    str_experiment_output += "{:<46}{:<14,.3}{:<14,.3}\n".format(
        "Avg. # of switches per multilingual sentence:", 
        (train_corpus_info['cs_labels']['cs'] 
         / train_corpus_info['sentences']['multilingual_sentences']),
        (test_corpus_info['cs_labels']['cs'] 
         / test_corpus_info['sentences']['multilingual_sentences']))

    str_experiment_output += "\n"

    str_experiment_output += "Code Switch Types:\n"

    cs_types = sorted(list(set(list(train_corpus_info['cs_types'].keys())
                               + list(test_corpus_info['cs_types'].keys()))), 
                      key=lambda cs_type:(len(cs_type), cs_type))
    for cs_type in cs_types:
        str_experiment_output += "{:^4}{:<20}{:<10,}{:<8.2%}{:<10,}{:8.2%}\n".format(
                "",
                cs_type,
                train_corpus_info['cs_types'][cs_type] 
                if cs_type in train_corpus_info['cs_types']
                else 0,
                train_corpus_info['cs_types'][cs_type] / train_corpus_info['cs_labels']['cs']
                if cs_type in train_corpus_info['cs_types']
                else 0,
                test_corpus_info['cs_types'][cs_type] 
                if cs_type in test_corpus_info['cs_types']
                else 0,
                test_corpus_info['cs_types'][cs_type] / test_corpus_info['cs_labels']['cs']
                if cs_type in test_corpus_info['cs_types']
                else 0)
    
    str_experiment_output += "\n"

    str_experiment_output += "Token Types:\n"

    labels = sorted(list(set(list(train_corpus_info['labels'].keys()) 
                      + list(test_corpus_info['labels'].keys()))))
    for label in labels:
        str_experiment_output += "{:^4}{:<20}{:<10,}{:<8.2%}{:<10,}{:8.2%}\n".format(
                "",
                label,
                train_corpus_info['labels'][label] 
                if label in train_corpus_info['labels']
                else 0,
                train_corpus_info['labels'][label] / sum(train_corpus_info['labels'].values())
                if label in train_corpus_info['labels']
                else 0,
                test_corpus_info['labels'][label] 
                if label in test_corpus_info['labels']
                else 0,
                test_corpus_info['labels'][label] / sum(test_corpus_info['labels'].values())
                if label in test_corpus_info['labels']
                else 0)

    str_experiment_output += "\n"

    str_experiment_output += "Language Types:\n"

    langs = sorted(list(set(list(train_corpus_info['langs'].keys())
                + list(test_corpus_info['langs'].keys()))))

    for lang in langs:
        str_experiment_output += "{:^4}{:<20}{:<10,}{:<8.2%}{:<10,}{:8.2%}\n".format(
                "",
                lang,
                train_corpus_info['langs'][lang] 
                if lang in train_corpus_info['langs']
                else 0,
                train_corpus_info['langs'][lang] / sum(train_corpus_info['langs'].values())
                if lang in train_corpus_info['langs']
                else 0,
                test_corpus_info['langs'][lang] 
                if lang in test_corpus_info['langs']
                else 0,
                test_corpus_info['langs'][lang] / sum(test_corpus_info['langs'].values())
                if lang in test_corpus_info['langs']
                else 0)

    str_experiment_output += "\n"

    str_experiment_output += "Language Pairs:\n"
    lang_pairs = set(list(train_corpus_info['lang_pairs'].keys())
                     + list(test_corpus_info['lang_pairs'].keys()))
    for lang_pair in lang_pairs:
        str_experiment_output += "{:^4}{:<20}{:<10,}{:<8.2%}{:<10,}{:8.2%}\n".format(
                "",
                lang_pair,
                train_corpus_info['lang_pairs'][lang_pair] 
                if lang_pair in train_corpus_info['lang_pairs']
                else 0,
                train_corpus_info['lang_pairs'][lang_pair] / sum(train_corpus_info['lang_pairs'].values())
                if lang_pair in train_corpus_info['lang_pairs']
                else 0,
                test_corpus_info['lang_pairs'][lang_pair] 
                if lang_pair in test_corpus_info['lang_pairs']
                else 0,
                test_corpus_info['lang_pairs'][lang_pair] / sum(test_corpus_info['lang_pairs'].values())
                if lang_pair in test_corpus_info['lang_pairs']
                else 0)

    print(str_experiment_output)
    return str_experiment_output

def create_corpora(corpora_dirpath):
    train_corpus, test_corpus = CorpusCSLangs(), CorpusCSLangs()
    train_corpus.ingest_corpus(corpora_dirpath / CORPUS_FILENAMES['train'])
    test_corpus.ingest_corpus(corpora_dirpath / CORPUS_FILENAMES['test'])
    return train_corpus, test_corpus

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create and evaluate a neural'
        + 'network based classifier for detecting code switching.') 
    parser.add_argument('corpora_dirpath', metavar='C', type=Path,
            help='Filepath to the corpora.')
    parser.add_argument('-o', '--output_dir', metavar='O', type=Path,
            help='Directory to store checkpoint and model files')
    parser.add_argument('-a', '--use_alphabets', action='store_true',
            help="Whether to use alphabetically based unknown character"
                + " vectors.")
    parser.add_argument('-e', '--epochs', metavar='E', type=int,
            help='Number of epochs to train')
    parser.add_argument('-b', '--batch_size', metavar='B', type=int,
            help='Size of batches on which to update gradient')
    parser.add_argument('-t', '--patience', metavar='T', type=int,
            help='Number of epochs to wait for improvement')



    args = parser.parse_args()
    main_args = {}
    main_args['corpora_dirpath'] = args.corpora_dirpath
    if args.output_dir:
        main_args['output_dirpath'] = args.output_dir
    if args.use_alphabets:
       main_args['use_alphabets'] = args.use_alphabets
    if args.epochs:
        main_args['epochs'] = args.epochs
    if args.batch_size:
        main_args['batch_size'] = args.batch_size
    if args.patience:
        main_args['patience'] = args.patience
       
    run(**main_args)
