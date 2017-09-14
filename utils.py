#!/usr/bin/python3

# Design of utils file and Corpus class inspired by 
# Columbia University
# COMS 4705W 2017 
# Assignment 4
# Professor Karl Stratos

# Code for accuracy metrics provided by
# Victor Soto Martinez

import csv
import os
import sys
import random

from collections import defaultdict
from math import sqrt, floor
from pathlib import Path
from copy import deepcopy

import numpy as np
from keras.utils import to_categorical

from alphabet_detector import AlphabetDetector
from unicode_alphabets import alphabets


                

           

def print_np_sentences(np_sentences, idx2char):
    """Prints all sentences in the corpus."""
    for np_sentence in np_sentences:
        print_np_sentence(np_sentence, idx2char)

def print_np_labels(np_slabels, idx2label):
    """Prints all sentences in the corpus."""
    for np_labels in np_slabels:
        print_np_label(np_labels, idx2label)

def print_np_sentences_np_gold_pred_labels(np_sentences, np_gold_slabels, np_pred_slabels, idx2char, idx2label):
    """Prints all sentences in the corpus."""
    for np_sentence, np_gold_labels, np_pred_labels in zip(np_sentences, np_gold_slabels, np_pred_slabels):
        print_np_sentence(np_sentence, idx2char)
        print_np_label(np_gold_labels, idx2label)
        print_np_label(np_pred_labels, idx2label)

def print_np_sentences_np_labels(np_sentences, np_slabels, idx2char, idx2label):
    """Prints all sentences in the corpus."""
    for np_sentence, np_labels in zip(np_sentences, np_slabels):
        print_np_sentence(np_sentence, idx2char)
        print_np_label(np_labels, idx2label)

def print_np_sentence(np_sentence, idx2char):
    """Prints a sentence.

    Keyword arguments:
    sentence -- An array of arrays of character indices
    """
    sentence_string = ""
    for word_indices in np_sentence:
        word = ""
        for char_idx in word_indices:
            char = idx2char[char_idx]
            if char != 0:
                word += char
        sentence_string += word + " " 
    print(sentence_string)
    return sentence_string

def print_np_label(np_labels, idx2label):
    """Prints the language classifications for a sentence.

    Keyword arguments:
    sentence -- An array of arrays of character indices
    """
    label_string = ""
    for label_index in np_labels:
        label_string += idx2label[np.argmax(label_index)] + " " 
    print(label_string)
    return label_string

def compute_accuracy_metrics(y_test, y_pred, list_tags):
    tagset_size = len(list_tags)
    num_tokens = 0
    num_eq = 0
    confusion_matrix = np.zeros((tagset_size, tagset_size))
    true_pos = np.zeros(tagset_size)
    false_pos = np.zeros(tagset_size)
    false_neg = np.zeros(tagset_size)

    for seq_idx in range(len(y_test)):
        if len(y_test[seq_idx]) != len(y_pred[seq_idx]):
            raise Exception("Test and Pred tokens have different lengths:" + str(len(y_test[seq_idx])) + " " + str(
                len(y_pred[seq_idx])))

        for i in range(len(y_test[seq_idx])):
            pos_test = y_test[seq_idx][i]
            if pos_test != list_tags['<PAD>']:
                pos_pred = y_pred[seq_idx][i]
                num_tokens += 1
                if pos_test == pos_pred:
                    num_eq += 1
                    true_pos[pos_test] += 1
                else:
                    false_neg[pos_test] += 1
                    false_pos[pos_pred] += 1
                confusion_matrix[pos_test, pos_pred] += 1

    accuracy = num_eq * 1.0 / num_tokens

    recall = np.zeros(tagset_size)
    precision = np.zeros(tagset_size)
    fscore = np.zeros(tagset_size)
    for idx in range(tagset_size):
        if false_neg[idx] + true_pos[idx] == 0:
            recall[idx] = 1.0
        else:
            recall[idx] = true_pos[idx] * 1.0 / (true_pos[idx] + false_neg[idx])
        if true_pos[idx] + false_pos[idx] == 0:
            precision[idx] = 1.0
        else:
            precision[idx] = true_pos[idx] * 1.0 / (true_pos[idx] + false_pos[idx])
        if recall[idx] + precision[idx] == 0.0:
            fscore[idx] = 0.0
        else:
            fscore[idx] = 2.0 * recall[idx] * precision[idx] / (recall[idx] + precision[idx])

    results = dict()
    results['accuracy'] = accuracy
    results['confusion_matrix'] = confusion_matrix
    results['precision'] = precision
    results['recall'] = recall
    results['fscore'] = fscore
    return results

def randomly_read_Corpus_CS_Langs(corpus_filepath, train_corpus, test_corpus, 
        dl=',', test_split=.1):
    with open(corpus_filepath) as corpus_file:
        corpus_reader = csv.reader(corpus_file, delimiter=dl)
        # Skip the header
        next(corpus_reader)
        # Create a set of sentences belonging to train and test
        train_sent = set()
        test_sent = set()
        for row in corpus_reader:
            sname = ''.join(row[0].split(sep='_')[:-1])
            if sname in train_sent:
                train_corpus.read_row(row)
            elif sname in test_sent:
                test_corpus.read_row(row)
            elif random.random() > test_split:
                train_corpus.read_row(row)
                train_sent.add(sname)
            else:
                test_corpus.read_row(row)
                test_sent.add(sname)

def np_idx_conversion(self, maxsentlen, maxwordlen):
        # Convert the sentences and labels to lists of indices
        self.ad = AlphabetDetector()
        # Convert words to indices 
        # And pad the sentences and labels
        if self.idx2char == None or self.char2idx == None:
            self.create_dictionary()

        list_sentences = self.sent_idx_conversion(maxsentlen, maxwordlen)

        list_labels, list_labels_weights = self.label_idx_conversion( 
            maxsentlen, maxwordlen)

        # Finally convert the sentence and label ids to numpy arrays
        np_sentences = np.array(list_sentences)
        del list_sentences
        np_labels_weights = np.array(list_labels_weights)
        del list_labels_weights
        np_slabels = np.array(list_labels)
        del list_labels
        return np_sentences, np_slabels, np_labels_weights

def sent_idx_conversion(self, maxsentlen, maxwordlen):
        # Create a list of lists of lists of indices
        # Randomly assign some letters the index of unknown characters for a 
        # given alphabet
        list_sentences = ([[[(self.char2idx[c] 
                         if (c in self.char2idx and not self.unk_replace(c))
                         else self.char2idx[self.get_unk(c)])
                     for c in word]
                + [0]*(maxwordlen-len(word)) 
            for word in sentence]
                + [[0]*maxwordlen]*(maxsentlen-len(sentence)) 
            for sentence in self.sentences])
        return list_sentences

def label_idx_conversion(self, maxsentlen, maxwordlen):
        list_cat_labels = ([[self.label2idx[label] for label in sentlabels] 
                + [0] * (maxsentlen-len(sentlabels)) 
            for sentlabels in self.labels])
        # Make labels one-hot
        list_labels = ([to_categorical(sentlabels, 
                num_classes=len(self.label2idx)) 
            for sentlabels in list_cat_labels])

        list_labels_weights = ([[(1 if label != 0 else 0) for label in list_slabels] 
            for list_slabels in list_cat_labels])

        return list_labels, list_labels_weights
 

def unk_replace(self, c):
        # Formula sourced from Quora:
        # https://www.quora.com/How-does-sub-sampling-of-frequent-words-work-in-the-context-of-Word2Vec
        # "Improving Distributional Similarity with Lessons Learned from Word Embeddings"
        # Levy, Goldberg, Dagan
        if not self.train:
            return False

        t = .00001
        f = self.char_frequency[c]
        p = 1 - sqrt(t/f)
        if random.random() > p:
            return True
        else:
            return False

def get_unk(self, c):
        unk = 'unk'
        if self.use_alphabets:
            alph = list(self.ad.detect_alphabet(c))
            if alph and alph[0] in alphabets:
                unk += alph[0]
        return unk

def create_dictionary(self):
        self.idx2char = []
        # Set the zero index to the null character
        self.idx2char.append('\0')
        self.char2idx = defaultdict(int)
        # set the null character index to zero
        self.char2idx['\0'] = 0

        for sentence in self.sentences:
            for word in sentence:
                for c in word:
                    if c not in self.char2idx:
                        self.char2idx[c] = len(self.idx2char)
                        self.idx2char.append(c)


        if self.use_alphabets:
            # Add indices for unseen chars for each alphabet representable 
            # by unicode
            for a in alphabets:
                self.char2idx['unk' + a] += len(self.idx2char)
                self.idx2char.append('unk' + a)
        # Finally add a generic unknown character
        self.char2idx['unk'] += len(self.idx2char)
        self.idx2char.append('unk')

        return self.char2idx, self.idx2char


