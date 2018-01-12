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
import re

from collections import defaultdict
from math import sqrt, floor
from pathlib import Path

import numpy as np

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

    print('\n' + np.array_str(y_test))

    print(np.array_str(y_pred) + '\n')

    for seq_idx in range(len(y_test)):
        if len(y_test[seq_idx]) != len(y_pred[seq_idx]):
            raise Exception("Test and Pred tokens have different lengths:" + str(len(y_test[seq_idx])) + " " + str(
                len(y_pred[seq_idx])))

        for i in range(len(y_test[seq_idx])):
            pos_test = y_test[seq_idx][i]
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
    results['confusion_matrix'] = confusion_matrix.tolist()
    results['precision'] = precision.tolist()
    results['recall'] = recall.tolist()
    results['fscore'] = fscore.tolist()
    return results

def get_sentence_id(word_id):
    return "_".join(word_id.split(sep='_')[:-1])

def get_label_dicts(labels):
    idx2label = ['<pad>'] + labels
    label2idx = {idx2label[i] : i for i in range(len(idx2label))}
    return idx2label, label2idx

def get_char_dicts(chars, use_alphabets):
    idx2char = ['<pad>'] + chars

    if use_alphabets:
        # Add indices for unseen chars for each alphabet representable 
        # by unicode
        for a in alphabets:
            idx2char.append('<unk' + a + '>')
    # Finally add a generic unknown character
    idx2char.append('<unk>')

    char2idx = {idx2char[i] : i for i in range(len(idx2char))}

    return idx2char, char2idx
