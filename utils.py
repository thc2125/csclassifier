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


from collections import defaultdict
from collections import Counter
from math import sqrt
from random import random

import numpy as np
from keras.utils import to_categorical

from alphabet_detector import AlphabetDetector

with open('unicode_alphabets.txt', 'r') as uc_file:
    alphabets = set([alphabet for alphabet in uc_file])

class Corpus():
    def __init__(self, char_dictionary=(None, None), 
        label_dictionary=(None, None), train=False):
        """Reads in a corpus file and sets the corpus variables.
    
        Keyword arguments:
        char_dictionary -- A tuple of dictionaries for characters to indices 
                           and indices to characters
        label_dictionary -- A tuple of dictionaries for labels to indices 
                           and indices to labels
        """

        # Set the dictionary if one is provided 
        self.char2idx, self.idx2char = char_dictionary

        # We also need a set of labels for each word
        self.label2idx, self.idx2label = label_dictionary
        self.train = train
        self.__init_data()

    def __init_data(self):
        self.sentences=[]
        self.labels=[]
        self.maxwordlen = 0
        self.maxsentlen = 0
        self.sentence2sidx = {}
        self.sidx = 0
        self.char_frequency = defaultdict(int)


    # TODO: Need to fix this so that tests don't fail
    def __add__(self, other):
        corp = Corpus()
        return self.__combine(corp, other)
        

    def __combine(self, corp, other):
        corp.sidx = len(self.sentences)
        corp.sentences = self.sentences + other.sentences
        corp.labels = self.labels + other.labels
        corp.sentence2sidx = {s : (i + self.sidx) for s, i in
                other.sentence2sidx.items()}
        corp.sidx = len(corp.sentences)
        corp.maxsentlen = max(self.maxsentlen, other.maxsentlen)
        corp.maxwordlen = max(self.maxwordlen, other.maxwordlen)
        # TODO: Is it okay to turn a defaultdict into a counter?
        corp.char_frequency = (Counter(self.char_frequency) 
            + Counter(other.char_frequency))
        return corp


    def read_corpus(self, corpus_filepath, dl):
        """Reads in a corpus file and sets the corpus variables.
    
        Keyword arguments:
        corpus_filepath -- The filepath to a normalized corpus
        """
        self.corpus_filepath = corpus_filepath

        with open(corpus_filepath) as corpus_file:
            corpus_reader = csv.reader(corpus_file, delimiter=dl)

            # Skip the header
            next(corpus_reader)

            for row in corpus_reader:
                self.read_row(row)

        self.char_frequency = Counter(self.char_frequency)

        # Figure out the maximum sentence length in the list of sentences
        for sentence in self.sentences:
            self.maxsentlen = max(self.maxsentlen, len(sentence))

    def read_row(self, row):
        """Reads a csv row and updates the Corpus variables.
    
        Keyword arguments:
        row -- a list of csv row values ([sentence_id, word, lang_label,...])
        """

        word = row[1]
        # TODO: This puts a max word length on a word
        # Length arbitrary based on
        # len("supercalifragilisticexpialidocious")
        if len(word) > 34:
           return
        self.maxwordlen = max(self.maxwordlen, len(word))
           
        label = self.label_word(row[2])

        # Remove the word id at the end of the sentence name
        sname = ''.join(row[0].split(sep='_')[0:4])

        if sname not in self.sentence2sidx:
           self.add_sentence(sname)

        nsidx = self.sentence2sidx[sname]
        self.sentences[nsidx].append(word)
        # Get the character frequency for a word.
        for c in word:
           self.char_frequency[c] += 1
        self.labels[nsidx].append(label)

    def add_sentence(self, sname):
           self.sentence2sidx[sname] = self.sidx
           self.sidx +=1
           self.sentences.append([])
           self.labels.append([])

    def label_word(self, label):
        return label

    def np_idx_conversion(self, maxsentlen, maxwordlen):
        # Convert the sentences and labels to lists of indices
        self.ad = AlphabetDetector()
        list_sentences, list_labels, list_labels_weights = (
            self.idx_conversion(maxsentlen, maxwordlen))
        # Finally convert the sentence and label ids to numpy arrays
        np_sentences = np.array(list_sentences)
        np_labels_weights = np.array(list_labels_weights)
        del list_sentences
        del list_labels_weights
        np_slabels = np.array(list_labels)
        del list_labels
        return np_sentences, np_slabels, np_labels_weights

    def idx_conversion(self, maxsentlen, maxwordlen):
        # Convert words to indices 
        # And pad the sentences and labels
        ab = AlphabetDetector()
        if self.idx2char == None or self.char2idx == None:
            self.create_dictionary()
        # Create a list of lists of lists of indices
        # Randomly assign some letters the index of unknown characters for a 
        # given alphabet
        list_sentences = ([[[(self.char2idx[c] 
                         if c in self.char2idx and not self.unk_replace(c)
                         else self.char2idx['unk' + ab.detect_alphabet(c)[0]])
                     for c in word]
                + [0]*(maxwordlen-len(word)) 
            for word in sentence]
                + [[0]*maxwordlen]*(maxsentlen-len(sentence)) 
            for sentence in self.sentences])

        list_cat_labels = ([[self.label2idx[label] for label in sentlabels] 
                + [0] * (maxsentlen-len(sentlabels)) 
            for sentlabels in self.labels])
        # Make labels one-hot
        list_labels = ([to_categorical(sentlabels, 
                num_classes=len(self.label2idx)) 
            for sentlabels in list_cat_labels])

        list_labels_weights = ([[(1 if label != 0 else 0) for label in list_slabels] 
            for list_slabels in list_cat_labels])


        return list_sentences, list_labels, list_labels_weights

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
        if random() > p:
            return False
        else:
            return True

    def create_dictionary(self, use_alphabets=False):
        self.idx2char = []
        # Set the zero index to the null character
        self.idx2char.append(0)
        self.char2idx = defaultdict(int)
        # set the null character index to zero
        self.char2idx[0] = 0

        for sentence in self.sentences:
            for word in sentence:
                for c in word:
                    if c not in self.char2idx:
                        self.char2idx[c] = len(self.idx2char)
                        self.idx2char.append(c)


        if use_alphabets:
            # Add indices for unseen chars for each alphabet representable 
            # by unicode
            for a in alphabets:
                self.char2idx['unk' + a] += len(self.idx2char)
                self.idx2char.append('unk' + a)
        else:
            self.char2idx['unk'] += len(self.idx2char)
            self.idx2char.append('unk')

        
        return self.char2idx, self.idx2char

    #def split_corpus(self

class Corpus_Aaron(Corpus):
    def __init__(self, char_dictionary=(None, None), label_dictionary=(None, None)):
        """Reads in a corpus file and sets the corpus variables.
    
        Keyword arguments:
        char_dictionary -- A tuple of dictionaries for characters to indices 
                           and indices to characters
        label_dictionary -- A tuple of dictionaries for labels to indices 
                           and indices to labels

        """
        label2idx = ({'<PAD>':0, 'lang1': 1, 'lang2':2, 'other':3, 'ne':4, 
        'ambiguous':5, 'fw':6, 'mixed':7, 'unk':8})
        idx2label = {i:l for l, i in self.label2idx.items()}

        Corpus.__init__(self, label_dictionary=(label2idx, idx2label))

    def __add__(self, other):
        corp = Corpus_Aaron()
        return self.__combine(corp, other)


class Corpus_CS_Langs(Corpus):
    def __init__(self, char_dictionary=(None, None), train=False):
        """Reads in a corpus file and sets the corpus variables.
    
        Keyword arguments:
        dictionary -- A tuple of dictionaries for characters to indices and
                      indices to characters
        """
        label2idx = {'<PAD>':0, 'no_cs': 1, 'cs':2}
        idx2label = {i:l for l, i in self.label2idx.items()}

        Corpus.__init__(self, char_dictionary,
                label_dictionary=(label2idx,idx2label))

    def __add__(self, other):
        corp = Corpus_CS_Langs()
        return self.__combine(corp, other)


    def label_word(self, label):
       if self.lang_stream == None:
           self.lang_stream = label
           return 'no_cs'
       elif (label != 'other' and label != self.lang_stream):
           self.lang_stream = label
           return 'cs'
       else:
           return 'no_cs'

    def read_corpus(self, corpus_filepath, dl):
        self.lang_stream = None
        Corpus.read_corpus(corpus_filepath, dl)

    def add_sentence(self, sname):
        Corpus.add_sentence(sname)
        # Note that the corpus must have words in sentences ordered and
        # row adjacent
        self.lang_stream = None

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

