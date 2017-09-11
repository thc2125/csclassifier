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
from collections import Counter
from math import sqrt, floor
from pathlib import Path
from copy import deepcopy

import numpy as np
from keras.utils import to_categorical

from alphabet_detector import AlphabetDetector
from unicode_alphabets import alphabets


class Corpus():
    def __init__(self, char_dictionary=(None, None), 
        label_dictionary=(None, None), train=False, use_alphabets=False):
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
        self.use_alphabets = use_alphabets
        if use_alphabets:
            self.ad = AlphabetDetector()
        self._init_data()

    def _init_data(self):
        self.sentences=[]
        self.labels=[]
        self.label_counts = defaultdict(int)
        self.maxwordlen = 0
        self.maxsentlen = 0
        self.sentence2sidx = {}
        self.sidx = 0
        self.char_frequency = defaultdict(int)



    def __add__(self, other):
        #TODO: add an exception if the two corpora have any common sentences
        corp = Corpus()
        return self._combine(corp, other)
        

    def _combine(self, corp, other):
        corp.sidx = len(self.sentences)
        corp.sentences = self.sentences + other.sentences

        corp.labels = self.labels + other.labels

        for k, v in self.label_counts.items():
            corp.label_counts[k] += v
        for k, v in other.label_counts.items():
            corp.label_counts[k] += v

        corp.sentence2sidx = self.sentence2sidx.copy() 
        corp.sentence2sidx.update({s : (i + self.sidx) for s, i in 
                other.sentence2sidx.items()})
        corp.sidx = len(corp.sentences)
        corp.maxsentlen = max(self.maxsentlen, other.maxsentlen)
        corp.maxwordlen = max(self.maxwordlen, other.maxwordlen)
        # TODO: Is it okay to turn a defaultdict into a counter?
        corp.char_frequency = (Counter(self.char_frequency) 
            + Counter(other.char_frequency))
        corp.train = True if (self.train or other.train) else False
        corp.use_alphabets = self.use_alphabets or other.use_alphabets

        return corp


    def read_corpus(self, corpus_filepath, dl):
        """Reads in a corpus file and sets the corpus variables.
    
        Keyword arguments:
        corpus_filepath -- The filepath to a normalized corpus
        """
        self.corpus_filepath = corpus_filepath
        print(corpus_filepath)
        with open(corpus_filepath) as corpus_file:
            corpus_reader = csv.reader(corpus_file, delimiter=dl, doublequote=False)

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
        # NOTE: This puts a max word length on a word
        # Length arbitrary based on
        # len("supercalifragilisticexpialidocious")
        if len(word) > 34:
           return
        self.maxwordlen = max(self.maxwordlen, len(word))
           

        # Remove the word id at the end of the sentence name
        sname = ''.join(row[0].split(sep='_')[:-1])

        if sname not in self.sentence2sidx:
           self.add_sentence(sname)

        label = self.label_word(row[2])

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
        self.label_counts[label] += 1
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
        if self.idx2char == None or self.char2idx == None:
            self.create_dictionary()
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
        if random.random() > p:
            return True
        else:
            return False

    def get_unk(self, c):
        unk = 'unk'
        if self.use_alphabets:
            print(c)
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

    def randomly_split_corpus(self, split=.9, new_corpus1=None, new_corpus2=None):
        
        if not new_corpus1:
            new_corpus1 = Corpus(train=True, use_alphabets=self.use_alphabets)
        if not new_corpus2:
            new_corpus2 = Corpus()
        self._split(split=split, new_corpus1=new_corpus1, 
            new_corpus2=new_corpus2)
        return new_corpus1, new_corpus2

    def _split(self, split, new_corpus1, new_corpus2):
        sentence2sidx = list(deepcopy(self.sentence2sidx).items())
        random.shuffle(sentence2sidx)
        split_point = floor(split*len(sentence2sidx))
        for sname, idx in sentence2sidx[:split_point]:
                new_corpus1.ext_add_sentence(self.sentences[idx], 
                    self.labels[idx], sname)
        for sname, idx in sentence2sidx[split_point:]:
                new_corpus2.ext_add_sentence(self.sentences[idx], 
                    self.labels[idx], sname)

    def ext_add_sentence(self, sentence, labels, sname):
        self.sentence2sidx[sname]=len(self.sentences)

        self.sentences.append(sentence)
        for word in sentence:
            for c in word:
                self.char_frequency[c] += 1

        self.labels.append(labels)

        self.maxsentlen = max(self.maxsentlen, len(sentence))
        self.maxwordlen = max(self.maxwordlen, 
            max([len(w) for w in sentence]))
        self.sidx = len(self.sentences)
                
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
        return Corpus._combine(self, corp, other)


class Corpus_CS_Langs(Corpus):
    def __init__(self, char_dictionary=(None, None), train=False, use_alphabets=False):
        """Reads in a corpus file and sets the corpus variables.
    
        Keyword arguments:
        dictionary -- A tuple of dictionaries for characters to indices and
                      indices to characters
        """
        label2idx = {'<PAD>':0, 'no_cs': 1, 'cs':2}
        idx2label = {i:l for l, i in label2idx.items()}

        Corpus.__init__(self, char_dictionary,
                label_dictionary=(label2idx,idx2label), train=train, 
                 use_alphabets=use_alphabets)

    def _init_data(self):
        self.switch_count = 0
        self.switch_label_counts = defaultdict(int)
        self.multilingual_sentence_count = 0
        Corpus._init_data(self)

    def __add__(self, other):
        corp = Corpus_CS_Langs()
        return self._combine(corp, other)

    def _combine(self, corp, other):
        # NOTE: The proper functioning of this depends on both corpora being 
        # composed of completely different sentences.
        corp.switch_count += other.switch_count 
        corp.multilingual_sentence_count += other.multilingual_sentence_count

        for k, v in self.switch_label_counts.items():
            corp.switch_label_counts[k] += v
        for k, v in other.label_counts.items():
            corp.switch_label_counts[k] += v


        return Corpus._combine(self, corp, other)

    def add_sentence(self, sname):
       # Lang_stream needs to be reset for every new sentence
       self.lang_stream = None
       # Another flag variable to indicate if we've seen a code-switch yet in 
       # this sentence
       self.first_cs = False
       Corpus.add_sentence(self, sname)

    def label_word(self, label):
       Corpus.label_word(self, label)
       if self.lang_stream == None and label != 'other' and label != 'punct':
           self.lang_stream = label
           return 'no_cs'

       elif (label != 'other' and label != 'punct' and label != self.lang_stream):
           self.switch_label_counts[self.lang_stream + " to " + label] += 1
           self.lang_stream = label
           self.switch_count += 1 
           if not self.first_cs:
               self.first_cs = True
               self.multilingual_sentence_count += 1
           return 'cs'

       else:
           return 'no_cs'

    def read_corpus(self, corpus_filepath, dl):
        self.lang_stream = None
        Corpus.read_corpus(self, corpus_filepath, dl)

    def randomly_split_corpus(self, split=.9):
        new_corpus1 = Corpus_CS_Langs(train=True, use_alphabets=self.use_alphabets)
        new_corpus2 = Corpus_CS_Langs(use_alphabets=self.use_alphabets)
        return Corpus.randomly_split_corpus(self, split=split, 
            new_corpus1=new_corpus1, new_corpus2=new_corpus2)

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

