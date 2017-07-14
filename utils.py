#!/usr/bin/python3

# Design of utils file and Corpus class inspired by 
# Columbia University
# COMS 4705W 2017 
# Assignment 4
# Professor Karl Stratos

import csv
import os

from collections import defaultdict

import numpy as np
from keras.utils import to_categorical

class Corpus():
    def __init__(self, dictionary=(None, None)):
        """Transforms a corpus file into numerical data.

        Keyword arguments:
        corpus_filepath -- The filepath to a normalized corpus
        """
        # Set the dictionary if one is provided 
        self.char2idx, self.idx2char = dictionary

        # We also need a set of labels for each word
        self.label2idx = {'<PAD>':0, 'no_cs': 1, 'cs':2}
        self.idx2label = {i:l for l, i in self.label2idx.items()}

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

            self.sentences=[]
            self.labels=[]
            self.lang_stream = None
            self.maxwordlen = 0
            self.maxsentlen = 0
            self.sentence2sidx = {}
            self.sidx = 0
            for row in corpus_reader:
                self.read_row(row)

        # Figure out the maximum sentence length in the list of sentences
        for sentence in self.sentences:
            self.maxsentlen = max(self.maxsentlen, len(sentence))

    def read_row(self, row):
           word = row[1]
           # TODO: This puts a max word length on a word
           # Length arbitrary based on
           # len("supercalifragilisticexpialidocious")
           if len(word) > 34:
               return
           self.maxwordlen = max(self.maxwordlen, len(word))
           lang = row[2]
           label = self.label_word(lang)

           # Remove the word id at the end of the sentence name
           sname = ''.join(row[0].split(sep='_')[0:3])

           if sname not in self.sentence2sidx:
               self.sentence2sidx[sname] = self.sidx
               self.sidx +=1
               self.sentences.append([])
               self.labels.append([])

           nsidx = self.sentence2sidx[sname]
           self.sentences[nsidx].append(word)
           self.labels[nsidx].append(label)

    def label_word(self, lang):
       if self.lang_stream == None:
           self.lang_stream = lang
           label = 'no_cs'
       if lang != 'other' and lang != self.lang_stream:
           self.lang_stream = lang
           label = 'cs'
       else:
           label = 'no_cs'

    def np_idx_conversion(self, maxsentlen, maxwordlen):
        # Convert the sentences and labels to lists of indices
        list_sentences, list_labels = self.idx_conversion(maxsentlen, maxwordlen)
        # Finally convert the sentence and label ids to numpy arrays
        np_sentences = np.array(list_sentences)
        del list_sentences
        np_slabels = np.array(list_labels)
        del list_labels
        return np_sentences, np_slabels

    def idx_conversion(self, maxsentlen, maxwordlen):
        # Convert words to indices 
        # And pad the sentences and labels
        if self.idx2char == None or self.char2idx == None:
            self.create_dictionary()
        list_sentences = ([[[(self.char2idx[c] if c in self.char2idx 
                         else self.char2idx['unk'])
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

        return list_sentences, list_labels

    def create_dictionary(self):
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

        # Add one more index for unseen chars
        # TODO: How do I make sure the frequencies are right during training?
        self.char2idx['unk'] += 1
        self.idx2char.append('unk')
        
        return self.char2idx, self.idx2char


class Corpus_Aaron(Corpus):
    def __init__(self, dictionary=(None, None)):
        """Reads in a corpus file and sets the corpus variables.
    
        Keyword arguments:
        corpus_filepath -- The filepath to a normalized corpus
        """
        Corpus.__init__(self)
        self.label2idx = ({'<PAD>':0, 'lang1': 1, 'lang2':2, 'other':3, 'ne':4, 
            'ambiguous':5, 'fw':6, 'mixed':7, 'unk':8})
        self.idx2label = {i:l for l, i in self.label2idx.items()}

    def label_word(self, lang):
        return lang

def print_np_sentences(np_sentences, idx2char):
    """Prints all sentences in the corpus."""
    for np_sentence in np_sentences:
        print_sentence(np_sentence, idx2char)

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


