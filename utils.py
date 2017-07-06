#!/usr/bin/python3

# Design of utils file and Corpus class inspired by 
# Columbia University
# COMS 4705W 2017 
# Assignment 4
# Professor Karl Stratos

import csv

class Corpus():
    def __init__(self, corpus_filepath)
    """Transforms a corpus file into numerical data.

    Keyword arguments:
    corpus_filepath -- The filepath to a normalized corpus
    """


        # We need a set representing every character in the corpus as well
        # as an element representing an unknown character
        # We also need a set of labels for each word
        self.char2idx = defaultdict(int)
        self.idx2char = []

        # Create values for each language type
        self.lang2idx = {'lang1':0, 'lang2':1, 'other':2}

        # Create a dictionary of sentences to indices
        self.sentence2sidx = defaultdict(int)
        # Create a list where each entry is a list of (word, lang) values in
        # the sentence
        self.sentences = []

        # Generate the corpus
        read_corpus(corpus_filepath)

    def read_corpus(self)
    """Reads in a corpus file and sets the corpus variables.
    
    Keyword arguments:
    corpus_filepath -- The filepath to a normalized corpus
    """
        with open(corpus_filepath) as corpus_file:
            corpus_reader = csv.reader(corpus_file)

            # Skip the header
            next(corpus_reader(None))
            for row in corpus_reader:
                word_string = row[1]
                lang = self.lang2idx[row[2]]
                word = []
                for c in word_string:
                    if c not in self.char2idx:
                        self.char2idx[c] += 1
                        self.idx2char.append(c)
                    word.append(self.char2idx[c])

                # Remove the word id at the end of the sentence name
                sname = row[0].split(sep='_')[0,3]
                if sname not in self.sentence2sidx:
                    self.sentence2sidx[sname] += 1
                    self.sentences[self.sentence2sidx[sname]] = []
                sidx = self.sentence2sidx[sname]
                self.sentences[sidx].append((word, lang))

