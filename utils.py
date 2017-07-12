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
    def __init__(self, corpus_filepath, np_sentences=None, np_lang=None):
        """Transforms a corpus file into numerical data.

        Keyword arguments:
        corpus_filepath -- The filepath to a normalized corpus
        """
        # We need a set representing every character in the corpus as well
        # as an element representing an unknown character
        # We also need a set of labels for each word
        self.char2idx = defaultdict(int)
        self.idx2char = []

        # Create a dictionary of sentences to indices
        self.sentence2sidx = defaultdict(int)

        # Generate the Corpus
        self.read_corpus(corpus_filepath)

    def read_corpus(self, corpus_filepath):
        """Reads in a corpus file and sets the corpus variables.
    
        Keyword arguments:
        corpus_filepath -- The filepath to a normalized corpus
        """
        with open(corpus_filepath) as corpus_file:
            corpus_reader = csv.reader(corpus_file)

            # Skip the header
            next(corpus_reader)

            # Set the char and sentence index
            # Zero is reserved for padding
            cidx = 1
            self.idx2char.append(None)
            sidx = 1
            # Create the zero placeholder
            raw_sentences=[]
            raw_sentences.append([])
            #TODO: Try with and without padding
            raw_cs=[]
            raw_cs.append([])
            lang_stream = None
            self.maxwordlen = 0
            self.maxsentlen = 0
            for row in corpus_reader:
                word_string = row[1]
                # TODO: This puts a max word length on a word
                # Length arbitrary based on
                # "supercalifragilisticexpialidocious"
                if len(word_string) > 34:
                    continue
                self.maxwordlen = max(self.maxwordlen, len(word_string))
                lang = row[2]
                if lang_stream == None:
                    lang_stream = lang
                    cs = 1
                elif lang != 'other' and lang != lang_stream:
                    lang_stream = lang
                    cs = 2
                else:
                    cs = 1
                word = []
                for c in word_string:
                    if c not in self.char2idx:
                        self.char2idx[c] = cidx
                        cidx += 1
                        self.idx2char.append(c)
                    word.append(self.char2idx[c])

                # Remove the word id at the end of the sentence name
                sname = ''.join(row[0].split(sep='_')[0:3])

                if sname not in self.sentence2sidx:
                    self.sentence2sidx[sname] = sidx
                    sidx +=1
                    raw_sentences.append([])
                    raw_cs.append([])

                nsidx = self.sentence2sidx[sname]
                raw_sentences[nsidx].append(word)
                raw_cs[nsidx].append(cs)

        # Add one more index for unseen chars
        # TODO: How do I make sure the frequencies are right during training?
        self.char2idx['unk'] += 1
        self.idx2char.append('unk')

        # Figure out the maximum sentence length in the list of sentences
        for sentence in raw_sentences:
            self.maxsentlen = max(self.maxsentlen, len(sentence))

        # Now let's pad the corpus
        # Pad the sentences and langs
        list_sentences = ([[word + [0]*(self.maxwordlen-len(word)) 
            for word in sentence] +
            [[0]*self.maxwordlen]*(self.maxsentlen-len(sentence)) 
            for sentence in raw_sentences])
        del raw_sentences
        list_cs = ([[cs for cs in sentcs] + 
            [0]*(self.maxsentlen-len(sentcs)) for sentcs in raw_cs])
        del raw_cs

        # Finally convert the sentence and lang ids to numpy arrays
        self.sentences = np.array(list_sentences)
        del list_sentences
        one_hot_cs = ([to_categorical(sentcs, num_classes=3) 
            for sentcs in list_cs])
        del list_cs
        self.cs = np.array(one_hot_cs)
        del one_hot_cs


    def print_sentences(self):
        """Prints all sentences in the corpus."""
        for sentence in self.sentences:
            self.print_sentence(sentence)

    def print_sentences_langs(self):
        """Prints all sentences in the corpus."""
        for sentence,langs in zip(self.sentences, self.langs):
            self.print_sentence(sentence)
            self.print_lang(langs)

    def print_sentence(self, sentence):
        """Prints a sentence.

        Keyword arguments:
        sentence -- An array of arrays of character indices
        """

        sentence_string = ""
        for word_indices in sentence:
            word = ""
            for char_idx in word_indices:
                char = self.idx2char[char_idx]
                if char != None:
                    word += char
            sentence_string += word + " " 
        print(sentence_string)
        return sentence_string

    def print_lang(self, langs):
        """Prints the language classifications for a sentence.

        Keyword arguments:
        sentence -- An array of arrays of character indices
        """
        lang_string = ""
        for lang_index in langs:
            lang_string += self.idx2lang[lang_index] + " " 
        print(lang_string)
        return lang_string
