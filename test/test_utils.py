#!/usr/bin/python3

import unittest
import csv
import random

import utils
from utils import Corpus, Corpus_CS_Langs
from collections import defaultdict
from collections import Counter
from pathlib import Path

import numpy as np


word_col = 1
dl = ','






class UtilsTestCase(unittest.TestCase):

    def setUp(self):
        self.corpora_filepaths = ['test/data/Corpus_corpus_de+ar.csv',
            'test/data/Corpus_corpus_fr+ar.csv']

        pass

    def tearDown(self):
        pass

    def test_randomly_read_CS_Langs_Corpus_comb(self):
       train_corpus = Corpus_CS_Langs(train=True)
       test_corpus = Corpus_CS_Langs()
       comb_corpus = Corpus_CS_Langs()
       for corpus in self.corpora_filepaths:
           utils.randomly_read_Corpus_CS_Langs(corpus, train_corpus, test_corpus)
           temp_corpus = Corpus_CS_Langs()
           temp_corpus.read_corpus(corpus, dl=',')
           comb_corpus += temp_corpus
       self.assertEqual(len(train_corpus.sentences) 
               + len(test_corpus.sentences), len(comb_corpus.sentences))

    def test_randomly_read_CS_Langs_Corpus_split(self):
       train_corpus = Corpus_CS_Langs(train=True)
       test_corpus = Corpus_CS_Langs()
       for corpus in self.corpora_filepaths:
           utils.randomly_read_Corpus_CS_Langs(corpus, train_corpus, test_corpus)
       self.assertAlmostEqual(len(train_corpus.sentences) , 9, delta=2)
       self.assertAlmostEqual(len(test_corpus.sentences) , 1, delta=2)

    '''
    def test_randomly_split_corpus_len_sentences(self):
        train_corpus, test_corpus = self.corpus1.randomly_split_corpus()
        self.assertEqual(len(self.corpus1.sentences), len(train_corpus.sentences) + len(test_corpus.sentences))

    def test_randomly_split_corpus_len_labels(self):
        train_corpus, test_corpus = self.corpus1.randomly_split_corpus()
        print(len(train_corpus.labels))
        print(len(test_corpus.labels))
        self.assertEqual(len(self.corpus1.labels), len(train_corpus.labels) + len(test_corpus.labels))

    def test_randomly_split_corpus_reconstitute_labels(self):
        train_corpus, test_corpus = self.corpus1.randomly_split_corpus()
        self.assertEqual(sorted(self.corpus1.labels), 
            sorted(train_corpus.labels + test_corpus.labels))

    def test_randomly_split_corpus_reconstitute_sentences(self):
        train_corpus, test_corpus = self.corpus1.randomly_split_corpus()
        self.assertEqual(sorted(self.corpus1.sentences), 
            sorted(train_corpus.sentences + test_corpus.sentences))
    '''


