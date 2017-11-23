#!/usr/bin/python3

import string
import unittest

from collections import Counter

import numpy as np

import corpus_utils

from corpus import Corpus

class CorpusUtilsTestCase(unittest.TestCase):

    def setUp(self):
        self.idx2char = ['<pad>'] + list(string.ascii_letters) + ['<unk>']
        self.char2idx = {self.idx2char[i] : i 
                         for i in range(len(self.idx2char))}

        self.idx2label = ['<pad>', 'label1', 'label2']
        self.label2idx = {'<pad>' : 0, 'label1' : 1, 'label2' : 2}

        self.sentences = [["this", "is", "a", "test"], 
                          ["this", "is", "another", "one"]]
        self.slabels = [['label1', 'label2', 'label2', 'label1'],
                       ['label2', 'label2', 'label1', 'label1']]


        self.maxsentlen = 4
        self.maxwordlen = 7

        self.char_count = Counter({'t': 5, 'h' : 3, 'i' : 4, 's': 5, 'a' : 1, 'e' : 2, 'n' : 2, 'o' : 2, 'r' : 1})

        self.corp = Corpus()
        self.corp.sentences = self.sentences
        self.corp.slabels = self.slabels
        self.corp.char_count = self.char_count

    def tearDown(self):
        pass

    def test_np_idx_conversion(self):
        expected_sent_idxs = np.array([
            [[20, 8, 9, 19, 0, 0, 0], 
             [9, 19, 0, 0, 0, 0, 0], 
             [1, 0, 0, 0, 0, 0, 0], 
             [20, 5, 19, 20, 0, 0, 0]],
            [[20, 8, 9, 19, 0, 0, 0], 
             [9, 19, 0, 0, 0, 0, 0], 
             [1, 14, 15, 20, 8, 5, 18], 
             [15, 14, 5, 0, 0, 0, 0]]])

        expected_label_idxs = np.array(
            [[[0,1,0], [0,0,1],[0,0,1], [0,1,0]],
            [[0,0,1], [0,0,1],[0,1,0],[0,1,0]]])

        test_sent_idxs, test_label_idxs = corpus_utils.np_idx_conversion(
                self.corp, 
                self.maxsentlen, 
                self.maxwordlen, 
                self.char2idx,
                self.idx2label)

        self.assertTrue(np.array_equal(expected_sent_idxs, test_sent_idxs))
        self.assertTrue(np.array_equal(expected_label_idxs, test_label_idxs))

    def test_sent_idx_conversion(self):

        expected_idxs = np.array([
            [[20, 8, 9, 19, 0, 0, 0], 
             [9, 19, 0, 0, 0, 0, 0], 
             [1, 0, 0, 0, 0, 0, 0], 
             [20, 5, 19, 20, 0, 0, 0]],
            [[20, 8, 9, 19, 0, 0, 0], 
             [9, 19, 0, 0, 0, 0, 0], 
             [1, 14, 15, 20, 8, 5, 18], 
             [15, 14, 5, 0, 0, 0, 0]]])

        test_idxs = corpus_utils.sent_idx_conversion(self.sentences, 
                                                     self.maxsentlen, 
                                                     self.maxwordlen, 
                                                     self.char2idx)


        self.assertTrue(np.array_equal(expected_idxs, test_idxs))

    def test_label_idx_conversion(self):
        expected_idxs = np.array(
            [[[0,1,0], [0,0,1],[0,0,1], [0,1,0]],
            [[0,0,1], [0,0,1],[0,1,0],[0,1,0]]])

        test_idxs = corpus_utils.np_label_idx_conversion(self.slabels, 
                                                      self.maxsentlen, 
                                                      self.idx2label)

        self.assertTrue(np.array_equal(expected_idxs, test_idxs))


