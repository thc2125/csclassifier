#!/usr/bin/python3

import string
import unittest

from collections import Counter

import numpy as np

import corpus_utils

class CorpusUtilsTestCase(unittest.TestCase):

    def setUp(self):
        self.idx2char = ['<pad>'] + list(string.ascii_letters) + ['<unk>']
        self.char2idx = {}

        self.idx2label = ['<pad>', 'label1', 'label2']
        self.label2idx = {'<pad>' : 0, 'label1' : 1, 'label2' : 2}
        self.idx2label = {i:l for l, i in self.label2idx.items()}

        self.sentences = [["this", "is", "a", "test"], 
                          ["this", "is", "another", "one"]]
        self.char_frequency = Counter({'t': 5, 'h' : 3, 'i' : 4, 's': 5, 'a' : 1, 'e' : 2, 'n' : 2, 'o' : 2, 'r' : 1})

        self.slabels

    def tearDown(self):
        pass

    def test_np_idx_conversion(self):
        pass

    def test_sent_idx_conversion(self):
        test_idxs = corpus_utils.sent_idx_conversion(self.sentences, 
                                                     4, 
                                                     7, 
                                                     self.char2idx)

        expected_idxs = np.array([
            [[116, 104, 105, 115, 0, 0, 0], 
             [105, 115, 0, 0, 0, 0, 0], 
             [97, 0, 0, 0, 0, 0, 0], 
             [116, 101, 115, 116, 0, 0, 0]],
            [[116, 104, 105, 115, 0, 0, 0], 
             [105, 115, 0, 0, 0, 0, 0], 
             [97, 110, 111, 116, 104, 101, 114], 
             [111, 110, 101, 0, 0, 0, 0]]])

        self.assertTrue(np.array_equal(expected_idxs, test_idxs))

    def test_label_idx_conversion(self):
