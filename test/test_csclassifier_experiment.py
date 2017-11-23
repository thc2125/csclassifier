#!/usr/bin/python3

import unittest
import csv
import random

from collections import defaultdict
from collections import Counter

import csclassifier_experiment

#python csclassifier_experiment.py -o test/output/ -x test/data/corpus_fr+ar.csv -p corpus_ -e 5 -t 2 test/data/
class CSClassifierExperimentTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    # Currently only makes sure that it executes
    def test_main_excluded_corpus(self):
        self.assertTrue(csclassifier_experiment.main('test/data', 
                output_dirname='test/output',
                excluded_corpus_filename='test/data/corpus_fr+ar.csv', 
                corpus_filename_prefix='corpus_', epochs=5, patience=2))

    def test_main_all_corpora(self):
        self.assertTrue(csclassifier_experiment.main('test/data', 
                output_dirname='test/output',
                corpus_filename_prefix='corpus_', epochs=5, patience=2))


