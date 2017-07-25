#!/usr/bin/python3

import unittest

class CorpusTestCase(unittest.TestCase):

    def setUp(self):
        self.label2idx = {'label1':1, 'label2':2, 'label3':3}
        self.idx2label = {i:l for l, i in self.label2idx.items()}
        self.corpus = Corpus(label_dictionary=(self.label2idx, self.idx2label))
        self.corpus1_filepath = 'data/corpus_file_1.csv'
        self.corpus2_filepath = 'data/corpus_file_2.csv'



    @unittest.skip("skipping initialization test")
    def test___init__(self):
        pass

    @unittest.skip("skipping read_row test")
    def test_read_row(self):
        pass
        #self.corpus.read_row

    def test_read_corpus(self):
        self.corpus.read_corpus(self.corpus1_filepath)
        assert

    def tearDown(self):
        pass

