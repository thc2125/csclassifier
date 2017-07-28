#!/usr/bin/python3

import unittest
import csv

from utils import Corpus 

word_col = 1
dl = ','

class CorpusTestCase(unittest.TestCase):

    def setUp(self):
        self.label2idx = {'label1':1, 'label2':2, 'label3':3}
        self.idx2label = {i:l for l, i in self.label2idx.items()}
        self.corpus1 = Corpus(label_dictionary=(self.label2idx, self.idx2label))
        self.corpus2 = Corpus(label_dictionary=(self.label2idx, self.idx2label))
        self.corpus1_filepath = 'tests/data/corpus1_file.csv'
        self.corpus2_filepath = 'tests/data/corpus2_file.csv'

        self.corpus1_num_words = self.get_num_words(self.corpus1_filepath, 0)
                
        self.corpus1.read_corpus(self.corpus1_filepath, dl=dl)
        self.corpus2.read_corpus(self.corpus2_filepath, dl=dl)

    @unittest.skip("skipping initialization test")
    def test___init__(self):
        pass

    @unittest.skip("skipping read_row test")
    def test_read_row(self):
        pass
        #self.corpus.read_row

    def test_read_corpus_num_sentences(self):
        print(self.corpus.sentences)
        self.assertEqual(self.corpus1_num_words, len(self.corpus.sentences))

        pass

    def test_read_corpus_num_sentences(self):
        self.assertEqual(self.corpus1_num_words, len(self.corpus1.sentences))
        pass

    def test_read_corpus_num_labels(self):
        self.assertEqual(self.corpus1_num_words, len(self.corpus1.labels))
        pass

    def test___add___sentences_len(self):
        c1_sent_len = len(self.corpus1.sentences)
        c2_sent_len = len(self.corpus1.sentences)
        new_corpus = self.corpus1 + self.corpus2
        self.assertEqual(len(new_corpus.sentences), c1_sent_len + c2_sent_len)

    def test___add___labels_len(self):
        c1_label_len = len(self.corpus1.labels)
        c2_label_len = len(self.corpus1.labels)
        new_corpus = self.corpus1 + self.corpus2
        self.assertEqual(len(new_corpus.labels), c1_label_len + c2_label_len)

    def test___add___sentence_elements(self):
        new_corpus = self.corpus1 + self.corpus2
        loc = 0
        for i in range(len(self.corpus1.sentences)):
            for j in range(len(new_corpus.sentences[i])):
                self.assertEqual(new_corpus.sentences[i][j], 
                    self.corpus1.sentences[i][j] if loc < len(self.corpus1.sentences)
                    else self.corpus2.sentences[i-len(self.corpus1.sentences)][j])
                loc += 1


    def tearDown(self):
        pass

    def get_num_words(self, corpus_filepath, num_words):
        with open(corpus_filepath) as corpus_file:
            corpus_reader = csv.reader(corpus_file, delimiter=dl)
            # Skip the header
            next(corpus_reader)
            for row in corpus_reader:
                if len(row[word_col]) <= 34:
                    num_words += 1
        return num_words
