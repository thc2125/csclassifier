#!/usr/bin/python3

import unittest
import csv
import random

from utils import Corpus 
from collections import defaultdict
from collections import Counter


word_col = 1
dl = ','




class CorpusTestCase(unittest.TestCase):

    def setUp(self):
        self.test_char2idx = {'a':1, 'b':2, 'c':3}
        self.test_idx2char = {v:k for k, v in self.test_char2idx.items()}

        self.test_label2idx = {'label1':1, 'label2':2, 'label3':3}
        self.test_idx2label = {i:l for l, i in self.test_label2idx.items()}
        self.corpus1 = Corpus(label_dictionary=(self.test_label2idx, self.test_idx2label))
        self.corpus2 = Corpus(label_dictionary=(self.test_label2idx, self.test_idx2label))
        self.corpus1_filepath = 'test/data/corpus_de+ar.csv'
        self.corpus2_filepath = 'test/data/corpus_fr+ar.csv'
        self.corpus1_num_words = self.get_num_words(self.corpus1_filepath, 0)
                
        self.corpus1.read_corpus(self.corpus1_filepath, dl=dl)
        self.corpus2.read_corpus(self.corpus2_filepath, dl=dl)

        self.sentences1, self.labels1 = self.get_sentences_labels(self.corpus1_filepath)

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

    def get_sentences_labels(self, corpus_filepath):
        sentences = defaultdict(list)
        labels = defaultdict(list)
        with open(corpus_filepath) as corpus_file:
            corpus_reader = csv.reader(corpus_file, delimiter=dl)
            # Skip the header
            next(corpus_reader)
            for row in corpus_reader:
                if len(row[1]) > 34:
                    continue
                sname = ''.join(row[0].split(sep='_')[:-1])
                sentences[sname].append(row[1])
                labels[sname].append(row[2])
        return sentences.values(), labels.values()

    def test___init___char_dictionary(self):
        test_corpus = Corpus(char_dictionary=(self.test_char2idx, self.test_idx2char))
        self.assertEqual(test_corpus.char2idx, self.test_char2idx)
        self.assertEqual(test_corpus.idx2char, self.test_idx2char)       

    def test___init___label_dictionary(self):
        test_corpus = Corpus(label_dictionary=(self.test_label2idx, self.test_idx2label))
        self.assertEqual(test_corpus.label2idx, self.test_label2idx)
        self.assertEqual(test_corpus.idx2label, self.test_idx2label)       

    def test___init___train(self):
        test_corpus1 = Corpus(train = False)
        test_corpus2 = Corpus(train = True)
        self.assertEqual(test_corpus1.train, False)
        self.assertEqual(test_corpus2.train, True)

    def test___init_____init_data__(self):
        test_corpus = Corpus()
        self.assertEqual(test_corpus.labels, [])
        self.assertEqual(test_corpus.maxwordlen, 0)
        self.assertEqual(test_corpus.maxsentlen, 0)
        self.assertEqual(test_corpus.sentence2sidx, {})
        self.assertEqual(test_corpus.sidx, 0)
        self.assertEqual(test_corpus.char_frequency, {})

    def test___add___sentences_len(self):
        c1_sent_len = len(self.corpus1.sentences)
        c2_sent_len = len(self.corpus2.sentences)
        new_corpus = self.corpus1 + self.corpus2
        self.assertEqual(len(new_corpus.sentences), c1_sent_len + c2_sent_len)

    def test___add___labels_len(self):
        c1_label_len = len(self.corpus1.labels)
        c2_label_len = len(self.corpus2.labels)
        new_corpus = self.corpus1 + self.corpus2
        self.assertEqual(len(new_corpus.labels), c1_label_len + c2_label_len)

    def test___add___sentence_elements(self):
        new_corpus = self.corpus1 + self.corpus2
        new_sentences = self.corpus1.sentences + self.corpus2.sentences
        self.assertEqual(new_corpus.sentences, new_sentences)

    def test___add___bookkeeping(self):
        new_corpus = self.corpus1 + self.corpus2

        self.assertEqual(new_corpus.sidx, len(new_corpus.sentences))
        self.assertEqual(new_corpus.maxsentlen, 
            max(self.corpus1.maxsentlen, self.corpus2.maxsentlen))
        self.assertEqual(new_corpus.maxsentlen, 
            max(self.corpus1.maxsentlen, self.corpus2.maxsentlen))

    def test___add___char_frequency(self):
        new_corpus = self.corpus1 + self.corpus2
        new_char_frequency = self.corpus1.char_frequency + self.corpus2.char_frequency

        self.assertEqual(new_corpus.char_frequency, new_char_frequency)

    def test_read_corpus_sentences(self):
        self.assertEqual(sorted(self.corpus1.sentences), sorted(self.sentences1))

    def test_read_corpus_labels(self):
        self.assertEqual(sorted(self.corpus1.labels), sorted(self.labels1))

    def test_read_corpus_char_frequency(self):
        self.assertEqual(self.corpus1.char_frequency, 
            Counter(c for sent in self.sentences1 for w in sent for c in w))

    def test_read_corpus_maxsentlen(self):
        self.assertEqual(self.corpus1.maxsentlen, max(len(sent) for sent in self.sentences1))


    def test_read_row_label(self):
        new_corpus = Corpus()

        with open(self.corpus1_filepath) as corpus_file:
            corpus_reader = csv.reader(corpus_file, delimiter=dl)
            next(corpus_reader)
            for row in corpus_reader:
               # 10% probability of checking a given row. 
               # Test corpus must have at least 10 sentences
               if random.random() < .1:
                   new_corpus.read_row(row)
                   label = new_corpus.label_word(row[2])
                   # Assumption is made that this is now the only 
                   # sentence/word in the corpus
                   self.assertEqual(new_corpus.labels[0][0], label)
                   break

    def test_read_row_word(self):
        new_corpus = Corpus()

        with open(self.corpus1_filepath) as corpus_file:
            corpus_reader = csv.reader(corpus_file, delimiter=dl)
            next(corpus_reader)
            for row in corpus_reader:
               # 10% probability of checking a given row. 
               # Test corpus must have at least 10 sentences
               if random.random() < .1:
                   new_corpus.read_row(row)
                   word = (row[1])
                   # Assumption is made that this is now the only 
                   # sentence/word in the corpus
                   self.assertEqual(new_corpus.sentences[0][0], word)
                   break

    def test_read_row_sentence(self):
        new_corpus = Corpus()

        with open(self.corpus1_filepath) as corpus_file:
            corpus_reader = csv.reader(corpus_file, delimiter=dl)
            next(corpus_reader)
            for row in corpus_reader:
               # 10% probability of checking a given row. 
               # Test corpus must have at least 10 sentences
               if random.random() < .1:
                   new_corpus.read_row(row)
                   label = new_corpus.label_word(row[2])
                   self.assertEqual(new_corpus.labels[0][0], label)
                   break

    def test_read_corpus_num_words(self):
        self.assertEqual(self.corpus1_num_words, 
            sum([len(sentence) for sentence in self.corpus1.sentences]))


    def test_read_corpus_num_sentences(self):
        self.assertEqual(len(self.sentences1), len(self.corpus1.sentences))

    def test_read_corpus_num_labels(self):
        self.assertEqual(self.corpus1_num_words, 
            sum([len(labels) for labels in self.corpus1.labels]))

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



