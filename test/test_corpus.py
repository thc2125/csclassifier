import unittest

from collections import Counter
from pathlib  import Path

from corpus import Corpus


class CorpusTestCase(unittest.TestCase):

    def setUp(self):
        self.data_dirpath = Path("test", "data")
        self.corpora_dirpath = self.data_dirpath / "corpora"
        self.unit_test_charset = {'a', 'b', 'c'}

        self.unit_test_labelset = {'l1', 'l2', 'l3'} 

        self.unit_test_corpus1_maxwordlen = 3
        self.unit_test_corpus1_maxsentlen = 3 

        self.unit_test_corpus1_sentences = [
            ["abc", "abc", "abc"],
            ["cba", "cba", "cba"],
            ["aaa", "aaa", "aaa"]]
        self.unit_test_corpus1_labels = [
            ["l1","l1","l1"],
            ["l2","l2","l2"],
            ["l3","l3","l3"]]
        self.unit_test_corpus1_sentence2sidx = {"1":0, "2":1, "3":2}

        self.unit_test_corpus1_label_frequency = Counter({"l1":3, "l2":3, "l3":3})
        self.unit_test_corpus1_char_frequency = Counter({'a':15, 'b':6, 'c':6})
        self.unit_test_corpus1_filepath = (self.corpora_dirpath 
                                           / ("unit_test_corpus1"
                                              + ".tsv"))

        self.unit_test_corpus2_sentences = [
            ["d", "de", "def", "defgh"],
            ["abc", "abc", "abc"]]
        self.unit_test_corpus2_labels = [
            ["l3", "l1", "l1", "l1"],
            ["l1", "l1", "l1"]]

        self.unit_test_corpus2_maxwordlen = 5
        self.unit_test_corpus2_maxsentlen = 4
        self.unit_test_corpus2_sentence2sidx = {"4":0, "5":1}
        self.unit_test_corpus2_label_frequency = Counter({"l1":6, "l3":1})
        self.unit_test_corpus2_char_frequency = Counter({
                'a':3, 'b':3, 'c':3, 'd':4, 
                'e':3, 'f':2, 'g':1, 'h':1,
                })
        self.unit_test_corpus2_filepath = (self.corpora_dirpath 
                                           / ("unit_test_corpus2"
                                              + ".tsv"))


    def tearDown(self):
        pass

    '''
    __init__() does the following:
        1. initializes sentences, labels, label_frequency, maxwordlen, maxsentlen,
             sentence2sidx, char_frequency)
    '''

    def test___init___(self):
        test_corpus = Corpus()

        self.assertEqual(test_corpus.maxwordlen, 0)
        self.assertEqual(test_corpus.maxsentlen, 0)

        self.assertEqual(test_corpus.sentences, [])
        self.assertEqual(test_corpus.labels, [])
        self.assertEqual(test_corpus.sentence2sidx, {})

        self.assertEqual(test_corpus.char_frequency, Counter())
        self.assertEqual(test_corpus.label_frequency, Counter())

        self.assertEqual(test_corpus.filenames, set())

    def test_read_corpus_1(self):
        test_corpus = Corpus()
        test_corpus.read_corpus(self.unit_test_corpus1_filepath)

        expected_corpus_filenames = set([self.unit_test_corpus1_filepath.name])
        test_corpus_filenames = test_corpus.filenames
        self.assertEqual(expected_corpus_filenames, test_corpus_filenames)

        expected_corpus_sentences = self.unit_test_corpus1_sentences
        test_corpus_sentences = test_corpus.sentences 
        self.assertEqual(expected_corpus_sentences, test_corpus_sentences)


        expected_corpus_sentence2sidx = self.unit_test_corpus1_sentence2sidx
        test_corpus_sentence2sidx = test_corpus.sentence2sidx
        self.assertEqual(expected_corpus_sentence2sidx, 
                         test_corpus_sentence2sidx)

        expected_corpus_labels = self.unit_test_corpus1_labels
        test_corpus_labels = test_corpus.labels
        self.assertEqual(expected_corpus_labels, 
                         test_corpus_labels)        

        expected_corpus_char_frequency = self.unit_test_corpus1_char_frequency
        test_corpus_char_frequency = test_corpus.char_frequency
        self.assertEqual(expected_corpus_char_frequency,
                         test_corpus_char_frequency)

        expected_corpus_label_frequency = self.unit_test_corpus1_label_frequency
        test_corpus_label_frequency = test_corpus.label_frequency
        self.assertEqual(expected_corpus_label_frequency,
                         test_corpus_label_frequency)


        expected_corpus_maxsentlen = self.unit_test_corpus1_maxsentlen
        test_corpus_maxsentlen = test_corpus.maxsentlen
        self.assertEqual(expected_corpus_maxsentlen,
                         test_corpus_maxsentlen)

        expected_corpus_maxwordlen = self.unit_test_corpus1_maxwordlen
        test_corpus_maxwordlen = test_corpus.maxwordlen
        self.assertEqual(expected_corpus_maxwordlen,
                         test_corpus_maxwordlen)

    def test_read_corpus_2(self):
        test_corpus = Corpus()
        test_corpus.read_corpus(self.unit_test_corpus2_filepath)

        expected_corpus_filenames = set([self.unit_test_corpus2_filepath.name])
        test_corpus_filenames = test_corpus.filenames
        self.assertEqual(expected_corpus_filenames, test_corpus_filenames)

        expected_corpus_sentences = self.unit_test_corpus2_sentences
        test_corpus_sentences = test_corpus.sentences 
        self.assertEqual(expected_corpus_sentences, test_corpus_sentences)


        expected_corpus_sentence2sidx = self.unit_test_corpus2_sentence2sidx
        test_corpus_sentence2sidx = test_corpus.sentence2sidx
        self.assertEqual(expected_corpus_sentence2sidx, 
                         test_corpus_sentence2sidx)

        expected_corpus_labels = self.unit_test_corpus2_labels
        test_corpus_labels = test_corpus.labels
        self.assertEqual(expected_corpus_labels, 
                         test_corpus_labels)        

        expected_corpus_char_frequency = self.unit_test_corpus2_char_frequency
        test_corpus_char_frequency = test_corpus.char_frequency
        self.assertEqual(expected_corpus_char_frequency,
                         test_corpus_char_frequency)

        expected_corpus_label_frequency = self.unit_test_corpus2_label_frequency
        test_corpus_label_frequency = test_corpus.label_frequency
        self.assertEqual(expected_corpus_label_frequency,
                         test_corpus_label_frequency)

        expected_corpus_maxsentlen = self.unit_test_corpus2_maxsentlen
        test_corpus_maxsentlen = test_corpus.maxsentlen
        self.assertEqual(expected_corpus_maxsentlen,
                         test_corpus_maxsentlen)

        expected_corpus_maxwordlen = self.unit_test_corpus2_maxwordlen
        test_corpus_maxwordlen = test_corpus.maxwordlen
        self.assertEqual(expected_corpus_maxwordlen,
                         test_corpus_maxwordlen)

    def test_read_corpus_1_and_2(self):
        test_corpus = Corpus()
        test_corpus.read_corpus(self.unit_test_corpus1_filepath)
        test_corpus.read_corpus(self.unit_test_corpus2_filepath)

        expected_corpus_filenames = set([self.unit_test_corpus1_filepath.name,
                                         self.unit_test_corpus2_filepath.name])
        test_corpus_filenames = test_corpus.filenames
        self.assertEqual(expected_corpus_filenames, test_corpus_filenames)

        expected_corpus_sentences = (self.unit_test_corpus1_sentences 
                                    + self.unit_test_corpus2_sentences)
        test_corpus_sentences = test_corpus.sentences 
        self.assertEqual(expected_corpus_sentences, test_corpus_sentences)


        tmp_corpus2_sentence2sidx = {
            k: v + len(self.unit_test_corpus1_sentence2sidx)
            for k, v in self.unit_test_corpus2_sentence2sidx.items()
            }
        expected_corpus_sentence2sidx = (
            self.unit_test_corpus1_sentence2sidx.copy()
            )
        expected_corpus_sentence2sidx.update(tmp_corpus2_sentence2sidx)
        test_corpus_sentence2sidx = test_corpus.sentence2sidx
        self.assertEqual(expected_corpus_sentence2sidx, 
                         test_corpus_sentence2sidx)

        expected_corpus_labels = (self.unit_test_corpus1_labels
                                  + self.unit_test_corpus2_labels)
        test_corpus_labels = test_corpus.labels
        self.assertEqual(expected_corpus_labels, 
                         test_corpus_labels)        

        expected_corpus_char_frequency =  (
            self.unit_test_corpus1_char_frequency
            + self.unit_test_corpus2_char_frequency
            )
        test_corpus_char_frequency = test_corpus.char_frequency
        self.assertEqual(expected_corpus_char_frequency,
                         test_corpus_char_frequency)

        expected_corpus_label_frequency =  (
            self.unit_test_corpus1_label_frequency
            + self.unit_test_corpus2_label_frequency
            )
        test_corpus_label_frequency = test_corpus.label_frequency
        self.assertEqual(expected_corpus_label_frequency,
                         test_corpus_label_frequency)

        expected_corpus_maxsentlen = max(self.unit_test_corpus1_maxsentlen,
                                         self.unit_test_corpus2_maxsentlen)
        test_corpus_maxsentlen = test_corpus.maxsentlen
        self.assertEqual(expected_corpus_maxsentlen,
                         test_corpus_maxsentlen)

        expected_corpus_maxwordlen = max(self.unit_test_corpus2_maxwordlen,
                                         self.unit_test_corpus2_maxwordlen)
        test_corpus_maxwordlen = test_corpus.maxwordlen
        self.assertEqual(expected_corpus_maxwordlen,
                         test_corpus_maxwordlen)

    def test_read_row(self):
        test_corpus = Corpus()
        row1 = ["tweet_id_867268522262581248_0_8", "vajase", "other"]
        row2 = ["tweet_id_867276476613251076_0_2", "dalaw", "lang1"]

        corpus_filename1 = self.unit_test_corpus1_filepath.name
        corpus_filename2 = self.unit_test_corpus2_filepath.name

        test_corpus.read_row(row1, corpus_filename1)
        test_corpus.read_row(row2, corpus_filename2)
        
        expected_maxwordlen = 6
        expected_maxsentlen = 1

        expected_sentences = [["vajase"], ["dalaw"]]
        test_sentences = test_corpus.sentences 
        self.assertEqual(expected_sentences, test_sentences)

        expected_labels = [["other"], ["lang1"]]
        test_labels = test_corpus.labels
        self.assertEqual(expected_labels, test_labels)

        expected_sentence2sidx = {"tweet_id_867268522262581248_0":0,
                                  "tweet_id_867276476613251076_0":1}
        test_sentence2sidx = test_corpus.sentence2sidx
        self.assertEqual(expected_sentence2sidx, test_sentence2sidx)

        expected_char_frequency = Counter({
                'v':1, 'a':4, 'j':1, 's':1, 'e':1, 'd':1, 'l':1, 'w':1
                })
        test_char_frequency = test_corpus.char_frequency
        self.assertEqual(expected_char_frequency, test_char_frequency)

        expected_label_frequency = Counter({'other':1, 'lang1':1})
        test_label_frequency = test_corpus.label_frequency
        self.assertEqual(expected_label_frequency, test_label_frequency)

        expected_filenames = set([corpus_filename1, corpus_filename2])
        test_filenames = test_corpus.filenames 
        self.assertEqual(expected_filenames, test_filenames)

    def test_get_sentence_id(self):
        word_id = "tweet_id_870275291293376512_0_0"
        expected_sentence_id = "tweet_id_870275291293376512_0"
        test_sentence_id = Corpus.get_sentence_id(word_id)
        self.assertEqual(expected_sentence_id, test_sentence_id)
        

