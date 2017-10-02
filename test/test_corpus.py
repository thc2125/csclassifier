import unittest

from collections import defaultdict


from corpus import Corpus


class CorpusTestCase(unittest.TestCase):

    def setUp(self):
        self.unit_test_charset = {'a', 'b', 'c'}

        self.unit_test_labelset = {'l1', 'l2', 'l3'} 

        self.unit_test_corpus1_langs =  {'<UNK>'}

        self.unit_test_corpus1_sentences = [
            ["abc", "abc", "abc"],
            ["cba", "cba", "cba"],
            ["aaa", "aaa", "aaa"]]
        self.unit_test_corpus1_labels = [
            ["l1","l1","l1"],
            ["l2","l2","l2"],
            ["l3","l3","l3"]]
        self.unit_test_corpus1_maxwordlen = 3
        self.unit_test_corpus1_maxsentlen = 3 
        self.unit_test_corpus1_sentence_id2sidx = {"1":0, "2":1, "3":2}
        self.unit_test_corpus1_label_frequency = {"l1":3, "l2":3, "l3":3}
        self.unit_test_corpus1_char_frequency = {'a':15, 'b':6, 'c':6}
        self.unit_test_corpus1_filename = "test/data/test_corpus_unittest_corpus1.csv"

        self.unit_test_corpus1 = Corpus()
        self.unit_test_corpus1.read_corpus(self.unit_test_corpus1_filename)

        self.unit_test_corpus2_langs =  {'ar', 'en'}
        self.unit_test_corpus2_sentences = [
            ["d", "de", "def", "defgh"],
            ["abc", "abc", "abc"]]
        self.unit_test_corpus2_labels = [
            ["l3", "l1", "l1", "l1"],
            ["l1", "l1", "l1"]]

        self.unit_test_corpus2_maxwordlen = 5
        self.unit_test_corpus2_maxsentlen = 4
        self.unit_test_corpus2_sentence_id2sidx = {"4":0, "5":1}
        self.unit_test_corpus2_label_frequency = {"l1":6, "l3":1}
        self.unit_test_corpus2_char_frequency = {'a':3, 'b':3, 'c':3, 'd':4, 
            'e':3, 'f':2, 'g':1, 'h':1}
        self.unit_test_corpus2_filename = "test/data/test_corpus_unittest_corpus2.csv"

        self.unit_test_corpus2 = Corpus()

        self.unit_test_corpus2.read_corpus(self.unit_test_corpus2_filename)


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
        self.assertEqual(test_corpus.sentence_id2sidx, {})

        self.assertEqual(test_corpus.char_frequency,defaultdict(int))
        self.assertEqual(test_corpus.label_frequency, defaultdict(int))

        self.assertEqual(filenames, set())

    def test_read_corpus_1(self):
        test_corpus = Corpus()
        test_corpus.read_corpus(self.unit_test_corpus1_filename)
        self.assertEqual(test_corpus.filenames, 
            [self.unit_test_corpus1_filename])

        self.assertEqual(test_corpus.langs, 
            self.unit_test_corpus1_langs)
        self.assertEqual(test_corpus.sentences, 
            self.unit_test_corpus1_sentences)
        self.assertEqual(test_corpus.sentence_id2sidx, 
            self.unit_test_corpus1_sentence_id2sidx)
        self.assertEqual(test_corpus.labels, 
            self.unit_test_corpus1_labels)
        self.assertEqual(test_corpus.char_frequency, 
            self.unit_test_corpus1_char_frequency)
        self.assertEqual(test_corpus.maxsentlen, 
            self.unit_test_corpus1_maxsentlen)
        self.assertEqual(test_corpus.maxwordlen, 
            self.unit_test_corpus1_maxwordlen)

    def test_read_corpus_2(self):
        test_corpus = Corpus()
        test_corpus.read_corpus(self.unit_test_corpus2_filename, langs=('ar', 'en'))
        self.assertEqual(test_corpus.filenames, 
            [self.unit_test_corpus2_filename])

        self.assertEqual(test_corpus.langs, 
            self.unit_test_corpus2_langs)
        self.assertEqual(test_corpus.sentences, 
            self.unit_test_corpus2_sentences)
        self.assertEqual(test_corpus.sentence_id2sidx, 
            self.unit_test_corpus2_sentence_id2sidx)
        self.assertEqual(test_corpus.labels, 
            self.unit_test_corpus2_labels)
        self.assertEqual(test_corpus.char_frequency, 
            self.unit_test_corpus2_char_frequency)
        self.assertEqual(test_corpus.maxsentlen, 
            self.unit_test_corpus2_maxsentlen)
        self.assertEqual(test_corpus.maxwordlen, 
            self.unit_test_corpus2_maxwordlen)
        self.assertEqual(test_corpus.label_frequency, 
            self.unit_test_corpus2_label_frequency)
        self.assertEqual(test_corpus.char_frequency, 
            self.unit_test_corpus2_char_frequency)

    def test_read_corpus_1_and_2(self):
        test_corpus = Corpus()
        test_corpus.read_corpus(self.unit_test_corpus1_filename)
        test_corpus.read_corpus(self.unit_test_corpus2_filename)
        self.assertEqual(test_corpus.filenames, 
            [self.unit_test_corpus1_filename, self.unit_test_corpus2_filename])
        self.assertEqual(test_corpus.langs, 
            union(self.unit_test_corpus1_langs, self.unit_test_corpus2_langs))
        self.assertEqual(test_corpus.sentences, 
            self.unit_test_corpus1_sentences + self.unit_test_corpus2_sentences)
        self.assertEqual(test_corpus.sentence_id2sidx, 
           self.unit_test_corpus1_sentence_id2sidx 
               + self.unit_test_corpus2_sentence_id2sidx)
        self.assertEqual(test_corpus.labels, 
            self.unit_test_corpus1_labels + self.unit_test_corpus2_labels)
        self.assertEqual(test_corpus.char_frequency, 
            self.unit_test_corpus2_char_frequency 
                + self.unit_test_corpus2_char_frequency)
        self.assertEqual(test_corpus.maxsentlen, 
            max(self.unit_test_corpus1_maxsentlen, 
               self.unit_test_corpus2_maxsentlen))
        self.assertEqual(test_corpus.maxwordlen, 
            max(self.unit_test_corpus1_maxwordlen, 
               self.unit_test_corpus2_maxwordlen))
        self.assertEqual(test_corpus.label_frequency, 
            self.unit_test_corpus2_label_frequency 
                + self.unit_test_corpus2_label_frequency)
        self.assertEqual(test_corpus.char_frequency, 
            self.unit_test_corpus2_char_frequency 
                + self.unit_test_corpus2_char_frequency)

    def test_get_sname(self):
        word_id = "tweet_id_870275291293376512_0_0"

