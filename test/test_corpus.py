import unittest

from collections import defaultdict


from corpus import Corpus


class CorpusTestCase(unittest.TestCase):

    def setUp(self):
        self.unit_test_char2idx = {'a':1, 'b':2, 'c':3}
        self.unit_test_idx2char = {1:'a', 2:'b', 3:'c'}
        self.unit_test_char_dictionary = (
            self.unit_test_char2idx, unit_test_idx2char)

        self.unit_test_label2idx = {'l1':0, 'l2':1, 'l3':2, 'l4':3} 
        self.unit_test_idx2label = {0:'l1', 1:'l2', 2:'l3', 3:'l4'}
        self.unit_test_label_dictionary = (unit_test_label2idx, unit_test_idx2label)


        self.unit_test_sentences1 = [
            ["abc abc abc"],
            ["cba cba cba"],
            ["aaa aaa aaa"]]
        self.unit_test_labels1 = [
            ["l1","l1","l1"],
            ["l2","l2","l2"],
            ["l3","l3","l3"]]
        self.unit_test_maxwordlen1 = 3
        self.unit_test_maxsentlen1 = 3 
        self.unit_test_corpus1_char_frequency = {'a':15, 'b':6, 'c':6}
        self.unit_test_corpus1_filename = "test/data/test_corpus_unittest_corpus1.csv"

        self.unit_test_corpus1 = Corpus(
            char_dictionary=self.unit_test_char_dictionary,
            label_dictionary=self.unit_test_label_dictionary)

        self.unit_test_corpus1.read_corpus(self.unit_test_corpus1_filename)

        self.unit_test_sentences2 = [
            ["d de def defgh"],
            ["abc abc abc"]]
        self.unit_test_maxwordlen2 = 5
        self.unit_test_maxsentlen2 = 4

        self.unit_test_corpus2_char_frequency = {'a':3, 'b':3, 'c':3, 'unk':11}
        self.unit_test_corpus2_filename = "test/data/test_corpus_unittest_corpus1.csv"

        self.unit_test_corpus2 = Corpus(
            char_dictionary=self.unit_test_char_dictionary,
            label_dictionary=self.unit_test_label_dictionary)

        self.unit_test_corpus2.read_corpus(self.unit_test_corpus2_filename)


        '''
        self.corpus1 = Corpus(label_dictionary=(self.unit_test_label2idx, self.unit_test_idx2label))
        self.corpus2 = Corpus(label_dictionary=(self.unit_test_label2idx, self.unit_test_idx2label))
        self.corpus1_filepath = 'test/data/Corpus_corpus_de+ar.csv'
        self.corpus2_filepath = 'test/data/Corpus_corpus_fr+ar.csv'
        self.corpus1_num_words = 56
                
        self.corpus1.read_corpus(self.corpus1_filepath, dl=dl)
        self.corpus2.read_corpus(self.corpus2_filepath, dl=dl)

        self.sentences1 = [
            ["ich", "habe", "ein", "video", 
                "zu", "einer", "-playlist", "hinzugefügt", "nancy", "ajram", 
                "-", "mestaniak", "(live)", "نانسي", "عجرم"],
            ["sehr", "informatives", "video", 
                "zu", "#", "von", "⬇also", "looki", "looki", "⬇"],
            ["sehr", "informatives", "video", 
                "zu", "#", "auf", "dem", "#", "von", "⬇also", "erneut", 
                "looki", "looki", "⬇"],
            ["haus", "in", "marienhölzungsweg",
                "#", "❒",],
            ["dänemark", "und", "in", 
                "deutschland", "1-0", "freundschaftsspiel", "2017.06.06", 
                "عبر"],
            ["dänemark","عبر","freundschaftsspiel","نانسي"]]


        self.labels1 = [
            ["lang1", "lang1", "lang1", "lang1",
                "lang1", "lang1", "other", "lang1", "lang1", "other", 
                "other", "other", "other", "lang2", "other"],
            ["lang1", "other", "lang1",
                "lang1", "other", "lang1", "other", "other", "other", "other"],
            ["lang1", "other", "lang1",
                "lang1", "other", "lang1", "lang1", "other", "lang1", "other", "lang1",
                "other", "other", "other"],
            ["lang1", "lang1", "other", 
                "other", "other"],
            ["lang1", "lang1", "lang1", 
                "lang1", "other", "lang1", "other",
                "lang2"],
            ["lang1", "lang2", "lang1", "lang2"]]
        self.label1_counts = defaultdict(int, lang1=28, lang2=4, other=24)

        self.maxsentlen1 = 15

        self.sentences2 = [
            ["j'aime", "une", "vidéo"],
            ["assawra", "–", "الثورة", "après", "avoir", "soutenu", 
                "l’isolement", "du", 'qatar,', "trump", "appelle", "le", 
                "golfe", "à", "l’unit", "é"], 
            ["les", "versets", "de", "patience", "pour", "toute", "personne", 
                'éprouvée,', "en", "difficulté", "آيات", "الصبر"], 
            ["salam", "akhi.", "qu’allah", "te", "facilite"]]

        self.labels2 = [
            ["other", "lang1", "lang1", "other", "lang1", "lang2", "lang1",
                "lang1", "lang1", "other", "lang1", "other", "other", "lang1",
                "lang1", "lang1", "lang1", "other", "lang1"],
            ["lang1", "lang1", "lang1", "lang1", "lang1", "lang1", 
                "lang1", "other", "lang1", "lang1", "lang2", "lang2"], 
            ["other", "other", "other", "lang1", "lang1"]] 
        self.label2_counts = defaultdict(int, other=10, lang1=23, lang2=3)
        '''

    def tearDown(self):
        pass

    '''
    __init__() does the following:
        1. initializes char2idx and idx2char
        2. initializes label2idx and idx2label
        3. sets a flag for whether it's a training corpus or not
        4. sets a flag for whether alphabetical unknowns are used
        5. initializes the other data (sentences, labels, label_counts, maxwordlen, maxsentlen, sentence2sidx, sidx, char_frequency)
    '''

    def test___init___no_params(self):
        test_corpus = Corpus()

        self.assertEqual(test_corpus.char2idx, None)
        self.assertEqual(test_corpus.idx2char, None)
        self.assertEqual(test_corpus.label2idx, None)
        self.assertEqual(test_corpus.idx2label, None)
        self.assertEqual(test_corpus.train, False)
        self.assertEqual(test_corpus.use_alphabets, False)
        self._test__init_data(test_corpus)

    def _test__init_data(self, test_corpus):
        # Init_Data test
        self.assertEqual(test_corpus.sentences, [])
        self.assertEqual(test_corpus.labels, [])
        self.assertEqual(test_corpus.label_counts, defaultdict(int))
        self.assertEqual(test_corpus.maxwordlen, 0)
        self.assertEqual(test_corpus.maxsentlen, 0)
        self.assertEqual(test_corpus.sentence2sidx, {})
        self.assertEqual(test_corpus.sidx, 0)
        self.assertEqual(test_corpus.char_frequency,defaultdict(int))

    def test___init___char_dictionary(self):
        test_corpus = Corpus(char_dictionary=(self.unit_test_char2idx, self.unit_test_idx2char))

        self.assertEqual(test_corpus.char2idx, self.unit_test_char2idx)
        self.assertEqual(test_corpus.idx2char, self.unit_test_idx2char)       

        self.assertEqual(test_corpus.label2idx, None)
        self.assertEqual(test_corpus.idx2label, None)
        self.assertEqual(test_corpus.train, False)
        self.assertEqual(test_corpus.use_alphabets, False)
        self._test__init_data(test_corpus)


    def test___init___label_dictionary(self):
        test_corpus = Corpus(label_dictionary=(self.unit_test_label2idx, self.unit_test_idx2label))
        self.assertEqual(test_corpus.char2idx, None)
        self.assertEqual(test_corpus.idx2char, None)

        self.assertEqual(test_corpus.label2idx, self.unit_test_label2idx)
        self.assertEqual(test_corpus.idx2label, self.unit_test_idx2label)       

        self.assertEqual(test_corpus.train, False)
        self.assertEqual(test_corpus.use_alphabets, False)
        self._test__init_data(test_corpus)

    def test___init___train(self):
        test_corpus = Corpus(train=True)

        self.assertEqual(test_corpus.char2idx, None)
        self.assertEqual(test_corpus.idx2char, None)
        self.assertEqual(test_corpus.label2idx, None)
        self.assertEqual(test_corpus.idx2label, None)

        self.assertEqual(test_corpus.train, True)

        self.assertEqual(test_corpus.use_alphabets, False)
        self._test__init_data(test_corpus)

    def test___init___use_alphabets(self):
        test_corpus = Corpus(use_alphabets=True)

        self.assertEqual(test_corpus.char2idx, None)
        self.assertEqual(test_corpus.idx2char, None)
        self.assertEqual(test_corpus.label2idx, None)
        self.assertEqual(test_corpus.idx2label, None)
        self.assertEqual(test_corpus.train, False)

        self.assertEqual(test_corpus.use_alphabets, True)
        self._test__init_data(test_corpus)

    def test___init___all_params(self):
        test_corpus = Corpus(
            label_dictionary=(self.unit_test_label2idx, self.unit_test_idx2label),
            char_dictionary=(self.unit_test_char2idx, self.unit_test_idx2char),
            train=True,
            use_alphabets=True)

        self.assertEqual(test_corpus.char2idx, self.unit_test_char2idx)
        self.assertEqual(test_corpus.idx2char, self.unit_test_idx2char)       

        self.assertEqual(test_corpus.label2idx, self.unit_test_label2idx)
        self.assertEqual(test_corpus.idx2label, self.unit_test_idx2label)       

        self.assertEqual(test_corpus.train, True)

        self.assertEqual(test_corpus.use_alphabets, True)

        self._test__init_data(test_corpus)

    def test_read_corpus_no_unks(self):
        test_corpus = Corpus()
        test_corpus.read_corpus(unit_test_corpus1_filename)
        self.assertEqual(sorted(test_corpus.sentences), 
            sorted(self.unit_test_sentences1))
        self.assertEqual(sorted(test_corpus.labels), 
            sorted(self.unit_test_labels1))
        self.assertEqual(self.test_corpus.char_frequency, 
            self.unit_test_corpus1_char_frequency)
        self.assertEqual(self.test_corpus.maxsentlen, 
            self.unit_test_maxsentlen1)
        self.assertEqual(self.test_corpus.maxwordlen, 
            self.unit_test_maxwordlen1)

    def test_read_corpus_unks(self):
        test_corpus = Corpus()
        test_corpus.read_corpus(unit_test_corpus2_filename)
        self.assertEqual(sorted(test_corpus.sentences), 
            sorted(self.unit_test_sentences2))
        self.assertEqual(sorted(test_corpus.labels), 
            sorted(self.unit_test_labels2))
        self.assertEqual(test_corpus.char_frequency, 
            self.unit_test_corpus2_char_frequency)
        self.assertEqual(test_corpus.maxsentlen, 
            self.unit_test_maxsentlen2)
        self.assertEqual(test_corpus.maxwordlen, 
            self.unit_test_maxwordlen2)
        self.assertEqual(test_corpus.label_counts, self.label1_counts)


    def test_read_corpus_labels1_count(self):

    def test_read_corpus_labels2_count(self):
        self.assertEqual(self.corpus2.label_counts, self.label2_counts)

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


    '''
    So to test add:
        1. We need to make sure the original corpora are unaffected and only a new
            corpus is created.
        2. We need to ensure combine works
        (3. We need to make sure a corpus cannot be added to itself? Or allow only new sentences to be added?)
    '''

    def test___add___different(self):
        test_corpus = self.unit_test_corpus1 + self.unit_test_corpus2

    def test___add___same(self):
        test_corpus = self.unit_test_corpus1 + self.unit_test_corpus1


    def test__combine_(self):


    '''
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
        new_sentences = self.sentences1 + self.sentences2
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

    def test___add___label_counts(self):
        new_corpus = self.corpus1 + self.corpus2
        new_label_counts = defaultdict(int, lang1=51, lang2=7, other=34)

        self.assertEqual(new_corpus.label_counts, new_label_counts)



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


