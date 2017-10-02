# Design of utils file and Corpus class inspired by 
# Columbia University
# COMS 4705W 2017 
# Assignment 4
# Professor Karl Stratos

import csv

from collections import defaultdict

class Corpus():
    # TODO: Add a label set to indicate malformed data
    WORD_LEN_LIMIT = 34

    def __init__(self):

        self.maxwordlen = 0
        self.maxsentlen = 0

        self.sentences=[]
        self.labels=[]
        self.sentence2sidx = {}

        self.char_frequency = defaultdict(int)
        self.label_frequency = defaultdict(int)

        self.filenames = set()

    def read_corpus(self, corpus_filepath, dl='\t'):
        """Reads in a corpus file and sets the corpus variables.
    
        Keyword arguments:
        corpus_filepath -- The filepath to a normalized corpus
        langs -- A tuple of 2-character language codes to be found in the corpus
        """

        with corpus_filepath.open() as corpus_file:
            corpus_reader = csv.reader(corpus_file, delimiter=dl)

            # Skip the header
            next(corpus_reader)
            for row in corpus_reader:
                self.read_row(row, corpus_filepath.name)

    def read_row(self, row, corpus_filename):
        """Reads a csv row and updates the Corpus variables.
    
        Keyword arguments:
        file
        row -- a list of csv row values ([sentence_id, word, lang_label,...])
        """

        word = row[1]

        # NOTE: This puts a max word length on a word
        # Length arbitrary based on
        # len("supercalifragilisticexpialidocious")
        if len(word) > WORD_LEN_LIMIT:
           return

        # Add to the set of filenames and language pairs
        self.filenames.add(corpus_filename)
           
        # Remove the word id at the end of the sentence name
        sentence_id = get_sentence_id(row[0])

        if sentence_id not in self.sentence2sidx:
           self._add_sentence(sentence_id)

        label = self.label_word(row[2])

        sidx = self.sentence_id2sidx[sentence_id]
        self.sentences[sidx].append(word)
        self.labels[sidx].append(label)

        self.maxwordlen = max(self.maxwordlen, len(word))
        self.maxsentlen = max(self.maxsentlen, 
            len(self.sentences[self.sentence2sidx[sname]]))

        # Get the character frequency for a word.
        for c in word:
           self.char_frequency[c] += 1


    def _add_sentence(self, sname):
           self.sentence2sidx[sname] = len(self.sentences)
           self.sentences.append([])
           self.labels.append([])

    def label_word(self, label):
        self.label_frequency[label] += 1
        return label

    @staticmethod
    def get_sentence_id(word_id):
        "".join(word_id.split(sep='_')[:-1])
