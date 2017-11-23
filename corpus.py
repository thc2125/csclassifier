# Design of utils file and Corpus class inspired by 
# Columbia University
# COMS 4705W 2017 
# Assignment 4
# Professor Karl Stratos

import csv
import math

from collections import Counter

import utils

class Corpus():
    # TODO: Add a label set to indicate malformed data
    WORD_LEN_LIMIT = 34

    def __init__(self):

        self.maxwordlen = 0
        self.maxsentlen = 0

        self.sentences = []
        self.slabels = []
        #self.slangs = []
        self.sentence2sidx = {}

        self.char_count = Counter()
        self.label_count = Counter()
        self.lang_count = Counter()

        self.filepaths = set()

    def ingest_corpus(self,
                      corpus_filepath, 
                      dl='\t'):

        """Reads in a corpus file and sets the corpus variables.
    
        Keyword arguments:
        corpus -- A corpus object to ingest the file
        corpus_filepath -- The filepath to a normalized corpus file
        langs -- A tuple of 2-character language codes to be found in the corpus
        """
        self.filepaths.add(str(corpus_filepath))
        with corpus_filepath.open() as corpus_file:
            corpus_reader = csv.DictReader(corpus_file, delimiter=dl)

            for token_entry in corpus_reader:
                self._ingest_token(token_entry)

    def _ingest_token(self, token_entry):
        """Reads a csv token_entry and updates the Corpus variables.
    
        Keyword arguments:
        token_entry -- a dictionary of csv field names to values 
                       ([sentence_id, word, langid,label])
        corpus_filename -- a filename of the corpus ingested
        """

        word = token_entry['token']
        # NOTE: This puts a max word length on a word.
        # Length arbitrary based on
        # len("supercalifragilisticexpialidocious")
        if len(word) > self.WORD_LEN_LIMIT:
           return float('nan')

        label =token_entry['label']
        lang = token_entry['langid']
        
        # Remove the word id at the end of the sentence name
        sentence_id = utils.get_sentence_id(token_entry['token_id'])

        if sentence_id not in self.sentence2sidx:
           self._add_sentence(sentence_id)

        sidx = self.sentence2sidx[sentence_id]

        self._add_word(word, sidx)
        self._add_label(label, sidx)
        self._add_lang(lang)

        return sidx

    def _add_sentence(self, sname):
           self.sentence2sidx[sname] = len(self.sentences)
           self.sentences.append([])
           self.slabels.append([])

    def _add_word(self, word, sidx):
        self.sentences[sidx].append(word)

        self.maxwordlen = max(self.maxwordlen, len(word))
        self.maxsentlen = max(self.maxsentlen, 
            len(self.sentences[sidx]))

        # Get the character count for a word.
        for c in word:
            self.char_count.update(c)
        return word

    def _add_label(self, label, sidx):
        self.slabels[sidx].append(label)
        self.label_count.update([label])
        return label

    def _add_lang(self, lang):
        self.lang_count.update([lang])
        return lang

    def get_chars(self):
        return list(self.char_count.keys())

    def get_labels(self):
        return list(self.label_count.keys())

    def get_langs(self):
        return list(self.lang_count.keys())
