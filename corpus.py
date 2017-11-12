# Design of utils file and Corpus class inspired by 
# Columbia University
# COMS 4705W 2017 
# Assignment 4
# Professor Karl Stratos

import csv

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

        self.char_frequency = Counter()
        self.label_frequency = Counter()
        self.lang_frequency = Counter()

        self.filenames = set()

    def ingest_token(self, token_entry, corpus_filename):
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
           return None

        label =token_entry['label']
        lang = token_entry['lang']

        # Add to the set of filenames and language pairs
        self.filenames.add(corpus_filename)
           
        # Remove the word id at the end of the sentence name
        sentence_id = utils.get_sentence_id(token_entry['token_id'])

        if sentence_id not in self.sentence2sidx:
           self._add_sentence(sentence_id)

        sidx = self.sentence2sidx[sentence_id]

        self._add_word(word, sidx)
        self._add_label(label, sidx)
        self._add_lang(lang, sidx)

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

        # Get the character frequency for a word.
        for c in word:
            self.char_frequency.update(c)
        return word

    def _add_label(self, label, sidx):
        self.slabels[sidx].append(label)
        self.label_frequency.update([label])
        return label

    def _add_lang(self, lang, sidx):
        #self.slangs[sidx].append(lang)
        self.lang_frequency.update([lang])
        return lang

    def get_chars(self):
        return list(self.char_frequency.keys())

    def get_labels(self):
        return list(self.label_frequency.keys())

    def get_langs(self):
        return list(self.lang_frequency.keys())

