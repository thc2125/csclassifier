# Design of utils file and Corpus class inspired by 
# Columbia University
# COMS 4705W 2017 
# Assignment 4
# Professor Karl Stratos

from collections import defaultdict

from alphabet_detector import AlphabetDetector

class Corpus():
    def __init__(self, char_dictionary=(None, None), 
        label_dictionary=(None, None), train=False, use_alphabets=False):
        """Reads in a corpus file and sets the corpus variables.
    
        Keyword arguments:
        char_dictionary -- A tuple of dictionaries for characters to indices 
                           and indices to characters
        label_dictionary -- A tuple of dictionaries for labels to indices 
                           and indices to labels
        """

        # Set the dictionary if one is provided 
        self.char2idx, self.idx2char = char_dictionary

        # We also need an a priori set of labels for each word
        self.label2idx, self.idx2label = label_dictionary
        self.train = train
        self.use_alphabets = use_alphabets
        if use_alphabets:
            self.ad = AlphabetDetector()
        self._init_data()

    def _init_data(self):
        self.sentences=[]
        self.labels=[]
        self.label_counts = defaultdict(int)
        self.maxwordlen = 0
        self.maxsentlen = 0
        self.sentence2sidx = {}
        self.sidx = 0
        self.char_frequency = defaultdict(int)


    def read_corpus(self, corpus_filename, dl=','):
        """Reads in a corpus file and sets the corpus variables.
    
        Keyword arguments:
        corpus_filepath -- The filepath to a normalized corpus
        """
        self.corpus_filename = corpus_filename
        with open(corpus_filename) as corpus_file:
            corpus_reader = csv.reader(corpus_file, delimiter=dl)

            # Skip the header
            next(corpus_reader)
            for row in corpus_reader:
                self.read_row(row)

    def read_row(self, row):

        """Reads a csv row and updates the Corpus variables.
    
        Keyword arguments:
        row -- a list of csv row values ([sentence_id, word, lang_label,...])
        """

        word = row[1]
        # NOTE: This puts a max word length on a word
        # Length arbitrary based on
        # len("supercalifragilisticexpialidocious")
        if len(word) > 34:
           return
        self.maxwordlen = max(self.maxwordlen, len(word))
           

        # Remove the word id at the end of the sentence name
        sname = ''.join(row[0].split(sep='_')[:-1])

        if sname not in self.sentence2sidx:
           self.add_sentence(sname)
        self.maxsentlen = max(self.maxsentlen, len(self.sentences[sname]))

        label = self.label_word(row[2])

        nsidx = self.sentence2sidx[sname]
        self.sentences[nsidx].append(word)
        # Get the character frequency for a word.
        for c in word:
           self.char_frequency[c] += 1
        self.labels[nsidx].append(label)

    def add_sentence(self, sname):
           self.sentence2sidx[sname] = self.sidx
           self.sidx +=1
           self.sentences.append([])
           self.labels.append([])

    def label_word(self, label):
        self.label_counts[label] += 1
        return label

    def __add__(self, other):
        #TODO: add an exception if the two corpora have any common sentences
        corp = Corpus()
        return self._combine(corp, other)
        

    def _combine(self, corp, other):
        corp.sidx = len(self.sentences)
        corp.sentences = self.sentences + other.sentences

        corp.labels = self.labels + other.labels

        for k, v in self.label_counts.items():
            corp.label_counts[k] += v
        for k, v in other.label_counts.items():
            corp.label_counts[k] += v

        corp.sentence2sidx = self.sentence2sidx.copy() 
        corp.sentence2sidx.update({s : (i + self.sidx) for s, i in 
                other.sentence2sidx.items()})
        corp.sidx = len(corp.sentences)
        corp.maxsentlen = max(self.maxsentlen, other.maxsentlen)
        corp.maxwordlen = max(self.maxwordlen, other.maxwordlen)
        # TODO: Is it okay to turn a defaultdict into a counter?
        corp.char_frequency = (Counter(self.char_frequency) 
            + Counter(other.char_frequency))
        corp.train = True if (self.train or other.train) else False
        corp.use_alphabets = self.use_alphabets or other.use_alphabets

        return corp




