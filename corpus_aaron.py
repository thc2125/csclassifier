from corpus import Corpus


class Corpus_Aaron(Corpus):
    def __init__(self, char_dictionary=(None, None), label_dictionary=(None, None)):
        """Reads in a corpus file and sets the corpus variables.
    
        Keyword arguments:
        char_dictionary -- A tuple of dictionaries for characters to indices 
                           and indices to characters
        label_dictionary -- A tuple of dictionaries for labels to indices 
                           and indices to labels

        """
        label2idx = ({'<PAD>':0, 'lang1': 1, 'lang2':2, 'other':3, 'ne':4, 
        'ambiguous':5, 'fw':6, 'mixed':7, 'unk':8})
        idx2label = {i:l for l, i in self.label2idx.items()}

        Corpus.__init__(self, label_dictionary=(label2idx, idx2label))

    def __add__(self, other):
        corp = Corpus_Aaron()
        return Corpus._combine(self, corp, other)


