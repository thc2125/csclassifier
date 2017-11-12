from corpus import Corpus

class Corpus_CS_Langs(Corpus):
    def __init__(self):

        Corpus.__init__(self)

        self.scslabels = []
        self.cslabel_frequency = Counter()

    def _init_data(self):
        self.switch_count = 0
        self.switch_label_counts = defaultdict(int)
        self.multilingual_sentence_count = 0
        self.cs_labels = []
        Corpus._init_data(self)

    def ingest_token(self, token_entry, corpus_filename):
        sidx = Corpus.ingest_token(self, token_entry, corpus_filename)
        if(not sidx):
            return None
        cslabel = token_entry['cs_label']
        self._add_cslabel(cslabel, sidx)

    def _add_cslabel(self, cslabel, sidx):
        self.scslabels[sidx].append(cslabel)
        self.cslabel_frequency.update([cslabel])
        return cslabel

    def get_cslabels(self):
        return list(self.cslabel_frequency.keys())

    def np_idx_conversion(self, maxsentlen, maxwordlen):
        list_sentences, list_labels, list_labels_weights = (
            Corpus.np_idx_conversion(self, maxsentlen, maxwordlen)

        list_cs_labels, list_cs_labels_weights = self.cs_labels_idx_conversion(
            maxsentlen, maxwordlen)

        return (list_sentences, list_labels, list_labels_weights, 
            list_cs_labels, list_cs_labels_weights)
    
    def cs_labels_idx_conversion(self, maxsentlen, maxwordlen):
        list_cat_cs_labels = ([[self.cs_label2idx[label] for label in sentlabels] 
                + [0] * (maxsentlen-len(sentlabels)) 
            for sentlabels in self.cs_labels])
        # Make labels one-hot
        list_cs_labels = ([to_categorical(sentlabels, 
                num_classes=len(self.cs_label2idx)) 
            for sentlabels in list_cat_cs_labels])

        list_cs_labels_weights = ([[(1 if label != 0 else 0) for label in list_slabels] 
            for list_slabels in list_cat_cs_labels])

        return list_cs_labels, list_cs_labels_weights


