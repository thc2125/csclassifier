import math

from collections import Counter

from corpus import Corpus

class CorpusCSLangs(Corpus):
    def __init__(self):

        Corpus.__init__(self)

        self.scslabels = []
        self.cslabel_count = Counter()
        self.cstype_count = Counter()

        self.lang_pair_count = Counter()

        self.multilingual_sentences = set()

    def _ingest_token(self, token_entry):
        sidx = Corpus._ingest_token(self, token_entry)
        if(math.isnan(sidx)):
            return None

        lang_pair = token_entry['lang_pair']
        self._add_lang_pair(lang_pair)

        cslabel = token_entry['cs_label']
        self._add_cslabel(cslabel, sidx)

        cstype = token_entry['cs_type']
        self._add_cstype(cstype, sidx)

    def _add_sentence(self, sname):
        Corpus._add_sentence(self, sname)
        self.scslabels.append([])

    def _add_lang_pair(self, lang_pair):
        self.lang_pair_count.update([lang_pair])
        return lang_pair

    def _add_cslabel(self, cslabel, sidx):
        self.scslabels[sidx].append(cslabel)
        self.cslabel_count.update([cslabel])
        if cslabel == 'cs':
            self.multilingual_sentences.add(sidx)
        return cslabel

    def _add_cstype(self, cstype, sidx):
        if cstype != '<na>':
            self.cstype_count.update([cstype])
        return cstype

    def get_cslabels(self):
        return list(self.cslabel_count.keys())

    def get_cstypes(self):
        return list(self.cstype_count.keys())
