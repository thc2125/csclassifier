from corpus import Corpus

class Corpus_CS_Langs(Corpus):
    def __init__(self, char_dictionary=(None, None), train=False, use_alphabets=False):
        """Reads in a corpus file and sets the corpus variables.
    
        Keyword arguments:
        dictionary -- A tuple of dictionaries for characters to indices and
                      indices to characters
        """
        cs_label2idx = {'<PAD>':0, 'no_cs': 1, 'cs':2}
        cs_idx2label = {i:l for l, i in label2idx.items()}
        label_dictionary = {'<PAD>':0, 'lang1':1, 'lang2':2, 'punct':3, 
            'other':4}

        Corpus.__init__(self, char_dictionary,
                label_dictionary=label_dictionary, train=train, 
                 use_alphabets=use_alphabets)

    def _init_data(self):
        self.switch_count = 0
        self.switch_label_counts = defaultdict(int)
        self.multilingual_sentence_count = 0
        self.cs_labels = []
        Corpus._init_data(self)

    def __add__(self, other):
        corp = Corpus_CS_Langs()
        return self._combine(corp, other)

    def _combine(self, corp, other):
        # NOTE: The proper functioning of this depends on both corpora being 
        # composed of completely different sentences.
        corp.switch_count += self.switch_count + other.switch_count 
        corp.multilingual_sentence_count += (self.multilingual_sentence_count 
            + other.multilingual_sentence_count)

        for k, v in self.switch_label_counts.items():
            corp.switch_label_counts[k] += v
        for k, v in other.label_counts.items():
            corp.switch_label_counts[k] += v

        corp.cs_labels += self.cs_labels + other.cs_labels


        return Corpus._combine(self, corp, other)

    def read_corpus(self, corpus_filepath, dl):
        Corpus.read_corpus(self, corpus_filepath, dl)
        for slabels in self.labels:
            cs_slabels = []
            fragments, fragment_indices = (compute_switch_errors())
            last_index = 0
            for fragment_index in range(len(fragments)):
                # Add to the CS label counts
                self.switch_label_counts[tuple(fragments[fragment_index])] += 1
                
                # Create a list of CS labels for the fragment and any 
                # labels between the previous fragment and the current one
                cs_slabels += (['no_cs'] * (len(fragments[fragment_index])-1 
                        + (fragment_indices[fragment_index][0]-last_index-1)))
                cs_slabels += ['cs']
                # Store the last index 
                last_index = fragment_indices[fragment_index][-1]
            # Add the CS labels to the corpus cs labels
            self.cs_labels.append(cs_slabels)


    # Credit Victor Soto
    def switch_at_the_fringe(langids):
        return (langids[0] == 'lang1' and langids[-1] == 'lang2') or (langids[0] == 'lang2' and langids[-1] == 'lang1')

    # Credit Victor Soto
    def compute_switch_errors(true_tags):
        num_switches = 0
        num_tokens_in_switch = 0
        j0 = 0
        j1 = 1
        fragment_labels = []
        fragment_indices = []
        while(j0 < j1 and j1 < len(true_tags)):
                while true_tags[j0]!='lang2' and true_tags[j0]!='lang1' and j0 < len(true_tags) - 1:
                        j0+=1
                        j1=j0 + 1
                if switch_at_the_fringe(true_tags[j0:j1+1]):
                        fragment_labels.append(true_tags[j0:j1+1])
                        fragment_indices.append(list(range(j0, j1+1)))
                        num_switches +=1
                        num_tokens_in_switch = j1 + 1 - j0
                        j0 = j1
                        j1 += 1
                else:
                        j1 += 1
                        if j1 < len(true_tags) and true_tags[j0] == true_tags[j1-1]:
                                j0 = j1-1
        if j0>=j1:
                raise Exception('This should not happen.')
        return fragment_labels, fragment_indices


    def read_corpus(self, corpus_filepath, dl):
        Corpus.read_corpus(self, corpus_filepath, dl)

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


