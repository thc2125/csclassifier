import numpy as np

from alphabet_detector import AlphabetDetector
from sklearn.preprocessing import label_binarize

import utils

ad = AlphabetDetector()

def read_corpus(corpus,
                corpus_filepath, 
                langs,
                dl='\t'):

    """Reads in a corpus file and sets the corpus variables.
    
    Keyword arguments:
    corpus -- A corpus object to ingest the file
    corpus_filepath -- The filepath to a normalized corpus file
    langs -- A tuple of 2-character language codes to be found in the corpus
    """
    with corpus_filepath.open() as corpus_file:
        corpus_reader = csv.DictReader(corpus_file, delimiter=dl)

        for token_entry in corpus_reader:
            corpus.ingest_token(token_entry, corpus_filepath.name)

def randomly_read_corpus(corpus1, 
                         corpus2, 
                         corpus_filepath, 
                         langs, 
                         dl='\t', 
                         test_split=.1):
    """Reads in a corpus file, splitting between two corpus objects.
    
    Keyword arguments:
    corpus -- A corpus object to ingest the file
    corpus_filepath -- The filepath to a normalized corpus file
    langs -- A tuple of 2-character language codes to be found in the corpus
    """
    with open(corpus_filepath) as corpus_file:
        corpus_reader = csv.DictReader(corpus_file, delimiter=dl)

        for row in corpus_reader:
            sentence_id = utils.get_sentence_id(row['token_id'])

            if sentence_id in corpus1.sentence2sidx:
                corpus1.read_row(row, corpus_filepath.name, langs)
            elif sentence_id in corpus2.sentence2sidx:
                corpus2.read_row(row, corpus_filepath.name, langs)
            elif random.random() > test_split:
                corpus1.read_row(row, corpus_filepath.name, langs)
            else:
                corpus2.read_row(row, corpus_filepath.name, langs)

    return corpus1, corpus2

def np_idx_conversion(corpus, maxsentlen, maxwordlen, char2idx, label2idx):
    # Convert the sentences and labels to lists of indices
    # Convert words to indices 
    # And pad the sentences and labels

    np_sentences = np.array(sent_idx_conversion(corpus.sentences, 
                                                maxsentlen, 
                                                maxwordlen,
                                                char2idx,
                                                corpus.char_frequency))

    np_slabels, np_labels_weights = tuple(map(
            np.array, label_idx_conversion(corpus, maxsentlen, maxwordlen)))

    # Finally convert the sentence and label ids to numpy arrays
    np_slabels = np.array(list_labels)
    return np_sentences, np_slabels, np_labels_weights

def sent_idx_conversion(sentences, 
                        maxsentlen, 
                        maxwordlen, 
                        char2idx, 
                        char_frequency=None):
        # Create a list of lists of lists of indices
        # Randomly assign some letters the index of unknown characters for a 
        # given alphabet
        list_sentences = ([[[(char2idx[c] 
                              if (c in char2idx and not unk_replace(c, char_frequency))
                              else char2idx[get_unk(c)])
                     for c in word]
                + [0]*(maxwordlen-len(word)) 
            for word in sentence]
                + [[0]*maxwordlen]*(maxsentlen-len(sentence)) 
            for sentence in sentences])
        return list_sentences

def label_idx_conversion(slabels, 
                         maxsentlen, 
                         maxwordlen, 
                         label2idx):
        list_cat_labels = ([
                [[label2idx[label]
                  for label in labels]
                + [0] * (maxsentlen-len(sentlabels))
                for labels in self.slabels])

        # Make labels one-hot
        enc = label_binarize(label2idx))
        list_labels = ([enc.transform(sentlabels
                ) 
            for sentlabels in list_cat_labels])

        list_labels_weights = ([[(1 if label != 0 else 0) for label in list_slabels] 
            for list_slabels in list_cat_labels])

        return list_labels, list_labels_weights

def unk_replace(c, char_frequency=None):
        # Formula sourced from Quora:
        # https://www.quora.com/How-does-sub-sampling-of-frequent-words-work-in-the-context-of-Word2Vec
        # "Improving Distributional Similarity with Lessons Learned from Word Embeddings"
        # Levy, Goldberg, Dagan
        if not char_frequency:
            return False

        t = .00001
        f = char_frequency[c]
        p = 1 - sqrt(t/f)
        if random.random() > p:
            return True
        else:
            return False

def get_unk(self, c, use_alphabets):
    unk = 'unk'
    if use_alphabets:
        alph = list(ad.detect_alphabet(c))
        if alph and alph[0] in alphabets:
            unk += alph[0]
    return unk

def deduce_cs_langs(cs_corpus_dirname):
    """Deduce the languages of code-switched tweets based on path names
    Keyword arguments:
    cs_corpus_dirpath -- a Path object 
    """
    lang_pattern = re.compile("[a-z]{2,2}[+][a-z]{2,2}")
    matches = lang_pattern.search(cs_corpus_dirname)
    if matches:
        langs = matches.group(0).split('+')
        return (langs[0], langs[1])
    else:
        return (None, None)

def get_cslabels(corpus):
    idx2label = corpus.get_cslabels()
    label2idx = {idx2label[i] : i for i in range(len(idx2label))}
    return idx2label, label2idx

def get_chars(corpus, use_alphabets):
    idx2char = train_corpus.get_chars()
    char2idx = {idx2char[i] : i for i in range(len(idx2char))}

    if use_alphabets:
        # Add indices for unseen chars for each alphabet representable 
        # by unicode
        for a in alphabets:
            char2idx['<unk' + a + '>'] += len(self.idx2char)
            self.idx2char.append('<unk' + a + '>')
    # Finally add a generic unknown character
    self.char2idx['<unk>'] += len(self.idx2char)
    self.idx2char.append('<unk>')

    return char2idx, idx2char
