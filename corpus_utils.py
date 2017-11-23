import random

from math import sqrt

import numpy as np

from alphabet_detector import AlphabetDetector
from sklearn.preprocessing import label_binarize

import utils

ad = AlphabetDetector()

def np_idx_conversion(sentences, slabels, maxsentlen, maxwordlen, char2idx, idx2label, char_count=None):
    # Convert the sentences and labels to lists of indices
    # Convert words to indices 
    # And pad the sentences and labels

    np_sentences = np.array(sent_idx_conversion(sentences, 
                                                maxsentlen, 
                                                maxwordlen,
                                                char2idx,
                                                char_count))

    np_slabels = np_label_idx_conversion(slabels, maxsentlen, idx2label)

    # Finally convert the sentence and label ids to numpy arrays
    return np_sentences, np_slabels

def sent_idx_conversion(sentences, 
                        maxsentlen, 
                        maxwordlen, 
                        char2idx, 
                        char_count=None):
        # Create a list of lists of lists of indices
        # Randomly assign some letters the index of unknown characters for a 
        # given alphabet
        list_sentences = ([[[(char2idx[c] 
                              if (c in char2idx and not unk_replace(c, char_count))
                              else char2idx[get_unk(c)])
                     for c in word]
                + [0]*(maxwordlen-len(word)) 
            for word in sentence]
                + [[0]*maxwordlen]*(maxsentlen-len(sentence)) 
            for sentence in sentences])
        return list_sentences

def np_label_idx_conversion(slabels, 
                            maxsentlen, 
                            idx2label):
        '''
        list_cat_labels = ([
                [[label2idx[label]
                  for label in labels]
                + [0] * (maxsentlen-len(sentlabels))
                for labels in self.slabels])
        '''

        # Make labels one-hot
        list_labels = np.stack(
            [np.concatenate((label_binarize(labels, 
                                           idx2label),
                            label_binarize(['<pad>'] 
                                           * (maxsentlen
                                              - len(labels)),
                                           idx2label)))
             if len(labels) < maxsentlen
             else label_binarize(labels, idx2label)
             for labels in slabels])
        '''
        np_slabels = []
        for labels in slabels:
            print("LABELS: " + str(labels))
            np_labels = label_binarize(labels, idx2label)
            np_padding = label_binarize(['<pad>'] * (maxsentlen - len(labels)), idx2label)

            print("NP_LABELS: " + str(np_labels))
            print("NP_PADDING: " + str(np_padding))
            np_slabels.append(np.concatenate((np_labels, np_padding)))
        

        list_labels = np.stack(np_slabels)
        '''
        return list_labels

def unk_replace(c, char_count=None):
        # Formula sourced from Quora:
        # https://www.quora.com/How-does-sub-sampling-of-frequent-words-work-in-the-context-of-Word2Vec
        # "Improving Distributional Similarity with Lessons Learned from Word Embeddings"
        # Levy, Goldberg, Dagan
        if not char_count:
            return False

        t = .00001
        f = char_count[c]
        p = 1 - sqrt(t/f)
        if random.random() > p:
            return True
        else:
            return False

def get_unk(c, use_alphabets=False):
    unk = '<unk'
    if use_alphabets:
        alph = list(ad.detect_alphabet(c))
        if alph and alph[0] in alphabets:
            unk += alph[0]
    unk += '>'
    return unk
