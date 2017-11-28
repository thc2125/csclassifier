import json
import time

from math import log, ceil
from pathlib import Path

import keras.backend as K
import keras.models
import numpy as np

from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Embedding, Conv1D, Dropout, GlobalMaxPooling1D
from keras.layers import Dense, Merge, TimeDistributed, Bidirectional, LSTM
from keras.layers import add, concatenate
from keras.models import Model

import corpus_utils
import utils

DEFAULT_HYPER_PARAMETERS = {'n_1':59,
                            'n_2':108, 
                            'lstm_dim':23, 
                            'dropout_rate':.25, 
                            'kernel_size':3, 
                            'loss':'categorical_crossentropy', 
                            'optimizer':'adam', 
                            'learning_rate':.001, 
                            'decay':0}

class Classifier:

    def __init__(self, 
                 chars, 
                 labels,
                 maxsentlen, 
                 maxwordlen, 
                 use_alphabets=False,
                 hyper_parameters=DEFAULT_HYPER_PARAMETERS,
                 model_path=None):

        self.idx2char, self.char2idx = utils.get_char_dicts(chars, use_alphabets)
        self.idx2label, self.label2idx = utils.get_label_dicts(labels)
        
        self.maxsentlen = maxsentlen
        self.maxwordlen = maxwordlen

        self.use_alphabets = use_alphabets

        if model_path:
            self.model = self._load_model(str(model_path))

        else:
            self._generate_model(**hyper_parameters)

    def _generate_model(self, 
                        n_1=59,
                        n_2=108, 
                        lstm_dim=23, 
                        dropout_rate=.25, 
                        kernel_size=3, 
                        loss='categorical_crossentropy', 
                        optimizer='adam', 
                        learning_rate=.001, 
                        decay=0):
        # Create the model
        self.num_labels = len(self.idx2label)

        self.C = self.char2idx.values()

        self.Cdim = ceil(log(len(self.C), 2))

        # Hyperparameters tuned by Jaech et. al. 
        self.n_1 = n_1
        self.n_2 = n_2
        self.lstm_dim = lstm_dim
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.loss = loss
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.decay = decay
        if self.optimizer_name == 'adam':
            self.optimizer = optimizers.Adam(lr=learning_rate, decay=decay)
        else:
            raise ValueError("Invalid optimizer passed to Classifier")

        # First let's set up Char2Vec
        # TODO: Here's an idea on masking: make another input vector for each 
        # sentence that applies masking yourself. I.e. after the embedding layer,
        # Multiply those values by a vector of [1,1,1,0,0,0] with 1 
        # corresponding to values you want and 0 to values you don't want.
        # This could be created during corpus ingestion and just be another value.
        # Maybe pop this input/merge right after the embedding layer and before 
        # the CNN.
        inputs = Input(shape=(self.maxsentlen, self.maxwordlen,))

        # Set up the embeddings
        # TODO: Add the masking layer for the padded value
        embeddings = TimeDistributed(Embedding(len(self.C),
            self.Cdim))(inputs)

        # Make T_1 (1st CNN)
        T_1 = TimeDistributed(Conv1D(
            # input shape can be deduced
            # 'filters' - the number of filters i.e. dimensionality of output
            filters=n_1,
            # 'kernel_size' - size of the window
            kernel_size=kernel_size,
            # 'padding' ensures the output will be the same length as input
            padding='same',
            activation = 'relu'))(embeddings)



        # Adding the dropout
        T_1_dropout = Dropout(rate=dropout_rate)(T_1)

        # Adding T_2 (2nd CNN)
        T_2_a = TimeDistributed(Conv1D(
            filters=n_2,
            kernel_size = (3),
            padding='valid',
            activation='relu'))(T_1_dropout)

        T_2_b = TimeDistributed(Conv1D(
            filters=n_2,
            kernel_size = (4),
            padding='valid',
            activation='relu'))(T_1_dropout)

        T_2_c = TimeDistributed(Conv1D(
            filters=n_2,
            kernel_size = (5),
            # TODO: T_2 uses padding so you end up with a vector for each
            # character right?
            padding='valid',
            activation='relu'))(T_1_dropout)

        # Adding y (max-pooling across time)
        y_a = TimeDistributed(GlobalMaxPooling1D())(T_2_a)
        y_b = TimeDistributed(GlobalMaxPooling1D())(T_2_b)
        y_c = TimeDistributed(GlobalMaxPooling1D())(T_2_c)
        y = concatenate([y_a, y_b, y_c])

        # Add f_r(y) 
        f_r_y = Dense(3 * n_2, activation='relu')(y)
        z = add([y, f_r_y])

        # The next phase is dealing with all the words in the sentence.     
        # TODO: Are default lstm activation functions okay?
        v = Bidirectional(LSTM(lstm_dim,
            return_sequences=True, dropout=dropout_rate))(z)
        p = Dense(self.num_labels, activation='softmax')(v)
        self.model = Model(inputs=inputs, outputs=p)
        # Note that 'adam' has a default learning rate of .001
        self.model.compile(loss=loss, 
            optimizer=optimizer, 
            metrics=['categorical_accuracy'], sample_weight_mode='temporal')

        self.trained_epochs = 0

    def _load_model(self, model_path):
        self.model = keras.models.load_model(model_path)

    def plot_model(self):
        keras.utils.plot_model(self.model, show_shapes=True)

    def train_model(self, 
                    train_corpus, 
                    epochs,
                    batch_size,
                    patience,
                    output_dirpath='.'):
        if (train_corpus.maxsentlen > self.maxsentlen or 
            train_corpus.maxwordlen > self.maxwordlen):
            raise Exception("'train_corpus' has greater maxsentlen or maxwordlen")

        raw_train_sentences = self._get_corpus_sentences(train_corpus)
        raw_train_labels = self._get_corpus_labels(train_corpus)

        train_sentences, train_labels = corpus_utils.np_idx_conversion(
                                            raw_train_sentences, 
                                            raw_train_labels, 
                                            self.maxsentlen,
                                            self.maxwordlen,
                                            self.char2idx,
                                            self.idx2label,
                                            train_corpus.char_count)

        # Train the model
        # Create a folder to store checkpoints if one does not exist
        checkpoints_dirpath = Path(output_dirpath / 'checkpoints')
        if not checkpoints_dirpath.exists():
            checkpoints_dirpath.mkdir()
        checkpoint = ModelCheckpoint(
                         filepath=str(checkpoints_dirpath/('cp_classifier_model_'
                                      + "_"
                                      + '.{epoch:02d}--'
                                      + '{val_loss:.2f}.hdf5')),
                         monitor='val_loss', 
                         mode='min')
        stop_early = EarlyStopping(monitor='val_categorical_accuracy',
                                   patience=patience)
        self.history = self.model.fit(x=train_sentences, 
                                      y=train_labels,
                                      epochs=epochs, 
                                      batch_size=batch_size, 
                                      validation_split=.1,
                                      #sample_weight=train_labels_weights, 
                                      callbacks=[checkpoint, 
                                      stop_early])
        self.trained_time = time.time()
        self.trained_epochs += len(self.history.epoch)
        
    def save_model(self, output_dirpath=Path('.')):
        # Save the model
        model_path = (output_dirpath
                            / ('classifier_model_' 
                               + str(self.trained_time)
                               + '_' 
                               + '.h5'))
        self.model.save(str(model_path))

        # Save the history
        model_history_path = (model_path.parent 
                             / (model_path.stem + '_history.json'))

        with (model_history_path).open('w') as history_file:
            json.dump(self.history.history, 
                      history_file, 
                      indent=4, 
                      sort_keys=True)

        return model_path, model_history_path 

    def evaluate_model(self, test_corpus):
        if (test_corpus.maxsentlen > self.maxsentlen or 
            test_corpus.maxwordlen > self.maxwordlen):
            raise Exception("'train_corpus' has greater maxsentlen or maxwordlen")

        raw_test_sentences = self._get_corpus_sentences(test_corpus)
        raw_test_labels = self._get_corpus_labels(test_corpus)

        test_sentences, test_labels = corpus_utils.np_idx_conversion(
                                            raw_test_sentences, 
                                            raw_test_labels, 
                                            self.maxsentlen,
                                            self.maxwordlen,
                                            self.char2idx,
                                            self.idx2label)

        print("Testing on sentences of shape: " + str(test_sentences.shape))
        pred_labels = self.model.predict(x=test_sentences)

        # Transform labels to represent category index
        test_cat_labels = np.argmax(test_labels, axis=2)
        pred_cat_labels = np.argmax(pred_labels, axis=2)

        metrics = {}
        metrics['word'] = (utils.compute_accuracy_metrics(
            test_cat_labels, pred_cat_labels, self.label2idx))

        test_sent_cat_labels = np.amax(test_labels, axis=1)
        pred_sent_cat_labels = np.amax(pred_labels, axis=1)

        metrics['sentence'] = (utils.compute_accuracy_metrics(
            test_sent_cat_labels, pred_sent_cat_labels, self.label2idx))

        return metrics 

    def _get_corpus_labels(self, corpus):
        return corpus.slabels

    def _get_corpus_sentences(self, corpus):
        return corpus.sentences
