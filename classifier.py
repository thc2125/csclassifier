from math import log, ceil

import keras.backend as K

from keras.layers import Input, Embedding, Conv1D, Dropout, GlobalMaxPooling1D
from keras.layers import Dense, Merge, TimeDistributed, Bidirectional, LSTM
from keras.layers import add, concatenate
from keras.models import Model
#from keras.utils import plot_model


class Classifier:
    def __init__(self, char2idx, maxsentlen, maxwordlen, num_labels):

        self.C = char2idx.values()

        self.Cdim = ceil(log(len(self.C), 2))

        # Hyperparameters tuned by Jaech et. al. 
        self.n_1 = 59
        self.n_2 = 108
        self.lstm_dim = 23
        self.dropout_rate = .25
        self.kernel_size = 3

        # First let's set up Char2Vec
        # TODO: Here's an idea on masking: make another input vector for each 
        # sentence that applies masking yourself. I.e. after the embedding layer,
        # Multiply those values by a vector of [1,1,1,0,0,0] with 1 
        # corresponding to values you want and 0 to values you don't want.
        # This could be created during corpus ingestion and just be another value.
        # Maybe pop this input/merge right after the embedding layer and before 
        # the CNN.
        self.inputs = Input(shape=(maxsentlen, maxwordlen,))

        # Set up the embeddings
        # TODO: Add the masking layer for the padded value
        self.embeddings = TimeDistributed(Embedding(len(self.C),
            self.Cdim))(self.inputs)

        # Make T_1 (1st CNN)
        self.T_1 = TimeDistributed(Conv1D(
            # input shape can be deduced
            # 'filters' - the number of filters i.e. dimensionality of output
            filters=self.n_1,
            # 'kernel_size' - size of the window
            kernel_size=self.kernel_size,
            # 'padding' ensures the output will be the same length as input
            padding='same',
            activation = 'relu'))(self.embeddings)



        # Adding the dropout
        self.T_1_dropout = Dropout(rate=self.dropout_rate)(self.T_1)

        # Adding T_2 (2nd CNN)
        self.T_2_a = TimeDistributed(Conv1D(
            filters=self.n_2,
            kernel_size = (3),
            padding='valid',
            activation='relu'))(self.T_1_dropout)

        self.T_2_b = TimeDistributed(Conv1D(
            filters=self.n_2,
            kernel_size = (4),
            padding='valid',
            activation='relu'))(self.T_1_dropout)

        self.T_2_c = TimeDistributed(Conv1D(
            filters=self.n_2,
            kernel_size = (5),
            # TODO: T_2 uses padding so you end up with a vector for each
            # character right?
            padding='valid',
            activation='relu'))(self.T_1_dropout)



        # Adding y (max-pooling across time)
        self.y_a = TimeDistributed(GlobalMaxPooling1D())(self.T_2_a)
        self.y_b = TimeDistributed(GlobalMaxPooling1D())(self.T_2_b)
        self.y_c = TimeDistributed(GlobalMaxPooling1D())(self.T_2_c)
        self.y = concatenate([self.y_a, self.y_b, self.y_c])

        # Add f_r(y) 
        self.f_r_y = Dense(3 * self.n_2, activation='relu')(self.y)
        self.z = add([self.y, self.f_r_y])

        # The next phase is dealing with all the words in the sentence.     
        # TODO: Are default lstm activation functions okay?
        self.v = Bidirectional(LSTM(self.lstm_dim,
            return_sequences=True, dropout=self.dropout_rate))(self.z)
        self.p = Dense(num_labels, activation='softmax')(self.v)
        self.model = Model(inputs=self.inputs, outputs=self.p)
        # Note that 'adam' has a default learning rate of .001
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', 
            metrics=['categorical_accuracy'], sample_weight_mode='temporal')
        #plot_model(self.model, show_shapes=True)

