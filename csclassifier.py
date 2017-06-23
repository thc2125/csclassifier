from keras.models import Sequential, 
from keras.layers import Conv1D, Dropout, MaxPooling1D, Dense, Add, LSTM, 

class csclassifier:
    def __init__(self):
        #TODO: Fix self.cdim=log(len(C))
        self.cdim=10

        # Numbers taken from Jaech et. al tuned hyperparameters
        self.n_1 = 59
        self.n_2 = 108
        self.dropout_rate = .25

        self.model = Sequential()
        # Adding T_1 (1st CNN)
        self.model.add(Conv1D(
            input_shape=(None, self.cdim),
            # 'filters' - the number of filters i.e. dimensionality of output
            # Number taken from Jaech et. al tuned hyperparameters
            filters=59
            # 'kernel_size' - size of the window
            kernel_size = 3
            # 'padding' ensures the output will be the same length as input
            padding='same'
            activation = 'relu'))

        # Adding the dropout
        self.model.add(Dropout(rate=self.dropout_rate))

        # Adding T_2 (2nd CNN)
        self.model.add(Conv1D(
            filters=self.n_2,
            # (I don't know if making this a tuple will address multiple 
            # filter windows...)
            kernel_size = (3, 4, 5),
            # TODO: T_2 uses padding so you end up with a vector for each
            # character right?
            padding='same'
            activation='relu'))

        # Adding y (max-pooling across time)
        self.model.add(MaxPooling1D(pool_size=1, padding='same'))

        # Add f_r(y) 
        self.model.add(Dense(3 * self.n_2, activation='relu'))
        # TODO: Complete z = y + f_r(y) (currently it's only f_r(y)



        pass

    def train(self):
        """TKTK

        Keyword arguments:
        TKTK -- TKTK
         """


def transform_data():
    pass
