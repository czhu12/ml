import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
import argparse

input_size = 50
def lstm_model(classes):
    model = Sequential()
    model.add(LSTM(256, input_shape=(input_size, 1)))
    model.add(Dropout(0.2))
    model.add(Dense(classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()

    return model


def train(textfile):
    text = open(textfile).read()
    text = text.lower()
    
    idx_to_char = sorted(list(set(text)))
    char_to_idx = dict((c, i) for i, c in enumerate(idx_to_char))

    X = []
    Y = []
    for i in range(0, len(text) - input_size):
        x = [char_to_idx[x] for x in text[i: i + input_size]]
        y = char_to_idx[text[i + input_size]]

        X.append(x)
        Y.append(y)

    X = np.array(X) / float(len(idx_to_char))
    X = X.reshape(X.shape[0], X.shape[1], 1)
    Y = np_utils.to_categorical(np.array(Y))
    model = lstm_model(Y.shape[1])
    model.fit(X, Y, nb_epoch=20, batch_size=128)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--textfile", help="force parse image files")
    args = parser.parse_args()
    train(args.textfile)

if __name__ == "__main__":
    main()
