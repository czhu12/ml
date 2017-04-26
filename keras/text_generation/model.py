from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import pdb

data = ""
for i in range(5):
    data += open('data/text{}.txt'.format(i + 1)).read()

batch_size = 200
vocab = list(set(data))
data_size, vocab_size = len(data), len(vocab)
char_to_ix = { ch:i for i,ch in enumerate(vocab) }
ix_to_char = { i:ch for i,ch in enumerate(vocab) }

X = np.zeros((
    len(list(data)) - batch_size,
    batch_size,
    len(vocab),
))

Y = np.zeros((len(list(data)) - batch_size, len(vocab)))

def encode_text(x):
    encoding = np.zeros((batch_size, len(vocab)))
    for i in range(len(x)):
        encoding[i, char_to_ix[x[i]]] = 1
    return encoding

for i in range(len(list(data)) - batch_size):
    x = data[i : i + batch_size]
    X[i, :, :] = encode_text(x)

    y = data[i + batch_size]

    Y[i, char_to_ix[y]] = 1

model = Sequential()
model.add(LSTM(128, input_shape=(batch_size, len(vocab))))
model.add(Dense(len(vocab)))
model.add(Activation('softmax'))
model.compile(optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'])


model.fit(X, Y, batch_size=32)
