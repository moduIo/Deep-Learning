####
# LSTM for sentiment analysis on the IMDB dataset.
# Tutorial code: https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py
# Baseline Model: 0.8063% validation accuracy
# Best Model:
#     Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.2, amsgrad=False): 0.8440% validation accuracy
####
from __future__ import print_function
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.layers.normalization import BatchNormalization
from keras.datasets import imdb

#
# Global parameters
#
max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 64

#
# Load and process data
#
print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

#
# LSTM Structure
#
print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
#model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))

opt = keras.optimizers.Adam(lr=0.0125, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

#
# Training
#
print('Training...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))

model.evaluate(x_test, y_test,
               batch_size=batch_size,
               verbose=2)