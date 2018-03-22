####
# Trains a Bidirectional LSTM on the IMDB sentiment classification task.
# Baseline Model: 0.8339% validation accuracy (4 epochs)
# Best Model:
#     LSTM(64, kernel_initializer='he_normal'),
#     Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
#     => 0.8432% validation accuracy (4 epochs)
####

from __future__ import print_function
import numpy as np
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb
from gensim.models import Word2Vec

#
# Main
#
max_features = 20000
maxlen = 100
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
y_train = np.array(y_train)
y_test = np.array(y_test)

#
# LSTM Structure
#
model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(64, kernel_initializer='he_normal')))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

opt = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
model.compile(optimizer=opt, 
	          loss='binary_crossentropy', 
	          metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=4,
          validation_data=[x_test, y_test])