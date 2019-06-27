# Neural network approaches to ECG sequences
"""
Created on Mon Jun 17 17:45:34 2019

@author: BB
"""

#%% Libraries

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
import numpy as np
from sklearn.model_selection import train_test_split

#%% Load sequence data
X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(subX, subY, test_size=.3)

X_train_seq = sequence.pad_sequences(X_train_seq , maxlen=9000)
X_test_seq = sequence.pad_sequences(X_test_seq, maxlen=9000)

X_train_seq = X_train_seq.astype('float')
X_test_seq = X_test_seq.astype('float')

# https://stackoverflow.com/questions/43396572/dimension-of-shape-in-conv1d
X_train_seq = X_train_seq.reshape(X_train_seq.shape[0], X_train_seq.shape[1], 1)
X_test_seq = X_test_seq.reshape(X_test_seq.shape[0], X_test_seq .shape[1], 1)

y_train_seq = np.array(y_train_seq)
y_test_seq = np.array(y_test_seq)


#%% model

model = Sequential()
model.add(layers.Conv1D(256, 9, activation='relu', input_shape=(9000, 1)))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(128, 9, activation='relu'))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(64, 9, activation='relu'))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(32, 9, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))

model.summary()
model.compile(optimizer=RMSprop(), loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train_seq, y_train_seq, epochs=3, batch_size=128, validation_split=.2)

