# -*- coding: utf-8 -*-
# Tensorflow 2.x

import pandas as pd
import numpy as np
import math
import random


import tensorflow as tf
from keras.models import Sequential  
from keras.layers.core import Dense, Activation  
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
in_out_neurons = 1
hidden_neurons = 300
length_of_sequences = 100

model = Sequential()  
model.add(LSTM(hidden_neurons, batch_input_shape=(None, length_of_sequences, in_out_neurons), return_sequences=False))  
model.add(Dense(in_out_neurons))  
model.add(Activation("linear"))  
model.compile(loss="mean_squared_error", optimizer="rmsprop")

# early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

X_train=[]
y_train=[]
model.fit(X_train, y_train, batch_size=1, epochs=15, validation_split=0.05, callbacks=[early_stopping]) 