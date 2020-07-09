from load_data import load_data
import matplotlib.pyplot as plt
from os import listdir
import keras
from keras.preprocessing import sequence
import tensorflow as tf
from keras.models import Sequential,Model
from keras.layers import *
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization


X_train, y_train, X_test, seq_len, fea_size = load_data()
input_layer=Input(shape=(seq_len,fea_size),name="input_layer")
lstm_layer=LSTM(256)(input_layer)
bp=BatchNormalization()(lstm_layer)
dense=Dense(128)(bp)
pred=Dense(19,activation='softmax')(dense)
model=Model(input_layer,pred)
model.summary()
adam=Adam(lr=0.001)
checkpoint=ModelCheckpoint('models/best_model.pkl',monitor='val_accuracy',save_best_only=True,verbose=1)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100)
model.compile(loss="binary_crossentropy",optimizer=adam,metrics=['accuracy'])


model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    callbacks=[checkpoint,early_stop],
    validation_split=0.2
)