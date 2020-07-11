#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: har.py 
@time: 2020-07-12 02:12
@description:
"""

from argparse import ArgumentParser
from pathlib import Path
from sklearn.model_selection import *
import matplotlib.pyplot as plt
from sklearn.metrics import *
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint,CSVLogger,ReduceLROnPlateau
from tensorflow.keras.layers import InputLayer, Conv1D, BatchNormalization, ReLU, Dense, Flatten, Softmax, LSTM
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from load_data import *
from utils import *

train_lstm, y1, test_lstm, seq_len, _ = load_lstm_data()
train_cnn, y2, test_cnn, dim_height, dim_width, dim_channel = load_cnn_data()
train_features, y3, test_features = load_features_data()
y = load_y()


x_train, x_test, y_train, y_test = train_lstm/20, test_lstm/20, y, _

def Net(model_type='ModelB'):
    if model_type == 'ModelA':
        model = Sequential([
            InputLayer(input_shape=(60, x_train.shape[2],)),
            Conv1D(16, 3),
            BatchNormalization(),
            ReLU(),
            Conv1D(32, 3),
            BatchNormalization(),
            ReLU(),
            Flatten(),
            Dense(19),
            Softmax()
        ], 'WISDM-ModelA')
    else:
        model = Sequential([
            InputLayer(input_shape=(60, x_train.shape[2],)),
            Conv1D(16, 3),
            BatchNormalization(),
            ReLU(),
            Conv1D(32, 3),
            BatchNormalization(),
            ReLU(),
            LSTM(128),
            Dense(19),
            Softmax()
        ], 'WISDM-ModelB')

    return  model
acc_scores = []
combo_scores = []
proba_t = np.zeros((7500, 19))
kfold = StratifiedKFold(5, shuffle=True, random_state=42)

for fold, (train_index, valid_index) in enumerate(kfold.split(train_lstm, y)):
    print("{}train {}th fold{}".format('==' * 20, fold + 1, '==' * 20))
    y_ = to_categorical(y, num_classes=19)
    model = Net(model_type='ModelB')
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    model.summary()
    plateau = ReduceLROnPlateau(monitor="val_acc",
                                verbose=1,
                                mode='max',
                                factor=0.5,
                                patience=20)
    early_stopping = EarlyStopping(monitor='val_acc',
                                   verbose=1,
                                   mode='max',
                                   patience=40)
    checkpoint = ModelCheckpoint(f'models/fold{fold}.h5',
                                 monitor='val_acc',
                                 verbose=0,
                                 mode='max',
                                 save_best_only=True)

    csv_logger = CSVLogger('logs/log.csv', separator=',', append=True)
    history=model.fit(train_lstm[train_index], y_[train_index],
              epochs=500,
              batch_size=64,
              verbose=2,
              shuffle=True,
              validation_data=(train_lstm[valid_index],
                               y_[valid_index]),
              callbacks=[plateau, early_stopping, checkpoint, csv_logger])
    model.load_weights(f'models/fold{fold}.h5')
    proba_x = model.predict(train_lstm[valid_index], verbose=0, batch_size=1024)
    proba_t += model.predict(train_lstm, verbose=0, batch_size=1024) / 5.

    oof_y = np.argmax(proba_x, axis=1)
    score1 = accuracy_score(y[valid_index], oof_y)
    # print('accuracy_score',score1)
    score = sum(acc_combo(y_true, y_pred) for y_true, y_pred in zip(y[valid_index], oof_y)) / oof_y.shape[0]
    print('accuracy_score', score1, 'acc_combo', score)
    acc_scores.append(score1)
    combo_scores.append(score)

    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

print("5kflod mean acc score:{}".format(np.mean(acc_scores)))
print("5kflod mean combo score:{}".format(np.mean(combo_scores)))

sub = pd.read_csv('data/提交结果示例.csv')
sub.behavior_id = np.argmax(proba_t, axis=1)
sub.to_csv('result/dnn_acc{}_combo{}.csv'.format(np.mean(acc_scores), np.mean(combo_scores)), index=False)

