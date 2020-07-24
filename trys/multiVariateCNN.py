# -*- coding: utf-8 -*-
"""
@author: quincyqiang
@software: PyCharm
@file: multiVariateCNN.py
@time: 2020/7/24 23:36
@description：https://github.com/salmansust/TimeSeries-CNN/blob/master/TimeSeries-CNN/multiVariateCNN.py
"""
from numpy import array
from numpy import hstack
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.layers import *

from load_data import load_lstm_data
from utils import acc_combo

X, y, X_test, seq_len, fea_size = load_lstm_data()
sub = pd.read_csv('../data/提交结果示例.csv')

n_steps = 60
n_features = 1


def MultiVariateCNN():
    # acc_x
    visible1 = Input(shape=(n_steps, n_features))
    cnn1 = Conv1D(filters=64, kernel_size=2, activation='relu')(visible1)
    cnn1 = MaxPooling1D(pool_size=2)(cnn1)
    cnn1=BatchNormalization()(cnn1)
    cnn1 = Flatten()(cnn1)
    # acc_y
    visible2 = Input(shape=(n_steps, n_features))
    cnn2 = Conv1D(filters=64, kernel_size=2, activation='relu')(visible2)
    cnn2 = MaxPooling1D(pool_size=2)(cnn2)
    cnn2=BatchNormalization()(cnn2)
    cnn2 = Flatten()(cnn2)
    # acc_z
    visible3 = Input(shape=(n_steps, n_features))
    cnn3 = Conv1D(filters=64, kernel_size=2, activation='relu')(visible3)
    cnn3 = MaxPooling1D(pool_size=2)(cnn3)
    cnn3=BatchNormalization()(cnn3)
    cnn3 = Flatten()(cnn3)

    # acc_xg
    visible4 = Input(shape=(n_steps, n_features))
    cnn4 = Conv1D(filters=64, kernel_size=2, activation='relu')(visible4)
    cnn4 = MaxPooling1D(pool_size=2)(cnn4)
    cnn4=BatchNormalization()(cnn4)
    cnn4 = Flatten()(cnn4)

    # acc_yg
    visible5 = Input(shape=(n_steps, n_features))
    cnn5 = Conv1D(filters=64, kernel_size=2, activation='relu')(visible5)
    cnn5 = MaxPooling1D(pool_size=2)(cnn5)
    cnn5=BatchNormalization()(cnn5)
    cnn5 = Flatten()(cnn5)
    # acc_zg
    visible6 = Input(shape=(n_steps, n_features))
    cnn6 = Conv1D(filters=64, kernel_size=2, activation='relu')(visible6)
    cnn6 = MaxPooling1D(pool_size=2)(cnn6)
    cnn6=BatchNormalization()(cnn6)
    cnn6 = Flatten()(cnn6)

    # merge input models
    merge = concatenate([cnn1, cnn2, cnn3, cnn4, cnn5, cnn6])
    output = Dense(256, activation='relu')(merge)
    output = Dropout(0.1)(output)
    output = Dense(19, activation='softmax')(output)
    model = Model(inputs=[visible1, visible2, visible3,
                          visible4, visible5, visible6], outputs=output)

    return model


acc_scores = []
combo_scores = []
proba_t = np.zeros((7500, 19))
kfold = StratifiedKFold(5, shuffle=True)

for fold, (xx, yy) in enumerate(kfold.split(X, y)):
    print("{}train {}th fold{}".format('==' * 20, fold + 1, '==' * 20))
    y_ = to_categorical(y, num_classes=19)
    model = MultiVariateCNN()
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
                                   patience=30)
    checkpoint = ModelCheckpoint(f'models/fold{fold}.h5',
                                 monitor='val_acc',
                                 verbose=0,
                                 mode='max',
                                 save_best_only=True)

    csv_logger = CSVLogger('../logs/log.csv', separator=',', append=True)
    model.fit([X[xx][:, :, i].reshape(X[xx].shape[0], X[xx].shape[1], n_features) for i in range(X.shape[2])], y_[xx],
              epochs=500,
              batch_size=256,
              verbose=2,
              shuffle=True,
              validation_data=([X[yy][:, :, i].reshape(X[yy].shape[0], X[yy].shape[1], n_features) for i in range(X.shape[2])], y_[yy]),
              callbacks=[plateau, early_stopping, checkpoint, csv_logger])
    model.load_weights(f'models/fold{fold}.h5')
    proba_x = model.predict([X[yy][:, :, i].reshape(X[yy].shape[0], X[yy].shape[1], n_features) for i in range(X.shape[2])], verbose=0, batch_size=1024)
    proba_t += model.predict([X_test[:, :, i].reshape(X_test.shape[0], X_test.shape[1], n_features) for i in range(X.shape[2])], verbose=0, batch_size=1024) / 5.

    oof_y = np.argmax(proba_x, axis=1)
    score1 = accuracy_score(y[yy], oof_y)
    # print('accuracy_score',score1)
    score = sum(acc_combo(y_true, y_pred) for y_true, y_pred in zip(y[yy], oof_y)) / oof_y.shape[0]
    print('accuracy_score', score1, 'acc_combo', score)
    acc_scores.append(score1)
    combo_scores.append(score)

print("acc_scores:", acc_scores)
print("combo_scores:", combo_scores)
print("5kflod mean acc score:{}".format(np.mean(acc_scores)))
print("5kflod mean combo score:{}".format(np.mean(combo_scores)))
sub.behavior_id = np.argmax(proba_t, axis=1)
sub.to_csv('result/MultiVariateCNN_acc{}_combo{}.csv'.format(np.mean(acc_scores), np.mean(combo_scores)), index=False)
pd.DataFrame(proba_t, columns=['pred_{}'.format(i) for i in range(19)]).to_csv(
    'result/MultiVariateCNN_proba_t_{}.csv'.format(np.mean(acc_scores)), index=False)
