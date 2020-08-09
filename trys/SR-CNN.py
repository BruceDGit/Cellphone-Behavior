# -*- coding: utf-8 -*-
"""
https://github.com/abdulwaheedsoudagar/SR-CNN/blob/master/SRCNN.ipynb
@author: quincyqiang
@software: PyCharm
@file: SR-CNN.py
@time: 2020/7/23 23:48
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical

from load_data import load_lstm_data
from utils import acc_combo

X, y, X_test, seq_len, fea_size = load_lstm_data()
sub = pd.read_csv('../data/提交结果示例.csv')


def SRCNN():
    input = Input(shape=(seq_len, fea_size), name="input_layer")
    inputx = Reshape((60, X.shape[2], 1), input_shape=(60, X.shape[2]))(input)

    c1 = Conv2D(128, (3, 3), activation='relu', padding='same')(inputx)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c1)
    d1 = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(c2)

    m1 = concatenate([d1, c2], axis=3)

    d2 = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(m1)

    m2 = concatenate([c1, d2], axis=3)

    output = Conv2D(128, (3, 3), activation='sigmoid', padding='same')(m2)
    output = GlobalAveragePooling2D()(output)
    output = Dropout(0.5)(output)
    output = Dense(512, activation='relu')(output)
    output = BatchNormalization()(output)
    output = Dense(19, activation='softmax')(output)
    model = Model(inputs=input, outputs=output)
    return model


def DSRCNN():
    input = Input(shape=(seq_len, fea_size), name="input_layer")
    inputx = Reshape((60, X.shape[2], 1), input_shape=(60, X.shape[2]))(input)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputx)

    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    d1 = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(c2)

    m1 = concatenate([d1, c2], axis=3)

    d2 = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(m1)

    m2 = concatenate([c1, d2], axis=3)

    output = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(m2)
    output = Dense(512, activation='relu')(output)
    output = BatchNormalization()(output)
    output = Dense(19, activation='softmax')(output)
    model = Model(inputs=input, outputs=output)
    return model


acc_scores = []
combo_scores = []
proba_t = np.zeros((7500, 19))
kfold = StratifiedKFold(5, shuffle=True)

for fold, (xx, yy) in enumerate(kfold.split(X, y)):
    print("{}train {}th fold{}".format('==' * 20, fold + 1, '==' * 20))
    y_ = to_categorical(y, num_classes=19)
    model = SRCNN()
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
    model.fit(X[xx], y_[xx],
              epochs=500,
              batch_size=256,
              verbose=2,
              shuffle=True,
              validation_data=(X[yy], y_[yy]),
              callbacks=[plateau, early_stopping, checkpoint, csv_logger])
    model.load_weights(f'models/fold{fold}.h5')
    proba_x = model.predict(X[yy], verbose=0, batch_size=1024)
    proba_t += model.predict(X_test, verbose=0, batch_size=1024) / 5.

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
sub.to_csv('result/emn_acc{}_combo{}.csv'.format(np.mean(acc_scores), np.mean(combo_scores)), index=False)
pd.DataFrame(proba_t, columns=['pred_{}'.format(i) for i in range(19)]).to_csv(
    'result/emn_proba_t_{}.csv'.format(np.mean(acc_scores)), index=False)
