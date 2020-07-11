import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.layers import *

from load_data import load_lstm_data
from utils import acc_combo

X, y, X_test, seq_len, fea_size = load_lstm_data()
sub = pd.read_csv('data/提交结果示例.csv')


def Net():
    input = Input(shape=(seq_len, fea_size), name="input_layer")
    model = Bidirectional(GRU(128, return_sequences=True))(input)
    model = Bidirectional(GRU(256, kernel_regularizer=tf.keras.regularizers.l2(0.001)))(model)

    model = BatchNormalization()(model)
    model = Dropout(0.2)(model)
    model = Flatten()(model)
    model = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(model)
    model = Dropout(0.2)(model)
    # model = BatchNormalization(model)

    pred = Dense(19, activation='softmax')(model)
    model = Model([input], pred)
    model.summary()
    return model


acc_scores = []
combo_scores = []
proba_t = np.zeros((7500, 19))
kfold = StratifiedKFold(5, shuffle=True)

for fold, (xx, yy) in enumerate(kfold.split(X, y)):
    print("{}train {}th fold{}".format('==' * 20, fold + 1, '==' * 20))
    y_ = to_categorical(y, num_classes=19)
    model = Net()
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    model.summary()
    plateau = ReduceLROnPlateau(monitor="val_acc",
                                verbose=1,
                                mode='max',
                                factor=0.5,
                                patience=10)
    early_stopping = EarlyStopping(monitor='val_acc',
                                   verbose=1,
                                   mode='max',
                                   patience=20)
    checkpoint = ModelCheckpoint(f'models/fold{fold}.h5',
                                 monitor='val_acc',
                                 verbose=0,
                                 mode='max',
                                 save_best_only=True)

    csv_logger = CSVLogger('logs/log.csv', separator=',', append=True)
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
print("5kflod mean acc score:{}".format(np.mean(acc_scores)))
print("5kflod mean combo score:{}".format(np.mean(combo_scores)))
sub.behavior_id = np.argmax(proba_t, axis=1)
sub.to_csv('result/lstm_acc{}_combo{}.csv'.format(np.mean(acc_scores), np.mean(combo_scores)), index=False)
