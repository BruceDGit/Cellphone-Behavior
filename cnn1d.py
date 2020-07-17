import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical

from load_data import load_lstm_data
from utils import acc_combo

X_train, y_train, test_lstm, seq_len, _ = load_lstm_data()

kfold = StratifiedKFold(5, shuffle=True)


def Net():
    input = Input(X_train.shape[1:])

    model = Conv1D(32, 3, activation='relu', )(input)
    model = BatchNormalization()(model)
    model = MaxPooling1D(pool_size=2)(model)

    model = Conv1D(64, 3, activation='relu')(model)
    model = BatchNormalization()(model)
    model = MaxPooling1D(pool_size=2)(model)

    model = Conv1D(128, 3, activation='relu')(model)
    model = BatchNormalization()(model)
    model = MaxPooling1D(pool_size=2)(model)

    model = Conv1D(256, 3, activation='relu')(model)
    model = BatchNormalization()(model)
    model = MaxPooling1D(pool_size=2)(model)

    model = Flatten()(model)
    model = Dense(512, activation='relu')(model)
    output = Dense(19, activation='softmax')(model)
    return Model([input], output)


acc_scores = []
combo_scores = []
proba_t = np.zeros((7500, 19))
for fold, (xx, yy) in enumerate(kfold.split(X_train, y_train)):
    print("{}train {}th fold{}".format('==' * 20, fold + 1, '==' * 20))
    y_ = to_categorical(y_train, num_classes=19)
    model = Net()
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    plateau = ReduceLROnPlateau(monitor="val_acc",
                                verbose=1,
                                mode='max',
                                factor=0.5,
                                patience=15)
    early_stopping = EarlyStopping(monitor='val_acc',
                                   verbose=1,
                                   mode='max',
                                   patience=20)
    checkpoint = ModelCheckpoint(f'models/fold{fold}.h5',
                                 monitor='val_acc',
                                 verbose=0,
                                 mode='max',
                                 save_best_only=True)
    model.fit(X_train[xx], y_[xx],
              epochs=500,
              batch_size=256,
              verbose=2,
              shuffle=True,
              validation_data=(X_train[yy], y_[yy]),
              callbacks=[plateau, early_stopping, checkpoint])
    model.load_weights(f'models/fold{fold}.h5')

    proba_x = model.predict(X_train[yy], verbose=0, batch_size=1024)
    proba_t += model.predict(test_lstm, verbose=0, batch_size=1024) / 5.

    oof_y = np.argmax(proba_x, axis=1)
    score1 = accuracy_score(y_train[yy], oof_y)
    # print('accuracy_score',score1)
    score = sum(acc_combo(y_true, y_pred) for y_true, y_pred in zip(y_train[yy], oof_y)) / oof_y.shape[0]
    print('accuracy_score', score1, 'acc_combo', score)
    acc_scores.append(score1)
    combo_scores.append(score)
print("5kflod mean acc score:{}".format(np.mean(acc_scores)))
print("5kflod mean combo score:{}".format(np.mean(combo_scores)))
sub = pd.read_csv('data/提交结果示例.csv')
sub.behavior_id = np.argmax(proba_t, axis=1)
sub.to_csv('result/cnn_acc{}_combo{}.csv'.format(np.mean(acc_scores), np.mean(combo_scores)), index=False)
