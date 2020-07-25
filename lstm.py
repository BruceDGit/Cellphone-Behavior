import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.layers import *
from custom_layer import ASPP
from load_data import load_lstm_data
from utils import acc_combo

X, y, X_test, seq_len, fea_size = load_lstm_data()
sub = pd.read_csv('data/提交结果示例.csv')


def LSTM_FCN():
    input = Input(shape=(seq_len, fea_size), name="input_layer")
    x = LSTM(64)(input)
    x = Dropout(0.8)(x)

    # y = Permute((2, 1))(input)
    y = Conv1D(512, 8, padding='same', kernel_initializer='he_uniform')(input)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 6, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(512, 4, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = ASPP(256, 3, activation=tf.nn.relu)(y)

    y = GlobalAveragePooling1D()(y)
    x = concatenate([x, y])

    pred = Dense(19, activation='softmax')(x)
    model = Model([input], pred)
    return model


def LSTM_FCN_v2():
    input = Input(shape=(seq_len, fea_size), name="input_layer")
    x = LSTM(64)(input)
    x = Dropout(0.8)(x)

    # y = Permute((2, 1))(input)
    y = Conv1D(512, 8, padding='same', kernel_initializer='he_uniform')(input)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 6, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(512, 4, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = ASPP(256, 3, activation=tf.nn.relu)(y)

    y = GlobalAveragePooling1D()(y)
    x = concatenate([x, y])

    pred = Dense(19, activation='softmax')(x)
    model = Model([input], pred)
    return model


def base_boock(input):
    x = LSTM(64)(input)
    x = Dropout(0.8)(x)

    # y = Permute((2, 1))(input)
    y = Conv1D(512, 8, padding='same', kernel_initializer='he_uniform')(input)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 6, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(512, 4, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = ASPP(256, 3, activation=tf.nn.relu)(y)

    y = GlobalAveragePooling1D()(y)
    x = concatenate([x, y])
    return x


def LSTM_FCN_v3():
    input_forward = Input(shape=(60, X.shape[2]))
    input_backward = Input(shape=(60, X.shape[2]))
    lstm_forward = base_boock(input_forward)
    lstm_backward = base_boock(input_backward)
    output = Concatenate(axis=-1)([lstm_forward, lstm_backward])
    output = BatchNormalization()(Dropout(0.2)(Dense(256, activation='relu')(Flatten()(output))))
    pred = Dense(19, activation='softmax')(output)
    model = Model([input_forward, input_backward], pred)
    return model


acc_scores = []
combo_scores = []
final_x = np.zeros((7292, 19))
proba_t = np.zeros((7500, 19))
kfold = StratifiedKFold(5, shuffle=True)

# 类别权重设置
class_weight = np.array([0.03304992, 0.09270433, 0.05608886, 0.04552935, 0.05965442,
                         0.04703785, 0.10175535, 0.03236423, 0.0449808, 0.0393582,
                         0.03236423, 0.06157433, 0.10065826, 0.03990675, 0.01727921,
                         0.06555129, 0.04731212, 0.03551838, 0.04731212])

for fold, (xx, yy) in enumerate(kfold.split(X, y)):
    print("{}train {}th fold{}".format('==' * 20, fold + 1, '==' * 20))
    y_ = to_categorical(y, num_classes=19)
    model = LSTM_FCN()
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    model.summary()
    plateau = ReduceLROnPlateau(monitor="val_acc",
                                verbose=1,
                                mode='max',
                                factor=0.5,
                                patience=15)
    early_stopping = EarlyStopping(monitor='val_acc',
                                   verbose=1,
                                   mode='max',
                                   patience=50)
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
              class_weight=dict(enumerate((1 - class_weight) ** 3)),
              validation_data=(X[yy], y_[yy]),
              callbacks=[plateau, early_stopping, checkpoint, csv_logger])
    model.load_weights(f'models/fold{fold}.h5')
    proba_x = model.predict(X[yy], verbose=0, batch_size=1024)
    proba_t += model.predict(X_test, verbose=0, batch_size=1024) / 5.
    final_x[yy] += proba_x
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
sub.to_csv('result/lstm_acc{}_combo{}.csv'.format(np.mean(acc_scores), np.mean(combo_scores)), index=False)
pd.DataFrame(proba_t, columns=['pred_{}'.format(i) for i in range(19)]).to_csv(
    'result/lstm_fcn_proba_t_{}.csv'.format(np.mean(acc_scores)), index=False)
pd.DataFrame(final_x, columns=['pred_{}'.format(i) for i in range(19)]).to_csv(
    'result/lstm_fcn_proba_x_{}.csv'.format(np.mean(acc_scores)), index=False)
