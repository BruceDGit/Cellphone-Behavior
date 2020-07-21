# -*- coding: utf-8 -*-
"""
@author: quincyqiang
@software: PyCharm
@file: MLSTM-FCN.py
@time: 2020/7/20 23:34
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical
from keras.callbacks import *
from keras.models import Model
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout

from layer_utils import AttentionLSTM

from load_data import load_lstm_data, load_features_data,load_y
from load_inv_data import load_lstm_inv_data
from utils import acc_combo

train_lstm, y1, test_lstm, seq_len, _ = load_lstm_data()
train_lstm_inv, _, test_lstm_inv, _, _ = load_lstm_inv_data()

train_features, _, test_features = load_features_data(feature_id=2)
y = load_y()
sub = pd.read_csv('data/提交结果示例.csv')


def generate_model():
    input_forward = Input(shape=(60, train_lstm.shape[2]))
    input_backward = Input(shape=(60, train_lstm.shape[2]))

    X_forward = Masking()(input_forward)
    X_forward = LSTM(8)(X_forward)
    X_forward = Dropout(0.8)(X_forward)

    X_backward = Masking()(input_backward)
    X_backward = LSTM(8)(X_backward)
    X_backward = Dropout(0.8)(X_backward)

    # y = Permute((2, 1))(ip)
    y_forward = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(input_forward)
    y_forward = BatchNormalization()(y_forward)
    y_forward = Activation('relu')(y_forward)
    y_forward = squeeze_excite_block(y_forward)

    y_forward = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y_forward)
    y_forward = BatchNormalization()(y_forward)
    y_forward = Activation('relu')(y_forward)
    y_forward = squeeze_excite_block(y_forward)

    y_forward = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y_forward)
    y_forward = BatchNormalization()(y_forward)
    y_forward = Activation('relu')(y_forward)
    y_forward = GlobalAveragePooling1D()(y_forward)

    y_backward = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(input_backward)
    y_backward = BatchNormalization()(y_backward)
    y_backward = Activation('relu')(y_backward)
    y_backward = squeeze_excite_block(y_backward)

    y_backward = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y_backward)
    y_backward = BatchNormalization()(y_backward)
    y_backward = Activation('relu')(y_backward)
    y_backward = squeeze_excite_block(y_backward)

    y_backward = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y_backward)
    y_backward = BatchNormalization()(y_backward)
    y_backward = Activation('relu')(y_backward)

    y_backward = GlobalAveragePooling1D()(y_backward)

    feainput = Input(shape=(train_features.shape[1],))
    dense = Dense(32, activation='relu')(feainput)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.2)(dense)
    dense = Dense(64, activation='relu')(dense)
    dense = Dropout(0.2)(dense)
    dense = Dense(128, activation='relu')(dense)
    dense = Dropout(0.2)(dense)
    dense = Dense(256, activation='relu')(dense)
    dense = BatchNormalization()(dense)

    output = concatenate([X_forward, X_backward, y_forward, y_backward, dense])
    output = BatchNormalization()(Dropout(0.2)(Dense(128, activation='relu')(output)))
    output = Dense(19, activation='softmax')(output)

    model = Model([input_forward, input_backward, feainput], output)

    return model


def generate_model_2():
    ip = Input(shape=(seq_len, fea_size), name="input_layer")
    # stride = 10

    # x = Permute((2, 1))(ip)
    # x = Conv1D(MAX_NB_VARIABLES // stride, 8, strides=stride, padding='same', activation='relu', use_bias=False,
    #            kernel_initializer='he_uniform')(x)  # (None, variables / stride, timesteps)
    # x = Permute((2, 1))(x)

    # ip1 = K.reshape(ip,shape=(MAX_TIMESTEPS,MAX_NB_VARIABLES))
    # x = Permute((2, 1))(ip)
    x = Masking()(ip)
    x = AttentionLSTM(8)(x)
    x = Dropout(0.8)(x)

    # y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(ip)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(19, activation='softmax')(x)

    model = Model(ip, out)
    model.summary()

    # add load model code here to fine-tune

    return model


def generate_model_3():
    ip = Input(shape=(seq_len, fea_size), name="input_layer")

    x = Masking()(ip)
    x = LSTM(8)(x)
    x = Dropout(0.8)(x)

    # y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(ip)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    # y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    # y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(19, activation='softmax')(x)

    model = Model(ip, out)
    model.summary()

    # add load model code here to fine-tune

    return model


def generate_model_4():
    ip = Input(shape=(seq_len, fea_size), name="input_layer")
    # stride = 3
    #
    # x = Permute((2, 1))(ip)
    # x = Conv1D(MAX_NB_VARIABLES // stride, 8, strides=stride, padding='same', activation='relu', use_bias=False,
    #            kernel_initializer='he_uniform')(x)  # (None, variables / stride, timesteps)
    # x = Permute((2, 1))(x)

    x = Masking()(ip)
    x = AttentionLSTM(8)(x)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    # y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    # y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    feainput = Input(shape=(train_features.shape[1],))
    dense = Dense(32, activation='relu')(feainput)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.2)(dense)
    dense = Dense(64, activation='relu')(dense)
    dense = Dropout(0.2)(dense)
    dense = Dense(128, activation='relu')(dense)
    dense = Dropout(0.2)(dense)
    dense = Dense(256, activation='relu')(dense)
    dense = BatchNormalization()(dense)

    x = concatenate([x, y])

    out = Dense(19, activation='softmax')(x)

    model = Model(ip, out)
    model.summary()

    # add load model code here to fine-tune

    return model


def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    filters = input._keras_shape[-1]  # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se


acc_scores = []
combo_scores = []
proba_t = np.zeros((7500, 19))
kfold = StratifiedKFold(5, shuffle=True)

# 类别权重设置
class_weight = np.array([0.03304992, 0.09270433, 0.05608886, 0.04552935, 0.05965442,
                         0.04703785, 0.10175535, 0.03236423, 0.0449808, 0.0393582,
                         0.03236423, 0.06157433, 0.10065826, 0.03990675, 0.01727921,
                         0.06555129, 0.04731212, 0.03551838, 0.04731212])

for fold, (train_index, valid_index) in enumerate(kfold.split(train_lstm, y)):
    print("{}train {}th fold{}".format('==' * 20, fold + 1, '==' * 20))
    y_ = to_categorical(y, num_classes=19)
    model = generate_model()
    model.compile(
        loss='categorical_crossentropy',
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

    csv_logger = CSVLogger('logs/log.csv', separator=',', append=True)
    history = model.fit([train_lstm[train_index],
                         train_lstm_inv[train_index],
                         train_features[train_index]],
                        y_[train_index],
                        epochs=500,
                        batch_size=256,
                        verbose=1,
                        shuffle=True,
                        class_weight=(1 - class_weight) ** 3,
                        validation_data=([train_lstm[valid_index],
                                          train_lstm_inv[valid_index],
                                          train_features[valid_index]],
                                         y_[valid_index]),
                        callbacks=[plateau, early_stopping, checkpoint, csv_logger])
    model.load_weights(f'models/fold{fold}.h5')
    proba_x = model.predict([train_lstm[valid_index], train_lstm_inv[valid_index], train_features[valid_index]],
                            verbose=0, batch_size=1024)
    proba_t += model.predict([test_lstm, test_lstm_inv, test_features], verbose=0, batch_size=1024) / 5.

    oof_y = np.argmax(proba_x, axis=1)
    score1 = accuracy_score(y[valid_index], oof_y)
    # print('accuracy_score',score1)
    score = sum(acc_combo(y_true, y_pred) for y_true, y_pred in zip(y[valid_index], oof_y)) / oof_y.shape[0]
    print('accuracy_score', score1, 'acc_combo', score)
    acc_scores.append(score1)
    combo_scores.append(score)

    # print(history.history.keys())
    # # summarize history for accuracy
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

print("acc_scores:", acc_scores)
print("combo_scores:", combo_scores)
print("5kflod mean acc score:{}".format(np.mean(acc_scores)))
print("5kflod mean combo score:{}".format(np.mean(combo_scores)))
sub.behavior_id = np.argmax(proba_t, axis=1)
sub.to_csv('result/mlstm_fcn_acc{}_combo{}.csv'.format(np.mean(acc_scores), np.mean(combo_scores)), index=False)
pd.DataFrame(proba_t, columns=['pred_{}'.format(i) for i in range(19)]).to_csv(
    'result/mlstm_fcn_proba_t_{}.csv'.format(np.mean(acc_scores)), index=False)
