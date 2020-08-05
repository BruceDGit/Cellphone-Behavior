# -*- coding: utf-8 -*-
"""
@author: quincyqiang
@software: PyCharm
@file: demo.py
@time: 2020/7/25 23:16
@description：https://www.jianshu.com/p/63c9ef510464
"""
import gc
from sklearn.metrics import *
from sklearn.model_selection import *
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from load_data import *
from load_inv_data import *
from utils import *

import tensorflow as tf
import numpy as np
from tensorflow import keras


def data_augmentation(feature, label, ws):
    seq_len = feature.shape[1]
    aug_feature, aug_label = feature[:, :ws], label
    for i in range(1, seq_len - ws + 1):
        _feature = feature[:, i:i + ws]
        aug_feature = np.concatenate((aug_feature, _feature), axis=0)
        aug_label = np.concatenate((aug_label, label), axis=0)
    return aug_feature, aug_label


def down_sampling(data, rates):
    """
    使用一组降采样因子 k1, k2, k3，每隔 ki-1 个数据取一个。
    :param data:
    :param rates: 频率 rates[2,3,4,5]
    :return:[(7292, 30,8),(7292, 20,8),(7292, 15,8), (7292, 12,8)]
    """
    ds_seq_len = []
    ds_data = []
    # down sampling by rate k
    for k in rates:
        if k > data.shape[1] / 3:
            break

        _data = data[:, ::k]  # temp after down sampling
        ds_data.append(_data)
        ds_seq_len.append(_data.shape[1])  # remark the length info
    return ds_data, ds_seq_len


def moving_average(data, moving_ws):
    """
    使用一组滑动窗口l1, l2, l3，每li个数据取平均。
    :param data:
    :param moving_ws: 滑动窗口列表 [5,8,11]
    :return:[(7292, 56,8), (7292, 53,8), (7292, 50,8)]
    """
    num, seq_len, feature_num = data.shape[0], data.shape[1], data.shape[2]  # num数据个数 seq_len序列长度，也是特征个数
    ma_data = []
    ma_seq_len = []
    for ws in moving_ws:
        if ws > data.shape[1] / 3:
            break
        _data = np.zeros((num, seq_len - ws + 1, feature_num))
        for i in range(seq_len - ws + 1):
            _data[:, i, :] = np.mean(data[:, i: i + ws, :], axis=1)
        ma_data.append(_data)
        ma_seq_len.append(_data.shape[1])

    return ma_data, ma_seq_len


def get_mcnn_input(feature):
    """
    构建多输入
    :param feature:
    :param label:
    :return:
    """
    origin = feature  # 原始特征
    ms_branch, ms_lens = down_sampling(feature, rates=[2, 3, 4, 5])  # 降采样
    mf_branch, mf_lens = moving_average(feature, moving_ws=[5, 10, 20, 30, 40, 50])  # 移动平均
    features = [origin, *ms_branch, *mf_branch]  # 长度为8：1+ 4+3=8
    features = [data.reshape(data.shape + (1,)) for data in features]  # 变成(390,seq_len,1),eg:(390, 176, 1)
    # data_lens = [origin.shape[1], *ms_lens, *mf_lens]
    return features


conv_size = 3
pooling_factor = 10


def MCNN_model(feature_lens, class_num):
    input_sigs = [Input(shape=(bra_len, 8, 1)) for bra_len in feature_lens]
    # local convolution
    ms_sigs = []
    for i in range(len(input_sigs)):
        _ms = Conv2D(filters=128,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same')(input_sigs[i])

        # _ms = Dropout(0.5)(_ms)
        ms_sigs.append(_ms)
    merged = concatenate(ms_sigs, axis=1)

    # fully convolution
    conved = Conv2D(padding='valid', kernel_size=conv_size, filters=256, activation='relu')(merged)
    conved = BatchNormalization()(conved)
    conved = Conv2D(filters=128,
                    kernel_size=(3, 3),
                    activation='relu',
                    padding='same')(conved)
    conved = BatchNormalization()(conved)

    conved = MaxPooling2D()(conved)
    conved = Dropout(0.2)(conved)
    conved = Conv2D(filters=256,
                    kernel_size=(3, 3),
                    activation='relu',
                    padding='same')(conved)
    # conved= BatchNormalization()(conved)

    conved = Dropout(0.3)(conved)
    conved = Conv2D(filters=512,
                    kernel_size=(3, 3),
                    activation='relu',
                    padding='same')(conved)
    pooled = GlobalAveragePooling2D()(conved)

    conved = Flatten()(pooled)
    x = Dense(256, activation='relu')(conved)
    x = Dense(class_num, activation='softmax')(x)
    MCNN = Model(inputs=input_sigs, outputs=x)
    return MCNN


def BLOCK(seq, filters, kernal_size):
    cnn = keras.layers.Conv1D(filters, 1, padding='SAME', activation='relu')(seq)
    cnn = keras.layers.LayerNormalization()(cnn)

    cnn = keras.layers.Conv1D(filters, kernal_size, padding='SAME', activation='relu')(cnn)
    cnn = keras.layers.LayerNormalization()(cnn)

    cnn = keras.layers.Conv1D(filters, 1, padding='SAME', activation='relu')(cnn)
    cnn = keras.layers.LayerNormalization()(cnn)

    seq = keras.layers.Conv1D(filters, 1)(seq)
    seq = keras.layers.Add()([seq, cnn])
    return seq


def BLOCK2(seq, filters=128, kernal_size=5):
    seq = BLOCK(seq, filters, kernal_size)
    seq = keras.layers.MaxPooling1D(2)(seq)
    seq = keras.layers.SpatialDropout1D(0.3)(seq)
    seq = BLOCK(seq, filters // 2, kernal_size)
    seq = keras.layers.GlobalAveragePooling1D()(seq)
    return seq


def ComplexConv1D(inputs):
    seq_3 = BLOCK2(inputs, kernal_size=3)
    seq_5 = BLOCK2(inputs, kernal_size=5)
    seq_7 = BLOCK2(inputs, kernal_size=7)
    seq = keras.layers.concatenate([seq_3, seq_5, seq_7])
    seq = keras.layers.Dense(512, activation='relu')(seq)
    seq = keras.layers.Dropout(0.3)(seq)
    seq = keras.layers.Dense(128, activation='relu')(seq)
    seq = keras.layers.Dropout(0.3)(seq)
    return seq


def MCNN_model_v2(feature_lens, class_num):
    input_sigs = [Input(shape=(bra_len, 8)) for bra_len in feature_lens]
    ms_sigs = []
    for i in range(len(input_sigs)):
        _ms = ComplexConv1D(input_sigs[i])
        _ms = Dropout(0.2)(_ms)
        ms_sigs.append(_ms)
    merged = concatenate(ms_sigs, axis=1)

    x = Flatten()(merged)
    x = Dense(512, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)

    x = Dense(class_num, activation='softmax')(x)
    MCNN = Model(inputs=input_sigs, outputs=x)
    return MCNN


acc_scores = []
combo_scores = []
final_x = np.zeros((7292, 19))
proba_t = np.zeros((7500, 19))
kfold = StratifiedKFold(5, shuffle=True, random_state=42)

# 类别权重设置
class_weight = np.array([0.03304992, 0.09270433, 0.05608886, 0.04552935, 0.05965442,
                         0.04703785, 0.10175535, 0.03236423, 0.0449808, 0.0393582,
                         0.03236423, 0.06157433, 0.10065826, 0.03990675, 0.01727921,
                         0.06555129, 0.04731212, 0.03551838, 0.04731212])
f_train, la_tr, f_test, _, _ = load_lstm_data()
# la_tr = to_categorical(la_tr)
# f_train, f_test = normalization(f_train, f_test)
f_trains = get_mcnn_input(f_train)
f_tests = get_mcnn_input(f_test)
feat_lens = [data.shape[1] for data in f_trains]  # get the length info of each transform sequence
print(feat_lens)
for fold, (train_index, valid_index) in enumerate(kfold.split(f_train, la_tr)):
    print("{}train {}th fold{}".format('==' * 20, fold + 1, '==' * 20))
    y_ = to_categorical(la_tr, num_classes=19)
    model = MCNN_model_v2(feat_lens, y_.shape[1])

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
    checkpoint = ModelCheckpoint(f'models/fold{fold}_mlp.h5',
                                 monitor='val_acc',
                                 verbose=0,
                                 mode='max',
                                 save_best_only=True)

    csv_logger = CSVLogger('logs/log.csv', separator=',', append=True)
    history = model.fit([f_train[train_index] for f_train in f_trains],
                        y_[train_index],
                        epochs=500,
                        batch_size=128,
                        verbose=1,
                        shuffle=True,
                        class_weight=dict(enumerate((1 - class_weight) ** 3)),
                        validation_data=([f_train[valid_index] for f_train in f_trains],
                                         y_[valid_index]),
                        callbacks=[plateau, early_stopping, checkpoint, csv_logger])
    model.load_weights(f'models/fold{fold}_mlp.h5')
    proba_x = model.predict([f_train[valid_index] for f_train in f_trains],
                            verbose=0, batch_size=128)
    proba_t += model.predict(f_tests, verbose=0, batch_size=128) / 5.
    final_x[valid_index] += proba_x

    oof_y = np.argmax(proba_x, axis=1)
    score1 = accuracy_score(la_tr[valid_index], oof_y)
    # print('accuracy_score',score1)
    score = sum(acc_combo(y_true, y_pred) for y_true, y_pred in zip(la_tr[valid_index], oof_y)) / oof_y.shape[0]
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
    gc.collect()

print("5kflod mean acc score:{}".format(np.mean(acc_scores)))
print("5kflod mean combo score:{}".format(np.mean(combo_scores)))

sub = pd.read_csv('data/提交结果示例.csv')
sub.behavior_id = np.argmax(proba_t, axis=1)
sub.to_csv('result/mlp_acc{}_combo{}.csv'.format(np.mean(acc_scores), np.mean(combo_scores)), index=False)
pd.DataFrame(proba_t, columns=['pred_{}'.format(i) for i in range(19)]).to_csv(
    'result/mlp_proba_t_{}.csv'.format(np.mean(acc_scores)), index=False)
pd.DataFrame(final_x, columns=['pred_{}'.format(i) for i in range(19)]).to_csv(
    'result/mlp_proba_x_{}.csv'.format(np.mean(acc_scores)), index=False)
