# -*- coding: utf-8 -*-
"""
@author: quincyqiang
@software: PyCharm
@file: demo.py
@time: 2020/7/25 23:16
@description：https://www.jianshu.com/p/63c9ef510464
"""
import numpy as np
import os
from keras.utils import np_utils
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import gc


def get_datasets_path(base_path):
    datasets_used = ['Adiac',
                     # 'Beef', 'CBF', 'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee', 'Cricket_X',
                     # 'Cricket_Y',
                     # 'Cricket_Z', 'DiatomSizeReduction', 'ECGFiveDays', 'FaceAll', 'FaceFour', 'FacesUCR', '50words',
                     # 'FISH', 'Gun_Point', 'Haptics', 'InlineSkate', 'ItalyPowerDemand', 'Lighting2', 'Lighting7',
                     # 'MALLAT',
                     # 'MedicalImages', 'MoteStrain', 'NonInvasiveFatalECG_Thorax1', 'NonInvasiveFatalECG_Thorax2',
                     # 'OliveOil',
                     # 'OSULeaf', 'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'StarLightCurves', 'SwedishLeaf',
                     # 'Symbols',
                     # 'synthetic_control', 'Trace', 'TwoLeadECG', 'Two_Patterns', 'UWaveGestureLibrary_X',
                     # 'UWaveGestureLibrary_Y', 'UWaveGestureLibrary_Z', 'wafer', 'WordSynonyms', 'yoga'
                     ]
    data_list = os.listdir(base_path)
    return ["%s/%s/%s_" % (base_path, data, data) for data in datasets_used if data in data_list]


def data_augmentation(feature, label, ws):
    seq_len = feature.shape[1]
    aug_feature, aug_label = feature[:, :ws], label
    for i in range(1, seq_len - ws + 1):
        _feature = feature[:, i:i + ws]
        aug_feature = np.concatenate((aug_feature, _feature), axis=0)
        aug_label = np.concatenate((aug_label, label), axis=0)
    return aug_feature, aug_label


def load_feature_label(path, aug_times=0):
    data = np.loadtxt(path, dtype=np.float, delimiter=',')  # (390, 177) 390条数据 177列，其中第一列为标签，其余为特征
    # the first column is label, and the rest are features
    feature, label = data[:, 1:], data[:, 0]
    if aug_times > 0:
        feature, label = data_augmentation(feature, label, data.shape[1] - aug_times)
    return feature, label


def down_sampling(data, rates):
    """
    使用一组降采样因子 k1, k2, k3，每隔 ki-1 个数据取一个。
    :param data:
    :param rates: 频率 rates[2,3,4,5]
    :return:[(390, 88),(390, 59),(390, 44), (390, 36)]
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
    :return:[(390, 172), (390, 169), (390, 166)]
    """
    num, seq_len = data.shape[0], data.shape[1]  # num数据个数 seq_len序列长度，也是特征个数
    ma_data = []
    ma_seq_len = []
    for ws in moving_ws:
        if ws > data.shape[1] / 3:
            break
        _data = np.zeros((num, seq_len - ws + 1))
        for i in range(seq_len - ws + 1):
            _data[:, i] = np.mean(data[:, i: i + ws], axis=1)
        ma_data.append(_data)
        ma_seq_len.append(_data.shape[1])

    return ma_data, ma_seq_len


def get_mcnn_input(feature, label):
    """
    构建多输入
    :param feature:
    :param label:
    :return:
    """
    origin = feature  # 原始特征
    ms_branch, ms_lens = down_sampling(feature, rates=[2, 3, 4, 5])  # 降采样
    mf_branch, mf_lens = moving_average(feature, moving_ws=[5, 8, 11])  # 移动平均

    label = np_utils.to_categorical(label)  # one hot
    features = [origin, *ms_branch, *mf_branch]  # 长度为8：1+ 4+3=8
    features = [data.reshape(data.shape + (1,)) for data in features] # 变成(390,seq_len,1),eg:(390, 176, 1)
    data_lens = [origin.shape[1], *ms_lens, *mf_lens]
    return features, label


conv_size = 6
pooling_factor = 2


def MCNN_model(feature_lens, class_num):
    input_sigs = [Input(shape=(bra_len, 1)) for bra_len in feature_lens]
    # local convolution
    ms_sigs = []
    for i in range(len(input_sigs)):
        _ms = Conv1D(padding='same', kernel_size=conv_size, filters=256, activation='relu')(input_sigs[i])
        pooling_size = (_ms.shape[1] - conv_size + 1) // pooling_factor
        _ms = MaxPooling1D(pool_size=pooling_size)(_ms)
        ms_sigs.append(_ms)
    merged = concatenate(ms_sigs, axis=1)

    # fully convolution
    conved = Conv1D(padding='valid', kernel_size=conv_size, filters=256, activation='relu')(merged)
    pooled = MaxPooling1D(pool_size=5)(conved)

    x = Flatten()(pooled)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(class_num, activation='softmax')(x)
    MCNN = Model(inputs=input_sigs, outputs=x)
    #     MCNN.summary()
    MCNN.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return MCNN


def on_single_dataset(data_path):
    data_name = data_path.split('/')[-2]  # Adiac
    # get the origin time series
    f_train, la_tr = load_feature_label(data_path + 'TRAIN', aug_times=0)  # 加载训练集
    f_test, la_te = load_feature_label(data_path + 'TEST', aug_times=0)  # 加载测试集
    # f_train, f_test = normalization(f_train, f_test)
    # get the transform sequences
    f_trains, la_tr = get_mcnn_input(f_train, la_tr)
    f_tests, la_te = get_mcnn_input(f_test, la_te)

    feat_lens = [data.shape[1] for data in f_trains]  # get the length info of each transform sequence

    class_num = la_tr.shape[1]
    mcnn = MCNN_model(feat_lens, class_num)
    mcnn.fit(f_trains, la_tr, batch_size=128, epochs=2, verbose=1)
    te_loss, te_acc = mcnn.evaluate(f_tests, la_te, verbose=0)
    print("[%(data)s]: Test loss - %(loss).2f, Test accuracy - %(acc).2f%%" % {'data': data_name, 'loss': te_loss,
                                                                               'acc': te_acc * 100})

    la_pred = mcnn.predict(f_tests)
    for data in f_trains, f_tests, la_te, la_tr:
        del data
    gc.collect()


def main():
    base_path = './UCR_TS_Archive_2015'
    dataPaths = get_datasets_path(base_path)
    for dp in dataPaths:
        print(dp)  # ./UCR_TS_Archive_2015/Adiac/Adiac_
        on_single_dataset(dp)


if __name__ == '__main__':
    main()
