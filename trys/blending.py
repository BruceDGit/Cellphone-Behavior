# -*- coding: utf-8 -*-
"""
@author: quincyqiang
@software: PyCharm
@file: blending.py
@time: 2020/7/26 11:15
@description：
"""
import pandas as pd
import numpy as np

resnet_oof_test = pd.read_csv('result/resnet_proba_t_0.798683635276431.csv').values  # 0.76
lstmfcn_oof_test = pd.read_csv('result/lstm_fcn_proba_t_0.8104754463803026.csv').values  # 0.76
har_oof_test = pd.read_csv('result/har_proba_t_0.8385912706807283.csv').values  # 78
fcn_test = pd.read_csv('result/fcn_proba_t_0.7628942348283348.csv').values  #
d1cnn = pd.read_csv('result/submit_1ddcnn_777(1).csv').values  #
d2cnn = pd.read_csv('result/submit_2d_9979_8742.csv').iloc[:,1:].values  #

finalpred = (
        d2cnn * 0.5 +
        # resnet_oof_test * 0.1 +
        lstmfcn_oof_test * 0.1 +
        har_oof_test * 0.4
        # fcn_test * 0.1
)
sub = pd.read_csv('../data/提交结果示例.csv')
sub.behavior_id = np.argmax(finalpred, axis=1)
sub.to_csv('result/blending.csv', index=False)
