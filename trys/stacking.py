# -*- coding: utf-8 -*-
"""
@author: quincyqiang
@software: PyCharm
@file: stacking.py
@time: 2020/7/26 10:43
@description：
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from utils import *
from load_data import *
y_train = load_y()

resnet_oof_train = pd.read_csv('result/resnet_proba_x_0.798683635276431.csv').values
lstmfcn_oof_train = pd.read_csv('result/lstm_fcn_proba_x_0.8104754463803026.csv').values
har_oof_train = pd.read_csv('result/har_proba_x_0.8385912706807283.csv').values
fcn_train = pd.read_csv('result/fcn_proba_x_0.7628942348283348.csv').values

resnet_oof_test = pd.read_csv('result/resnet_proba_t_0.798683635276431.csv').values
lstmfcn_oof_test = pd.read_csv('result/lstm_fcn_proba_t_0.8104754463803026.csv').values
har_oof_test = pd.read_csv('result/har_proba_t_0.8385912706807283.csv').values
fcn_test = pd.read_csv('result/fcn_proba_t_0.7628942348283348.csv').values

x_train = np.concatenate((resnet_oof_train, lstmfcn_oof_train, har_oof_train,fcn_train), axis=1)
x_test = np.concatenate((resnet_oof_test, lstmfcn_oof_test, har_oof_test,fcn_test), axis=1)

print("{},{}".format(x_train.shape, x_test.shape))

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test)

xgb_params = {
    'objective': 'multi:softprob',
    'eta': 0.1,
    'max_depth': 2,
    'num_class': 19,
    'eval_metric': "mlogloss",
    'min_child_weight': 1,
    'subsample': 0.7,
    'colsample_bytree': 0.7
}
SEED=20200726
res = xgb.cv(xgb_params, dtrain, num_boost_round=50, nfold=5, seed=SEED,
             early_stopping_rounds=10, show_stdv=True)

best_nrounds = res.shape[0] - 1
cv_mean = res.iloc[-1, 0]
cv_std = res.iloc[-1, 1]

print('Ensemble-CV: {0}+{1}'.format(cv_mean, cv_std))
gbdt = xgb.train(xgb_params, dtrain, best_nrounds)

out_df = pd.DataFrame(gbdt.predict(dtest))
sub = pd.read_csv('../data/提交结果示例.csv')
sub.behavior_id = np.argmax(out_df.values, axis=1)
sub.to_csv('result/stacking.csv', index=False)

# 0.7792380952380953