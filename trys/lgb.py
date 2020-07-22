# -*- coding: utf-8 -*-

"""
@author: quincyqiang
@software: PyCharm
@file: lgb.py
@time: 2020/7/22 21:50
"""

import time
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from create_features import get_feat
from sklearn.metrics import accuracy_score
from utils import acc_combo


def train_model_classification(X, X_test, y, params, num_classes=2,
                               folds=None, model_type='lgb',
                               eval_metric='logloss', columns=None,
                               plot_feature_importance=False,
                               model=None, verbose=10000,
                               early_stopping_rounds=200,
                               splits=None, n_folds=3):
    """
    分类模型函数
    返回字典，包括： oof predictions, test predictions, scores and, if necessary, feature importances.
    :params: X - 训练数据， pd.DataFrame
    :params: X_test - 测试数据，pd.DataFrame
    :params: y - 目标
    :params: folds - folds to split data
    :params: model_type - 模型
    :params: eval_metric - 评价指标
    :params: columns - 特征列
    :params: plot_feature_importance - 是否展示特征重要性
    :params: model - sklearn model, works only for "sklearn" model type
    """

    global y_pred_valid, y_pred

    # columns = X.columns if columns is None else columns
    # X_test = X_test[columns]
    splits = folds.split(X,y) if splits is None else splits
    n_splits = folds.n_splits if splits is None else n_folds

    # to set up scoring parameters
    metrics_dict = {
        'logloss': {
            'lgb_metric_name': 'logloss',
            'xgb_metric_name': 'logloss',
            'catboost_metric_name': 'Logloss',
            'sklearn_scoring_function': metrics.log_loss
        },
        'lb_score_method': {
            'sklearn_scoring_f1': metrics.f1_score,  # 线上评价指标
            'sklearn_scoring_accuracy': metrics.accuracy_score,  # 线上评价指标
            'sklearn_scoring_auc': metrics.roc_auc_score
        },
    }
    result_dict = {}
    # out-of-fold predictions on train data
    oof = np.zeros(shape=(len(X), num_classes))
    # averaged predictions on train data
    prediction = np.zeros(shape=(len(X_test), num_classes))
    # list of scores on folds
    acc_scores = []
    combo_scores = []
    feature_importance = pd.DataFrame()
    # split and train on folds
    for fold_n, (train_index, valid_index) in enumerate(splits):
        if verbose:
            print(f'Fold {fold_n + 1} started at {time.ctime()}')
        if type(X) == np.ndarray:
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        if model_type == 'lgb':
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_valid, y_valid)],
                      eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                      verbose=verbose,
                      early_stopping_rounds=early_stopping_rounds)

            y_pred_valid = model.predict_proba(X_valid)
            y_pred = model.predict_proba(X_test, num_iteration=model.best_iteration_)

        if model_type == 'xgb':
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_valid, y_valid)],
                      eval_metric=metrics_dict[eval_metric]['xgb_metric_name'],
                      verbose=bool(verbose),  # xgb verbose bool
                      early_stopping_rounds=early_stopping_rounds)
            y_pred_valid = model.predict_proba(X_train[valid_index])
            y_pred = model.predict_proba(X_test, ntree_limit=model.best_ntree_limit)
        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)
            y_pred_valid = model.predict_proba(X_valid)
            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)
            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')
            print('')
            y_pred = model.predict_proba(X_test)

        if model_type == 'cat':
            model = CatBoostClassifier(iterations=20000, eval_metric=metrics_dict[eval_metric]['catboost_metric_name'],
                                       **params,
                                       loss_function=metrics_dict[eval_metric]['catboost_metric_name'])
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True,
                      verbose=False)

            y_pred_valid = model.predict_proba(X_valid)
            y_pred = model.predict_proba(X_test)

        oof[valid_index] = y_pred_valid
        # 评价指标
        y_pred_valid = np.argmax(y_pred_valid, axis=1)

        acc_scores.append(accuracy_score(y_valid, y_pred_valid))
        score = sum(acc_combo(y_true, y_pred) for y_true, y_pred in zip(y_valid, y_pred_valid)) / y_pred_valid.shape[0]
        combo_scores.append(score)
        prediction += y_pred

        if model_type == 'lgb' and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

        if model_type == 'xgb' and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
    prediction /= n_splits
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(acc_scores), np.std(acc_scores)))

    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['acc_scores'] = acc_scores
    result_dict['combo_scores'] = combo_scores

    if model_type == 'lgb' or model_type == 'xgb':
        if plot_feature_importance:
            feature_importance["importance"] /= n_splits
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12))
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
            plt.title('LGB Features (avg over folds)')

            result_dict['feature_importance'] = feature_importance

    return result_dict


lgb_params = {
    'learning_rate': 0.1,
    'metric': 'multi_error',
    'objective': 'multiclass',
    'num_class': 19,
    'feature_fraction': 0.80,
    'bagging_fraction': 0.75,
    'bagging_freq': 2,
    'n_jobs': 6,
    'seed': 2020,
    'max_depth': 10,
    'num_leaves': 64,
    'lambda_l1': 0.5,
    'lambda_l2': 0.5,
}

n_fold = 5
folds = StratifiedKFold(n_splits=n_fold, random_state=1314)

train_features, y, test_features = get_feat()

num_classes = 19

result_dict_lgb = train_model_classification(X=train_features,
                                             X_test=test_features,
                                             y=y,
                                             params=lgb_params,
                                             num_classes=num_classes,
                                             folds=folds,
                                             model_type='lgb',
                                             eval_metric='logloss',
                                             plot_feature_importance=False,
                                             verbose=100,
                                             early_stopping_rounds=200)

proba_t = np.argmax(result_dict_lgb['prediction'], axis=1)
acc_scores = result_dict_lgb['acc_scores']
combo_scores = result_dict_lgb['combo_scores']
print("5kflod mean acc score:{}".format(np.mean(acc_scores)))
print("5kflod mean combo score:{}".format(np.mean(combo_scores)))

sub = pd.read_csv('data/提交结果示例.csv')
sub.behavior_id = np.argmax(proba_t, axis=1)
sub.to_csv('result/dnn_acc{}_combo{}.csv'.format(np.mean(acc_scores), np.mean(combo_scores)), index=False)

pd.DataFrame(proba_t, columns=['pred_{}'.format(i) for i in range(19)]).to_csv(
    'result/dnn_proba_test_{}.csv'.format(np.mean(acc_scores)), index=False)
pd.DataFrame(result_dict_lgb['oof'], columns=['pred_{}'.format(i) for i in range(19)]).to_csv(
    'result/dnn_proba_train_{}.csv'.format(np.mean(acc_scores)), index=False)
