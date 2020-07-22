import os
import warnings
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

warnings.filterwarnings('ignore')
os.environ['PYTHONHASHSEED'] = '0'


def hashfxn(astring):
    return ord(astring[0])


def tfidf(input_values, output_num, output_prefix, seed=1024):
    tfidf_enc = TfidfVectorizer()
    tfidf_vec = tfidf_enc.fit_transform(input_values)
    svd_tmp = TruncatedSVD(n_components=output_num, n_iter=20, random_state=seed)
    svd_tmp = svd_tmp.fit_transform(tfidf_vec)
    svd_tmp = pd.DataFrame(svd_tmp)
    svd_tmp.columns = ['{}_tfidf_{}'.format(output_prefix, i) for i in range(output_num)]
    return svd_tmp


def get_grad_tfidf(df, group_id, group_target, num):
    grad_df = df.groupby(group_id)['acc_x'].apply(lambda x: np.gradient(x)).reset_index()
    grad_df['acc_y'] = df.groupby(group_id)['acc_y'].apply(lambda x: np.gradient(x)).reset_index()['acc_y']
    grad_df['acc_z'] = df.groupby(group_id)['acc_z'].apply(lambda x: np.gradient(x)).reset_index()['acc_z']
    grad_df['acc_x'] = grad_df['acc_x'].apply(lambda x: np.round(x, 4))
    grad_df['acc_y'] = grad_df['acc_y'].apply(lambda x: np.round(x, 4))
    grad_df['acc_z'] = grad_df['acc_z'].apply(lambda x: np.round(x, 4))
    grad_df[group_target] = grad_df.apply(
        lambda x: ' '.join(['{}_{}'.format(z[0], z[1], z[2]) for z in zip(x['acc_x'], x['acc_y'], x['acc_z'])]), axis=1)

    tfidf_tmp = tfidf(grad_df[group_target], num, group_target)
    return pd.concat([grad_df[[group_id]], tfidf_tmp], axis=1)


def q10(x):
    return x.quantile(0.1)


def q20(x):
    return x.quantile(0.2)


def q30(x):
    return x.quantile(0.3)


def q40(x):
    return x.quantile(0.4)


def q60(x):
    return x.quantile(0.6)


def q70(x):
    return x.quantile(0.7)


def q80(x):
    return x.quantile(0.8)


def q90(x):
    return x.quantile(0.9)


def get_feat():
    if not os.path.exists('data/train_x.pkl'):
        print("get feat...")
        train = pd.read_csv('data/sensor_train.csv')
        test = pd.read_csv('data/sensor_test.csv')
        test['fragment_id'] += 10000
        data = pd.concat([train, test], axis=0).reset_index(drop=True)

        group_df = data.drop_duplicates(subset=['fragment_id']).reset_index(drop=True)[['fragment_id', 'behavior_id']]

        # 历史特征相减
        base_fea = ['acc_x', 'acc_y', 'acc_z', 'acc_xg',
                    'acc_yg', 'acc_zg', ]
        for fea in tqdm(base_fea):
            data['{}_diff'.format(fea)] = data.groupby('fragment_id')[fea].diff(1)
            data['{}_diff'.format(fea)].fillna(method='bfill', inplace=True)

        # 统计特征
        stat_functions = ['min', 'std', 'mean', 'median', 'nunique', q10, q20, q30, q40, q60, q70, q80, q90]
        stat_ways = ['min', 'std', 'mean', 'median', 'nunique', 'q_10', 'q_20', 'q_30', 'q_40', 'q_60', 'q_70', 'q_80',
                     'q_90']

        stat_cols = ['acc_x', 'acc_y', 'acc_z', 'acc_xg',
                     'acc_yg', 'acc_zg', ]
        group_tmp = data.groupby('fragment_id')[stat_cols].agg(stat_functions).reset_index()
        group_tmp.columns = ['fragment_id'] + ['{}_{}'.format(i, j) for i in stat_cols for j in stat_ways]
        group_df = group_df.merge(group_tmp, on='fragment_id', how='left')

        # tfidf特征
        grad_tfidf = get_grad_tfidf(data, 'fragment_id', 'grad', 30)
        group_df = group_df.merge(grad_tfidf, on='fragment_id', how='left')
        print('gradient tfidf finished.')

        no_fea = ['time_point', 'fragment_id', 'behavior_id']
        used_feat = [f for f in group_df.columns if f not in no_fea]
        print(len(used_feat))
        print(used_feat)

        train_df = group_df[group_df['behavior_id'].isna() == False].reset_index(drop=True)
        test_df = group_df[group_df['behavior_id'].isna() == True].reset_index(drop=True)
        train_x = train_df[used_feat].values
        train_y = train_df['behavior_id'].values
        test_x = test_df[used_feat].values

        with open('data/train_x.pkl', 'wb') as f:
            pickle.dump(train_x, f)
        with open('data/train_y.pkl', 'wb') as f:
            pickle.dump(train_y, f)
        with open('data/test_x.pkl', 'wb') as f:
            pickle.dump(test_x, f)
    else:
        with open('data/train_x.pkl', 'rb') as f:
            train_x = pickle.load(f)
        with open('data/train_y.pkl', 'rb') as f:
            train_y = pickle.load(f)
        with open('data/test_x.pkl', 'rb') as f:
            test_x = pickle.load(f)

    return train_x, train_y, test_x
