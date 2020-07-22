import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from scipy.signal import resample
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import mode
import os

data_path = 'data/'
train = pd.read_csv(data_path + 'sensor_train.csv')
test = pd.read_csv(data_path + 'sensor_test.csv')
y_train = train.groupby('fragment_id')['behavior_id'].min()

train_size = len(train)
# y_train = train.drop_duplicates(subset=['fragment_id']).reset_index(drop=True)['behavior_id'].values
# y_train = to_categorical(y_train)
print("y_train.shape:", y_train.shape)

data = pd.concat([train, test], sort=False)
data['acc'] = (data.acc_x ** 2 + data.acc_y ** 2 + data.acc_z ** 2) ** .5
data['accg'] = (data.acc_xg ** 2 + data.acc_yg ** 2 + data.acc_zg ** 2) ** .5
#
# data['acc1'] = (data['acc_x'] ** 2 + data['acc_y'] ** 2) ** 0.5
# data['accg1'] = (data['acc_xg'] ** 2 + data['acc_yg'] ** 2) ** 0.5
#
# data['acc2'] = (data['acc_x'] ** 2 + data['acc_z'] ** 2) ** 0.5
# data['accg2'] = (data['acc_xg'] ** 2 + data['acc_zg'] ** 2) ** 0.5
#
# #     data['acc3'] = (data['acc_y'] ** 2 + data['acc_z'] ** 2) ** 0.5
# #     data['accg3'] = (data['acc_yg'] ** 2 + data['acc_zg'] ** 2) ** 0.5  # y - z系列 under 4%%
#
#
# data['acc_sub'] = ((data['acc_xg'] - data['acc_x']) ** 2 + (data['acc_yg'] - data['acc_y']) ** 2 + (
#         data['acc_zg'] - data['acc_z']) ** 2) ** 0.5
# data['acc_sub1'] = ((data['acc_xg'] - data['acc_x']) ** 2 + (data['acc_yg'] - data['acc_y']) ** 2) ** 0.5
# data['acc_sub2'] = ((data['acc_xg'] - data['acc_x']) ** 2 + (data['acc_zg'] - data['acc_z']) ** 2) ** 0.5
# #     data['acc_sub3'] = ((data['acc_yg'] - data['acc_y']) ** 2 + (data['acc_zg'] - data['acc_z'])**2) ** 0.5
#
#
# data['accxg_diff_accx'] = data['acc_xg'] - data['acc_x']
# data['accyg_diff_accy'] = data['acc_yg'] - data['acc_y']
# data['acczg_diff_accz'] = data['acc_zg'] - data['acc_z']

# abs

train, test = data[:train_size], data[train_size:]

no_fea = ['fragment_id', 'behavior_id', 'time_point', 'inv_fragment_id', 'inv_behavior_id', 'inv_time_point']
use_fea = [fea for fea in data.columns if fea not in no_fea]
print("use_fea", use_fea)
num_cols = len(use_fea)


# for col in use_fea:
#     print(col)
#     min_max_scaler = MinMaxScaler()
#     data[[col]] = min_max_scaler.fit(data[[col]])
#     train[[col]]=min_max_scaler.transform(train[[col]])
#     test[[col]]=min_max_scaler.transform(test[[col]])


def load_lstm_data():
    # =============训练集=================
    train_sequences = list()

    for index, group in train.groupby(by='fragment_id'):
        train_sequences.append(group[use_fea].values)

    # 找到序列的最大长度
    len_sequences = []
    for one_seq in train_sequences:
        len_sequences.append(len(one_seq))

    # 填充序列
    to_pad = 61
    train_new_seq = []
    for one_seq in train_sequences:
        len_one_seq = len(one_seq)
        last_val = one_seq[-1]
        n = to_pad - len_one_seq
        # to_concat = np.repeat(last_val, n).reshape(len(use_fea), n).transpose()
        # new_one_seq = np.concatenate([one_seq, to_concat])
        if n != 0:
            to_concat = one_seq[:n]
            new_one_seq = np.concatenate([one_seq, to_concat])
        else:
            new_one_seq = one_seq
        train_new_seq.append(new_one_seq)

    train_final_seq = np.stack(train_new_seq)
    # final_seq.shape (314, 129, 4)
    print("train_final_seq.shape", train_final_seq.shape)
    # 进行截断

    seq_len = 60
    train_final_seq = sequence.pad_sequences(train_final_seq, maxlen=seq_len, padding='post',
                                             dtype='float', truncating='post')
    print("train_final_seq.shape", train_final_seq.shape)

    # =============测试集=================
    test_sequences = list()
    for index, group in test.groupby(by='fragment_id'):
        test_sequences.append(group[use_fea].values)

    for one_seq in test_sequences:
        len_sequences.append(len(one_seq))
    print(pd.Series(len_sequences).describe())  # 最长的序列有61个

    # 填充到最大长度
    to_pad = 61
    test_new_seq = []
    for one_seq in test_sequences:
        len_one_seq = len(one_seq)
        last_val = one_seq[-1]
        n = to_pad - len_one_seq
        # to_concat = np.repeat(last_val, n).reshape(len(use_fea), n).transpose()
        # new_one_seq = np.concatenate([one_seq, to_concat])
        if n != 0:
            to_concat = one_seq[:n]
            new_one_seq = np.concatenate([one_seq, to_concat])
        else:
            new_one_seq = one_seq

        test_new_seq.append(new_one_seq)

    test_final_seq = np.stack(test_new_seq)
    print("test_final_seq.shape", test_final_seq.shape)

    # 进行截断

    seq_len = 60
    test_final_seq = sequence.pad_sequences(test_final_seq, maxlen=seq_len, padding='post',
                                            dtype='float', truncating='post')
    print("test_final_seq.shape", test_final_seq.shape)
    return train_final_seq, y_train, test_final_seq, seq_len, len(use_fea)


def load_cnn_data():
    x = np.zeros((7292, 60, num_cols, 1))
    t = np.zeros((7500, 60, num_cols, 1))
    for i in tqdm(range(7292)):
        tmp = train[train.fragment_id == i][:60]
        x[i, :, :, 0] = resample(tmp[use_fea], 60, np.array(tmp.time_point))[0]
    for i in tqdm(range(7500)):
        tmp = test[test.fragment_id == i][:60]
        t[i, :, :, 0] = resample(tmp[use_fea], 60, np.array(tmp.time_point))[0]
    return x, y_train, t, x.shape[1], x.shape[2], x.shape[3]


def load_features_data(feature_id=1):
    if feature_id == 1:
        if not os.path.exists('data/df_train_test_features.csv'):
            data_path = 'data/'
            df_train = pd.read_csv(data_path + 'sensor_train.csv')
            df_test = pd.read_csv(data_path + 'sensor_test.csv')
            df_train['flag'] = 'train'
            df_test['flag'] = 'test'
            df_test['behavior_id'] = -1
            df_train_test = pd.concat([df_train, df_test])
            df_train_test['acc_all'] = (df_train_test['acc_x'] ** 2 + df_train_test['acc_y'] ** 2 + df_train_test[
                'acc_z'] ** 2) ** 0.5
            df_train_test['acc_allg'] = (df_train_test['acc_xg'] ** 2 + df_train_test['acc_yg'] ** 2 + df_train_test[
                'acc_zg'] ** 2) ** 0.5

            agg_func = lambda x: list(x)
            map_agg_func = {
                'time_point': agg_func,

                'acc_all': agg_func,
                'acc_allg': agg_func,

                'acc_x': agg_func,
                'acc_y': agg_func,
                'acc_z': agg_func,

                'acc_xg': agg_func,
                'acc_yg': agg_func,
                'acc_zg': agg_func
            }
            df_train_test_list = df_train_test.groupby(['flag', 'fragment_id', 'behavior_id']).agg(
                map_agg_func).reset_index()
            map_features_fun = {
                # 时域
                'time_sum': lambda x: np.sum(x),
                'time_mean': lambda x: np.mean(x),
                'time_std': lambda x: np.std(x),
                'time_var': lambda x: np.var(x),
                'time_max': lambda x: np.max(x),
                'time_min': lambda x: np.min(x),
                'time_median': lambda x: np.median(x),
                'time_energy': lambda x: np.sum(np.power(x, 2)),
                'time_mad': lambda x: np.mean(np.absolute(x - np.mean(x))),
                'time_percent_9': lambda x: np.percentile(x, 0.9),
                'time_percent_75': lambda x: np.percentile(x, 0.75),
                'time_percent_25': lambda x: np.percentile(x, 0.25),
                'time_percent_1': lambda x: np.percentile(x, 0.1),
                'time_percent_75_25': lambda x: np.percentile(x, 75) - np.percentile(x, 25),
                'time_range': lambda x: np.max(x) - np.min(x),
                'time_zcr': lambda x: (np.diff(np.sign(x)) != 0).sum(),
                'time_mcr': lambda x: (np.diff(np.sign(x - np.mean(x))) != 0).sum(),
                'time_minind': lambda x: np.argmin(x),
                'time_maxind': lambda x: np.argmax(x),
                'time_skew': lambda x: skew(x),
                'time_kurtosis': lambda x: kurtosis(x),
                'time_zero_big': lambda x: np.sum(np.sign(x) > 0),
                'time_zero_small': lambda x: np.sum(np.sign(x) < 0),
                'time_len': lambda x: np.size(x),

                # 频域
                'fft_dc': lambda x: np.abs(np.fft.fft(x))[0],
                'fft_mean': lambda x: np.mean(np.abs(np.fft.fft(x))[1:int(len(x) / 2) + 1]),
                'fft_var': lambda x: np.var(np.abs(np.fft.fft(x))[1:int(len(x) / 2) + 1]),
                'fft_std': lambda x: np.std(np.abs(np.fft.fft(x))[1:int(len(x) / 2) + 1]),
                'fft_sum': lambda x: np.sum(np.abs(np.fft.fft(x))[1:int(len(x) / 2) + 1]),
                'fft_entropy': lambda x: -1.0 * np.sum(np.log2(
                    np.abs(np.fft.fft(x))[1:int(len(x) / 2) + 1] / np.sum(
                        np.abs(np.fft.fft(x))[1:int(len(x) / 2) + 1]))),
                'fft_energy': lambda x: np.sum(np.power(np.abs(np.fft.fft(x))[1:int(len(x) / 2) + 1], 2)),
                'fft_skew': lambda x: skew(np.abs(np.fft.fft(x))[1:int(len(x) / 2) + 1]),
                'fft_kurtosis': lambda x: kurtosis(np.abs(np.fft.fft(x))[1:int(len(x) / 2) + 1]),
                'fft_max': lambda x: np.max(np.abs(np.fft.fft(x))[1:int(len(x) / 2) + 1]),
                'fft_min': lambda x: np.min(np.abs(np.fft.fft(x))[1:int(len(x) / 2) + 1]),
                'fft_maxind': lambda x: np.argmax(np.abs(np.fft.fft(x))[1:int(len(x) / 2) + 1]),
                'fft_minind': lambda x: np.argmin(np.abs(np.fft.fft(x))[1:int(len(x) / 2) + 1])
            }
            df_train_test_features = df_train_test_list[['flag', 'fragment_id', 'behavior_id']]
            for col in ['acc_all', 'acc_allg', 'acc_x', 'acc_y', 'acc_z', 'acc_xg', 'acc_yg', 'acc_zg']:
                for f_name, f_fun in tqdm(map_features_fun.items()):
                    df_train_test_features[col + '_' + f_name] = df_train_test_list[col].map(f_fun)
            df_train_test_features.to_csv('data/df_train_test_features.csv', index=None)
        else:
            df_train_test_features = pd.read_csv('data/df_train_test_features.csv')

        cols = [c for c in df_train_test_features.columns if c not in ['flag', 'fragment_id', 'behavior_id']]

        for col in cols:
            try:
                min_max_scaler = MinMaxScaler()
                df_train_test_features[[col]] = min_max_scaler.fit_transform(df_train_test_features[[col]])
            except Exception as e:
                print(e, col)
                cols.remove(col)
        X_train = df_train_test_features[df_train_test_features['flag'] == 'train'][cols].values
        y_train = df_train_test_features[df_train_test_features['flag'] == 'train']['behavior_id'].values
        X_test = df_train_test_features[df_train_test_features['flag'] == 'test'][cols].values

        return X_train, y_train, X_test
    elif feature_id == 2:
        data_path = 'data/'
        data_train = pd.read_csv(data_path + 'sensor_train.csv')
        data_test = pd.read_csv(data_path + 'sensor_test.csv')
        data_test['fragment_id'] += 10000
        label = 'behavior_id'

        data = pd.concat([data_train, data_test], sort=False)
        df = data.drop_duplicates(subset=['fragment_id']).reset_index(drop=True)[['fragment_id', 'behavior_id']]

        # data['acc'] = (data['acc_x'] ** 2 + data['acc_y'] ** 2 + data['acc_z'] ** 2) ** 0.5
        # data['accg'] = (data['acc_xg'] ** 2 + data['acc_yg'] ** 2 + data['acc_zg'] ** 2) ** 0.5
        # for f in tqdm([f for f in data.columns if 'acc' in f]):
        #     for stat in ['min', 'mean', 'median', 'std']:
        #         df[f + '_' + stat] = data.groupby('fragment_id')[f].agg(stat).values

        data['acc'] = (data['acc_x'] ** 2 + data['acc_y'] ** 2 + data['acc_z'] ** 2) ** 0.5
        data['accg'] = (data['acc_xg'] ** 2 + data['acc_yg'] ** 2 + data['acc_zg'] ** 2) ** 0.5

        data['accxy'] = (data['acc_x'] ** 2 + data['acc_y'] ** 2) ** 0.5
        data['accyz'] = (data['acc_y'] ** 2 + data['acc_z'] ** 2) ** 0.5
        data['accxz'] = (data['acc_x'] ** 2 + data['acc_z'] ** 2) ** 0.5
        data['accxyg'] = (data['acc_xg'] ** 2 + data['acc_yg'] ** 2) ** 0.5
        data['accyzg'] = (data['acc_yg'] ** 2 + data['acc_zg'] ** 2) ** 0.5
        data['accxzg'] = (data['acc_xg'] ** 2 + data['acc_zg'] ** 2) ** 0.5

        #     data['acc_sub'] = ((data['acc_xg'] - data['acc_x']) ** 2 + (data['acc_yg'] - data['acc_y']) ** 2 + (data['acc_zg'] - data['acc_z']) ** 2) ** 0.5
        #     data['acc_subxy'] = ((data['acc_xg'] - data['acc_x']) ** 2 + (data['acc_yg'] - data['acc_y']) ** 2) ** 0.5
        #     data['acc_subxz'] = ((data['acc_xg'] - data['acc_x']) ** 2 + (data['acc_zg'] - data['acc_z']) ** 2) ** 0.5
        #     data['acc_subyz'] = ((data['acc_yg'] - data['acc_y']) ** 2 + (data['acc_zg'] - data['acc_z']) ** 2) ** 0.5

        # 统计特征
        for f in tqdm([f for f in data.columns if 'acc' in f]):
            for stat in ['min', 'mean', 'median', 'std']:  # skew
                df['{}_{}'.format(f, stat)] = data.groupby('fragment_id')[f].agg(stat).values

        train_df = df[df[label].isna() == False].reset_index(drop=True)
        test_df = df[df[label].isna() == True].reset_index(drop=True)

        drop_feat = []
        used_feat = [f for f in train_df.columns if f not in (['fragment_id', label] + drop_feat)]
        print(len(used_feat))
        print(used_feat)

        train_x = train_df[used_feat].values
        train_y = train_df[label].values
        test_x = test_df[used_feat].values

        return train_x, train_y, test_x


def load_y():
    return y_train


train_lstm, y1, test_lstm, seq_len, _ = load_lstm_data()