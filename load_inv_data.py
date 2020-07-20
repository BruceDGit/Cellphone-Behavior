import numpy as np
import pandas as pd
from scipy.signal import resample
from tensorflow.keras.preprocessing import sequence
from tqdm import tqdm
from  augment_data import jitter

data_path = 'data/'
train = pd.read_csv(data_path + 'sensor_train.csv')
test = pd.read_csv(data_path + 'sensor_test.csv')
train = train.sort_values(by=['fragment_id', 'time_point'], ascending=[True, False])
test = test.sort_values(by=['fragment_id', 'time_point'], ascending=[True, False])

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
use_fea = [fea for fea in train.columns if fea not in no_fea]
print("use_fea", use_fea)
num_cols = len(use_fea)


# for col in use_fea:
#     min_max_scaler = MinMaxScaler()
#     data[[col]] = min_max_scaler.fit(data[[col]])
#     train[[col]]=min_max_scaler.transform(train[[col]])
#     test[[col]]=min_max_scaler.transform(test[[col]])


def load_lstm_inv_data():
    y_train = train.groupby('fragment_id')['behavior_id'].min()
    select_index = np.in1d(y_train, [13, 9, 17, 0, 10, 7, 14])
    # =============训练集=================
    train_sequences = list()

    for index, group in train.groupby(by='fragment_id'):
        train_sequences.append(group[use_fea].values)

    # 找到序列的最大长度
    len_sequences = []
    for one_seq in train_sequences:
        len_sequences.append(len(one_seq))
    print(pd.Series(len_sequences).describe())  # 最长的序列有61个

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
    with_noise = True
    if with_noise:
        # 对类别较少的数据进行数据增强
        noise_SNR_db = [5, 15]
        print("添加随机噪声,SNR_db:{}".format(noise_SNR_db))
        train_noise = jitter(train_final_seq, [5, 15])
        train_final_seq = np.concatenate([train_final_seq, train_noise[select_index]], axis=0)
        y_train = np.concatenate([y_train, y_train[select_index]], axis=0)
    print("train_final_seq.shape", train_final_seq.shape)

    seq_len = 60
    train_final_seq = sequence.pad_sequences(train_final_seq, maxlen=seq_len, padding='post',
                                             dtype='float', truncating='post')
    print("train_final_seq.shape", train_final_seq.shape)

    # =============测试集=================
    test_sequences = list()
    for index, group in test.groupby(by='fragment_id'):
        test_sequences.append(group[use_fea].values)

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


def load_cnn_inv_data():
    x = np.zeros((7292, 60, num_cols, 1))
    t = np.zeros((7500, 60, num_cols, 1))
    for i in tqdm(range(7292)):
        tmp = train[train.fragment_id == i][:60]
        x[i, :, :, 0] = resample(tmp[use_fea], 60, np.array(tmp.time_point))[0]
    for i in tqdm(range(7500)):
        tmp = test[test.fragment_id == i][:60]
        t[i, :, :, 0] = resample(tmp[use_fea], 60, np.array(tmp.time_point))[0]
    return x, y_train, t, x.shape[1], x.shape[2], x.shape[3]
