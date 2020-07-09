import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler

data_path = 'data/'
train = pd.read_csv(data_path + 'sensor_train.csv')
test = pd.read_csv(data_path + 'sensor_test.csv')
y_train = train.groupby('fragment_id')['behavior_id'].min()
# y_train = train.drop_duplicates(subset=['fragment_id']).reset_index(drop=True)['behavior_id'].values
# y_train = to_categorical(y_train)
print("y_train.shape:", y_train.shape)

data = pd.concat([train, test], sort=False)
base_fea = ['acc_x',
            'acc_y', 'acc_z', 'acc_xg', 'acc_yg', 'acc_zg']
# for col in base_fea:
#     min_max_scaler = MinMaxScaler()
#     data[[col]] = min_max_scaler.fit(data[[col]])
#     train[[col]]=min_max_scaler.transform(train[[col]])
#     test[[col]]=min_max_scaler.transform(test[[col]])

# =============训练集=================
train_sequences = list()

for index, group in train.groupby(by='fragment_id'):
    train_sequences.append(group[base_fea].values)

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
    to_concat = np.repeat(last_val, n).reshape(len(base_fea), n).transpose()
    new_one_seq = np.concatenate([one_seq, to_concat])
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
    test_sequences.append(group[base_fea].values)

# 填充到最大长度
to_pad = 61
test_new_seq = []
for one_seq in test_sequences:
    len_one_seq = len(one_seq)
    last_val = one_seq[-1]
    n = to_pad - len_one_seq
    to_concat = np.repeat(last_val, n).reshape(len(base_fea), n).transpose()
    new_one_seq = np.concatenate([one_seq, to_concat])
    test_new_seq.append(new_one_seq)

test_final_seq = np.stack(test_new_seq)
print("test_final_seq.shape", test_final_seq.shape)

# 进行截断

seq_len = 60
test_final_seq = sequence.pad_sequences(test_final_seq, maxlen=seq_len, padding='post',
                                        dtype='float', truncating='post')
print("test_final_seq.shape", test_final_seq.shape)


def load_data():
    return train_final_seq, y_train, test_final_seq, seq_len, len(base_fea)
