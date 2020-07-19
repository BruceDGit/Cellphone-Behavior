import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.model_selection import *
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.layers import *
from disout_tf2 import *
from utils import *
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical

data_path = '../data/'
train = pd.read_csv(data_path + 'sensor_train.csv')
test = pd.read_csv(data_path + 'sensor_test.csv')
y_train = train.groupby('fragment_id')['behavior_id'].min()
train_size = len(train)

data = pd.concat([train, test], sort=False)
num_cats = 1  # number of categorical features
n_steps = 60  # number of timesteps in each sample
n_numerical_feats = 6  # number of numerical features in each sample
cat_size = []  # number of categories in each categorical feature
cat_embd_dim = [50]  # embedding dimension for each categorical feature
# https://stackoverflow.com/questions/52627739/how-to-merge-numerical-and-embedding-sequential-models-to-treat-categories-in-rn
# acc_id = pd.DataFrame(data[['acc_x', 'acc_y', 'acc_z']].values.reshape(-1, 1))
# acc_id.columns = ['values']
# acc_id['value_id'] = acc_id.groupby(['values']).ngroup()
# acc_dict = dict(zip(acc_id['values'], acc_id['value_id']))
# cat_size.append(len(acc_dict))
# for col in ['acc_x', 'acc_y', 'acc_z']:
#     data[col] = data[col].apply(lambda x: acc_dict[x])
#
# accg_id = pd.DataFrame(data[['acc_xg', 'acc_yg', 'acc_zg']].values.reshape(-1, 1))
# accg_id.columns = ['values']
# accg_id['value_id'] = accg_id.groupby(['values']).ngroup()
# accg_dict = dict(zip(accg_id['values'], accg_id['value_id']))
# cat_size.append(len(accg_dict))
# for col in ['acc_xg', 'acc_yg', 'acc_zg']:
#     data[col] = data[col].apply(lambda x: accg_dict[x])


acc_id = pd.DataFrame(data[['acc_x', 'acc_y', 'acc_z', 'acc_xg', 'acc_yg', 'acc_zg']].values.reshape(-1, 1))
acc_id.columns = ['values']
acc_id['value_id'] = acc_id.groupby(['values']).ngroup()
acc_dict = dict(zip(acc_id['values'], acc_id['value_id']))
cat_size.append(len(acc_dict))
for col in ['acc_x', 'acc_y', 'acc_z', 'acc_xg', 'acc_yg', 'acc_zg']:
    data[col] = data[col].apply(lambda x: acc_dict[x])

no_fea = ['fragment_id', 'behavior_id', 'time_point', 'inv_fragment_id', 'inv_behavior_id', 'inv_time_point']
use_fea = [fea for fea in data.columns if fea not in no_fea]
print("use_fea", use_fea)
num_cols = len(use_fea)

train, test = data[:train_size], data[train_size:]


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


train_lstm, y, test_lstm, seq_len, _ = load_lstm_data()


def emb_lstm():
    cat_inputs = []
    for i in range(num_cats):
        cat_inputs.append(Input(shape=(n_steps, 6), name='cat' + str(i + 1) + '_input'))

    # cat_embedded = []
    # for i in range(num_cats):
    embed = TimeDistributed(Embedding(cat_size[i], cat_embd_dim[i]))(cat_inputs[i])
    # cat_embedded.append(embed)

    # cat_merged = concatenate(cat_embedded)
    cat_merged = Reshape((60, 50))(embed)
    lstm_out = LSTM(64)(cat_merged)
    lstm_out = Dense(19, activation='softmax')(lstm_out)
    return Model([cat_inputs], lstm_out)


acc_scores = []
combo_scores = []
proba_t = np.zeros((7500, 19))
kfold = StratifiedKFold(5, shuffle=True, random_state=42)

# 类别权重设置
class_weight = np.array([0.03304992, 0.09270433, 0.05608886, 0.04552935, 0.05965442,
                         0.04703785, 0.10175535, 0.03236423, 0.0449808, 0.0393582,
                         0.03236423, 0.06157433, 0.10065826, 0.03990675, 0.01727921,
                         0.06555129, 0.04731212, 0.03551838, 0.04731212])
for fold, (train_index, valid_index) in enumerate(kfold.split(train_lstm, y)):
    print("{}train {}th fold{}".format('==' * 20, fold + 1, '==' * 20))
    y_ = to_categorical(y, num_classes=19)
    model = emb_lstm()
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
    checkpoint = ModelCheckpoint(f'models/fold{fold}.h5',
                                 monitor='val_acc',
                                 verbose=0,
                                 mode='max',
                                 save_best_only=True)

    csv_logger = CSVLogger('logs/log.csv', separator=',', append=True)
    history = model.fit(train_lstm[train_index],
                        y_[train_index],
                        epochs=500,
                        batch_size=256,
                        verbose=1,
                        shuffle=True,
                        class_weight=(1 - class_weight) ** 3,
                        validation_data=(train_lstm[valid_index],
                                         y_[valid_index]),
                        callbacks=[plateau, early_stopping, checkpoint, csv_logger])
    model.load_weights(f'models/fold{fold}.h5')
    proba_x = model.predict(train_lstm[valid_index],
                            verbose=0, batch_size=1024)
    proba_t += model.predict(test_lstm, verbose=0, batch_size=1024) / 5.

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

print("5kflod mean acc score:{}".format(np.mean(acc_scores)))
print("5kflod mean combo score:{}".format(np.mean(combo_scores)))

sub = pd.read_csv('../data/提交结果示例.csv')
sub.behavior_id = np.argmax(proba_t, axis=1)
sub.to_csv('result/har_acc{}_combo{}.csv'.format(np.mean(acc_scores), np.mean(combo_scores)), index=False)
pd.DataFrame(proba_t, columns=['pred_{}'.format(i) for i in range(19)]).to_csv(
    'result/har_proba_t_{}.csv'.format(np.mean(acc_scores)), index=False)
