from scipy.signal import resample
import pandas as pd
import tensorflow as tf
import numpy as np
import pandas as pd
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.utils import to_categorical
from scipy.signal import resample
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.layers import *
from tqdm import tqdm
from utils import acc_combo
from tensorflow.keras.initializers import glorot_uniform
train = pd.read_csv('../data/sensor_train.csv')
test = pd.read_csv('../data/sensor_test.csv')
train_size = len(train)
data = pd.concat([train, test], sort=False)
sub = pd.read_csv('../data/提交结果示例.csv')
y = train.groupby('fragment_id')['behavior_id'].min()

data['mod'] = (data.acc_x ** 2 + data.acc_y ** 2 + data.acc_z ** 2) ** .5
data['modg'] = (data.acc_xg ** 2 + data.acc_yg ** 2 + data.acc_zg ** 2) ** .5
data['mod2'] = data.acc_x ** 2 + data.acc_y ** 2 + data.acc_z ** 2
data['modg2'] = data.acc_xg ** 2 + data.acc_yg ** 2 + data.acc_zg ** 2

#
data['mod'] = (data.acc_x ** 2 + data.acc_y ** 2 + data.acc_z ** 2) ** .5
data['modg'] = (data.acc_xg ** 2 + data.acc_yg ** 2 + data.acc_zg ** 2) ** .5
data['mod2'] = data.acc_x ** 2 + data.acc_y ** 2 + data.acc_z ** 2
data['modg2'] = data.acc_xg ** 2 + data.acc_yg ** 2 + data.acc_zg ** 2

#
data['acc1'] = (data['acc_x'] ** 2 + data['acc_y'] ** 2) ** 0.5
data['accg1'] = (data['acc_xg'] ** 2 + data['acc_yg'] ** 2) ** 0.5
data['acc2'] = (data['acc_x'] ** 2 + data['acc_z'] ** 2) ** 0.5
data['accg2'] = (data['acc_xg'] ** 2 + data['acc_zg'] ** 2) ** 0.5
data['acc3'] = (data['acc_y'] ** 2 + data['acc_z'] ** 2) ** 0.5
data['accg3'] = (data['acc_yg'] ** 2 + data['acc_zg'] ** 2) ** 0.5  # y - z系列 under 4%%

data['acc_sub'] = ((data['acc_xg'] - data['acc_x']) ** 2 + (data['acc_yg'] - data['acc_y']) ** 2 + (
        data['acc_zg'] - data['acc_z']) ** 2) ** 0.5
data['acc_sub1'] = ((data['acc_xg'] - data['acc_x']) ** 2 + (data['acc_yg'] - data['acc_y']) ** 2) ** 0.5
data['acc_sub2'] = ((data['acc_xg'] - data['acc_x']) ** 2 + (data['acc_zg'] - data['acc_z']) ** 2) ** 0.5
data['acc_sub3'] = ((data['acc_yg'] - data['acc_y']) ** 2 + (data['acc_zg'] - data['acc_z']) ** 2) ** 0.5

data['accxg_diff_accx'] = data['acc_xg'] - data['acc_x']
data['accyg_diff_accy'] = data['acc_yg'] - data['acc_y']
data['acczg_diff_accz'] = data['acc_zg'] - data['acc_z']

data['accxg_add_accx'] = data['acc_xg'] + data['acc_x']
data['accyg_add_accy'] = data['acc_yg'] + data['acc_y']
data['acczg_add_accz'] = data['acc_zg'] + data['acc_z']



# abs

train, test = data[:train_size], data[train_size:]

no_fea = ['fragment_id', 'behavior_id', 'time_point', 'inv_fragment_id', 'inv_behavior_id', 'inv_time_point']
use_fea = [fea for fea in train.columns if fea not in no_fea]
print("use_fea", use_fea)
fea_size = len(use_fea)

x = np.zeros((7292, 60, fea_size, 1))
t = np.zeros((7500, 60, fea_size, 1))
for i in tqdm(range(7292)):
    tmp = train[train.fragment_id == i][:60]
    x[i, :, :, 0] = resample(tmp[use_fea], 60, np.array(tmp.time_point))[0]
for i in tqdm(range(7500)):
    tmp = test[test.fragment_id == i][:60]
    t[i, :, :, 0] = resample(tmp[use_fea], 60, np.array(tmp.time_point))[0]

kfold = StratifiedKFold(5, shuffle=True)


filter_stride = 1
pool_size = 6

def Net():
    X_input = Input(shape=(60, fea_size, 1))

    X = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(filter_stride, 1), padding='same', depth_multiplier=1,
                        activation='relu',
                        depthwise_initializer=glorot_uniform(seed=921212),
                        pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X_input)
    X = MaxPooling2D((1, pool_size), strides=(3, 1))(X)
    X = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(filter_stride, 1), padding='same', depth_multiplier=1,
                        activation='relu',
                        depthwise_initializer=glorot_uniform(seed=921212),
                        pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X)
    X = MaxPooling2D((1, pool_size), strides=(3, 1))(X)
    X = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(filter_stride, 1), padding='same', depth_multiplier=1,
                        activation='relu',
                        depthwise_initializer=glorot_uniform(seed=921212),
                        pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X)
    X = MaxPooling2D((1, pool_size), strides=(3, 1))(X)
    X = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(filter_stride, 1), padding='same', depth_multiplier=1,
                        activation='relu',
                        depthwise_initializer=glorot_uniform(seed=921212),
                        pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X)
    X = MaxPooling2D((1, pool_size), strides=(3, 1))(X)

    X1 = SeparableConv2D(filters=64, kernel_size=(5, 1), strides=(filter_stride, 1), padding='same', depth_multiplier=1,
                         activation='relu',
                         depthwise_initializer=glorot_uniform(seed=921212),
                         pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X_input)
    X1 = MaxPooling2D((1, pool_size), strides=(3, 1))(X1)
    X1 = SeparableConv2D(filters=64, kernel_size=(5, 1), strides=(filter_stride, 1), padding='same', depth_multiplier=1,
                         activation='relu',
                         depthwise_initializer=glorot_uniform(seed=921212),
                         pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X1)
    X1 = MaxPooling2D((1, pool_size), strides=(3, 1))(X1)
    X1 = SeparableConv2D(filters=64, kernel_size=(5, 1), strides=(filter_stride, 1), padding='same', depth_multiplier=1,
                         activation='relu',
                         depthwise_initializer=glorot_uniform(seed=921212),
                         pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X1)
    X1 = MaxPooling2D((1, pool_size), strides=(3, 1))(X1)
    X1 = SeparableConv2D(filters=64, kernel_size=(5, 1), strides=(filter_stride, 1), padding='same', depth_multiplier=1,
                         activation='relu',
                         depthwise_initializer=glorot_uniform(seed=921212),
                         pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X1)
    X1 = MaxPooling2D((1, pool_size), strides=(3, 1))(X1)

    X2 = SeparableConv2D(filters=64, kernel_size=(9, 1), strides=(filter_stride, 1), padding='same', depth_multiplier=1,
                         activation='relu',
                         depthwise_initializer=glorot_uniform(seed=921212),
                         pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X_input)
    X2 = MaxPooling2D((1, pool_size), strides=(3, 1))(X2)
    X2 = SeparableConv2D(filters=64, kernel_size=(9, 1), strides=(filter_stride, 1), padding='same', depth_multiplier=1,
                         activation='relu',
                         depthwise_initializer=glorot_uniform(seed=921212),
                         pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X2)
    X2 = MaxPooling2D((1, pool_size), strides=(3, 1))(X2)
    X2 = SeparableConv2D(filters=64, kernel_size=(9, 1), strides=(filter_stride, 1), padding='same', depth_multiplier=1,
                         activation='relu',
                         depthwise_initializer=glorot_uniform(seed=921212),
                         pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X2)
    X2 = MaxPooling2D((1, pool_size), strides=(3, 1))(X2)
    X2 = SeparableConv2D(filters=64, kernel_size=(9, 1), strides=(filter_stride, 1), padding='same', depth_multiplier=1,
                         activation='relu',
                         depthwise_initializer=glorot_uniform(seed=921212),
                         pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X2)
    X2 = MaxPooling2D((1, pool_size), strides=(3, 1))(X2)

    X3 = SeparableConv2D(filters=64, kernel_size=(13, 1), strides=(filter_stride, 1), padding='same',
                         depth_multiplier=1, activation='relu',
                         depthwise_initializer=glorot_uniform(seed=921212),
                         pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X_input)
    X3 = MaxPooling2D((1, pool_size), strides=(3, 1))(X3)
    X3 = SeparableConv2D(filters=64, kernel_size=(13, 1), strides=(filter_stride, 1), padding='same',
                         depth_multiplier=1, activation='relu',
                         depthwise_initializer=glorot_uniform(seed=921212),
                         pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X3)
    X3 = MaxPooling2D((1, pool_size), strides=(3, 1))(X3)
    X3 = SeparableConv2D(filters=64, kernel_size=(13, 1), strides=(filter_stride, 1), padding='same',
                         depth_multiplier=1, activation='relu',
                         depthwise_initializer=glorot_uniform(seed=921212),
                         pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X3)
    X3 = MaxPooling2D((1, pool_size), strides=(3, 1))(X3)
    X3 = SeparableConv2D(filters=64, kernel_size=(13, 1), strides=(filter_stride, 1), padding='same',
                         depth_multiplier=1, activation='relu',
                         depthwise_initializer=glorot_uniform(seed=921212),
                         pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X3)
    X3 = MaxPooling2D((1, pool_size), strides=(3, 1))(X3)

    X4 = SeparableConv2D(filters=64, kernel_size=(17, 1), strides=(filter_stride, 1), padding='same',
                         depth_multiplier=1, activation='relu',
                         depthwise_initializer=glorot_uniform(seed=921212),
                         pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X_input)
    X4 = MaxPooling2D((1, pool_size), strides=(3, 1))(X4)
    X4 = SeparableConv2D(filters=64, kernel_size=(17, 1), strides=(filter_stride, 1), padding='same',
                         depth_multiplier=1, activation='relu',
                         depthwise_initializer=glorot_uniform(seed=921212),
                         pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X4)
    X4 = MaxPooling2D((1, pool_size), strides=(3, 1))(X4)
    X4 = SeparableConv2D(filters=64, kernel_size=(17, 1), strides=(filter_stride, 1), padding='same',
                         depth_multiplier=1, activation='relu',
                         depthwise_initializer=glorot_uniform(seed=921212),
                         pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X4)
    X4 = MaxPooling2D((1, pool_size), strides=(3, 1))(X4)
    X4 = SeparableConv2D(filters=64, kernel_size=(17, 1), strides=(filter_stride, 1), padding='same',
                         depth_multiplier=1, activation='relu',
                         depthwise_initializer=glorot_uniform(seed=921212),
                         pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X4)
    X4 = MaxPooling2D((1, pool_size), strides=(3, 1))(X4)

    X5 = SeparableConv2D(filters=64, kernel_size=(21, 1), strides=(filter_stride, 1), padding='same',
                         depth_multiplier=1, activation='relu',
                         depthwise_initializer=glorot_uniform(seed=921212),
                         pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X_input)
    X5 = MaxPooling2D((1, pool_size), strides=(3, 1))(X5)
    X5 = SeparableConv2D(filters=64, kernel_size=(21, 1), strides=(filter_stride, 1), padding='same',
                         depth_multiplier=1, activation='relu',
                         depthwise_initializer=glorot_uniform(seed=921212),
                         pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X5)
    X5 = MaxPooling2D((1, pool_size), strides=(3, 1))(X5)
    X5 = SeparableConv2D(filters=64, kernel_size=(21, 1), strides=(filter_stride, 1), padding='same',
                         depth_multiplier=1, activation='relu',
                         depthwise_initializer=glorot_uniform(seed=921212),
                         pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X5)
    X5 = MaxPooling2D((1, pool_size), strides=(3, 1))(X5)
    X5 = SeparableConv2D(filters=64, kernel_size=(21, 1), strides=(filter_stride, 1), padding='same',
                         depth_multiplier=1, activation='relu',
                         depthwise_initializer=glorot_uniform(seed=921212),
                         pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X5)
    X5 = MaxPooling2D((1, pool_size), strides=(3, 1))(X5)

    X6 = SeparableConv2D(filters=64, kernel_size=(25, 1), strides=(filter_stride, 1), padding='same',
                         depth_multiplier=1, activation='relu',
                         depthwise_initializer=glorot_uniform(seed=921212),
                         pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X_input)
    X6 = MaxPooling2D((1, pool_size), strides=(3, 1))(X6)
    X6 = SeparableConv2D(filters=64, kernel_size=(25, 1), strides=(filter_stride, 1), padding='same',
                         depth_multiplier=1, activation='relu',
                         depthwise_initializer=glorot_uniform(seed=921212),
                         pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X6)
    X6 = MaxPooling2D((1, pool_size), strides=(3, 1))(X6)
    X6 = SeparableConv2D(filters=64, kernel_size=(25, 1), strides=(filter_stride, 1), padding='same',
                         depth_multiplier=1, activation='relu',
                         depthwise_initializer=glorot_uniform(seed=921212),
                         pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X6)
    X6 = MaxPooling2D((1, pool_size), strides=(3, 1))(X6)
    X6 = SeparableConv2D(filters=64, kernel_size=(25, 1), strides=(filter_stride, 1), padding='same',
                         depth_multiplier=1, activation='relu',
                         depthwise_initializer=glorot_uniform(seed=921212),
                         pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X6)
    X6 = MaxPooling2D((1, pool_size), strides=(3, 1))(X6)

    X7 = SeparableConv2D(filters=64, kernel_size=(29, 1), strides=(filter_stride, 1), padding='same',
                         depth_multiplier=1, activation='relu',
                         depthwise_initializer=glorot_uniform(seed=921212),
                         pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X_input)
    X7 = MaxPooling2D((1, pool_size), strides=(3, 1))(X7)
    X7 = SeparableConv2D(filters=64, kernel_size=(29, 1), strides=(filter_stride, 1), padding='same',
                         depth_multiplier=1, activation='relu',
                         depthwise_initializer=glorot_uniform(seed=921212),
                         pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X7)
    X7 = MaxPooling2D((1, pool_size), strides=(3, 1))(X7)
    X7 = SeparableConv2D(filters=64, kernel_size=(29, 1), strides=(filter_stride, 1), padding='same',
                         depth_multiplier=1, activation='relu',
                         depthwise_initializer=glorot_uniform(seed=921212),
                         pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X7)
    X7 = MaxPooling2D((1, pool_size), strides=(3, 1))(X7)
    X7 = SeparableConv2D(filters=64, kernel_size=(29, 1), strides=(filter_stride, 1), padding='same',
                         depth_multiplier=1, activation='relu',
                         depthwise_initializer=glorot_uniform(seed=921212),
                         pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X7)
    X7 = MaxPooling2D((1, pool_size), strides=(3, 1))(X7)

    X8 = SeparableConv2D(filters=64, kernel_size=(33, 1), strides=(filter_stride, 1), padding='same',
                         depth_multiplier=1, activation='relu',
                         depthwise_initializer=glorot_uniform(seed=921212),
                         pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X_input)
    X8 = MaxPooling2D((1, pool_size), strides=(3, 1))(X8)
    X8 = SeparableConv2D(filters=64, kernel_size=(33, 1), strides=(filter_stride, 1), padding='same',
                         depth_multiplier=1, activation='relu',
                         depthwise_initializer=glorot_uniform(seed=921212),
                         pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X8)
    X8 = MaxPooling2D((1, pool_size), strides=(3, 1))(X8)
    X8 = SeparableConv2D(filters=64, kernel_size=(33, 1), strides=(filter_stride, 1), padding='same',
                         depth_multiplier=1, activation='relu',
                         depthwise_initializer=glorot_uniform(seed=921212),
                         pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X8)
    X8 = MaxPooling2D((1, pool_size), strides=(3, 1))(X8)
    X8 = SeparableConv2D(filters=64, kernel_size=(33, 1), strides=(filter_stride, 1), padding='same',
                         depth_multiplier=1, activation='relu',
                         depthwise_initializer=glorot_uniform(seed=921212),
                         pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X8)
    X8 = MaxPooling2D((1, pool_size), strides=(3, 1))(X8)

    X_collection = concatenate([X, X1, X2, X3, X4, X5, X6, X7, X8], axis=3)

    X_collection = Flatten()(X_collection)
    X_collection = Dense(4096, activation='relu', kernel_initializer=glorot_uniform(seed=921212))(X_collection)
    X_collection = Dense(128, activation='tanh', kernel_initializer=glorot_uniform(seed=921212))(X_collection)

    X = Dense(19, activation='softmax')(X_collection)
    return Model([X_input], X, name="inception_net")


acc_scores = []
combo_scores = []
proba_t = np.zeros((7500, 19))
for fold, (xx, yy) in enumerate(kfold.split(x, y)):
    print("{}train {}th fold{}".format('==' * 20, fold + 1, '==' * 20))
    y_ = to_categorical(y, num_classes=19)
    model = Net()
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    model.summary()
    plateau = ReduceLROnPlateau(monitor="val_acc",
                                verbose=1,
                                mode='max',
                                factor=0.5,
                                patience=15)
    early_stopping = EarlyStopping(monitor='val_acc',
                                   verbose=1,
                                   mode='max',
                                   patience=20)
    checkpoint = ModelCheckpoint(f'models/fold{fold}.h5',
                                 monitor='val_acc',
                                 verbose=0,
                                 mode='max',
                                 save_best_only=True)

    csv_logger = CSVLogger('../logs/log.csv', separator=',', append=True)
    model.fit(x[xx], y_[xx],
              epochs=500,
              batch_size=64,
              verbose=2,
              shuffle=True,
              validation_data=(x[yy], y_[yy]),
              callbacks=[plateau, early_stopping, checkpoint, csv_logger])
    model.load_weights(f'models/fold{fold}.h5')
    proba_x = model.predict(x[yy], verbose=0, batch_size=1024)
    proba_t += model.predict(t, verbose=0, batch_size=1024) / 5.

    oof_y = np.argmax(proba_x, axis=1)
    score1 = accuracy_score(y[yy], oof_y)
    # print('accuracy_score',score1)
    score = sum(acc_combo(y_true, y_pred) for y_true, y_pred in zip(y[yy], oof_y)) / oof_y.shape[0]
    print('accuracy_score', score1, 'acc_combo', score)
    acc_scores.append(score1)
    combo_scores.append(score)
print("5kflod mean acc score:{}".format(np.mean(acc_scores)))
print("5kflod mean combo score:{}".format(np.mean(combo_scores)))
sub.behavior_id = np.argmax(proba_t, axis=1)
sub.to_csv('result/submit_acc{}_combo{}.csv'.format(np.mean(acc_scores), np.mean(combo_scores)), index=False)
