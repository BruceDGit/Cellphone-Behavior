from scipy.signal import resample
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
import numpy as np
from tqdm import tqdm
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import *
from tensorflow_addons.layers import *
from SpatialPyramidPooling import SpatialPyramidPooling
from sklearn.metrics import *
from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(2)

def acc_combo(y,y_pred):
    # 数值ID与行为编码的对应关系
    mapping = {0: 'A_0', 1: 'A_1', 2: 'A_2', 3: 'A_3',
               4: 'D_4', 5: 'A_5', 6: 'B_1', 7: 'B_5',
               8: 'B_2', 9: 'B_3', 10: 'B_0', 11: 'A_6',
               12: 'C_1', 13: 'C_3', 14: 'C_0', 15: 'B_6',
               16: 'C_2', 17: 'C_5', 18: 'C_6'}
    # 将行为ID转为编码
    code_y, code_y_pred = mapping[y], mapping[y_pred]
    if code_y == code_y_pred:
        # 编码完全相同得分1.0
        return 1.0
    elif code_y.split("_")[0] == code_y_pred.split("_")[0]:  # 编码仅字母部分相同得分1.0/7
        return 1.0 / 7
    elif code_y.split("_")[1] == code_y_pred.split("_")[1]:  # 编码仅数字部分相同得分1.0/3
        return 1.0 / 3
    else:
        return 0.0


train = pd.read_csv('data/sensor_train.csv')
test = pd.read_csv('data/sensor_test.csv')
train_size = len(train)

data = pd.concat([train, test], sort=False)

sub = pd.read_csv('data/提交结果示例.csv')
y = train.groupby('fragment_id')['behavior_id'].min()

train['mod'] = (train.acc_x ** 2 + train.acc_y ** 2 + train.acc_z ** 2) ** .5
train['modg'] = (train.acc_xg ** 2 + train.acc_yg ** 2 + train.acc_zg ** 2) ** .5
test['mod'] = (test.acc_x ** 2 + test.acc_y ** 2 + test.acc_z ** 2) ** .5
test['modg'] = (test.acc_xg ** 2 + test.acc_yg ** 2 + test.acc_zg ** 2) ** .5

x = np.zeros((7292, 60, 8, 1))
t = np.zeros((7500, 60, 8, 1))
for i in tqdm(range(7292)):
    tmp = train[train.fragment_id == i][:60]
    x[i, :, :, 0] = resample(tmp.drop(['fragment_id', 'time_point', 'behavior_id'],
                                      axis=1), 60, np.array(tmp.time_point))[0]
for i in tqdm(range(7500)):
    tmp = test[test.fragment_id == i][:60]
    t[i, :, :, 0] = resample(tmp.drop(['fragment_id', 'time_point'],
                                      axis=1), 60, np.array(tmp.time_point))[0]

kfold = StratifiedKFold(5, shuffle=True)


def Net(normalization='bn'):
    input = Input(shape=(60, 8, 1))
    X = Conv2D(filters=64,
               kernel_size=(3, 3),
               activation='relu',
               padding='same')(input)
    if normalization == 'bn':
        X = BatchNormalization()(X)
    elif normalization == 'gn':
        X = GroupNormalization()(X)
    elif normalization == 'in':
        X = InstanceNormalization(center=True,
                                  scale=True,
                                  beta_initializer="random_uniform",
                                  gamma_initializer="random_uniform")(X)
    else:
        X = LayerNormalization(center=True, scale=True)(X)

    X = Conv2D(filters=128,
               kernel_size=(3, 3),
               activation='relu',
               padding='same')(X)
    if normalization == 'bn':
        X = BatchNormalization()(X)
    elif normalization == 'gn':
        X = GroupNormalization()(X)
    elif normalization == 'in':
        X = InstanceNormalization(center=True,
                                  scale=True,
                                  beta_initializer="random_uniform",
                                  gamma_initializer="random_uniform")(X)
    else:
        X = LayerNormalization(center=True, scale=True)(X)

    X = MaxPooling2D()(X)
    X = Dropout(0.2)(X)
    X = Conv2D(filters=256,
               kernel_size=(3, 3),
               activation='relu',
               padding='same')(X)
    if normalization == 'bn':
        X = BatchNormalization()(X)
    elif normalization == 'gn':
        X = GroupNormalization()(X)
    elif normalization == 'in':
        X = InstanceNormalization(center=True,
                                  scale=True,
                                  beta_initializer="random_uniform",
                                  gamma_initializer="random_uniform")(X)
    else:
        X = LayerNormalization(center=True, scale=True)(X)

    X = Dropout(0.3)(X)
    X = Conv2D(filters=512,
               kernel_size=(3, 3),
               activation='relu',
               padding='same')(X)
    if normalization == 'bn':
        X = BatchNormalization()(X)
    elif normalization == 'gn':
        X = GroupNormalization()(X)
    elif normalization == 'in':
        X = InstanceNormalization(center=True,
                                  scale=True,
                                  beta_initializer="random_uniform",
                                  gamma_initializer="random_uniform")(X)
    else:
        X = LayerNormalization(center=True, scale=True)(X)

    # X = AveragePooling2D(pool_size=(10, 1))(X)
    # X = BatchNormalization()(Dropout(0.2)(Dense(128, activation='relu')(Flatten()(X))))
    # X =SpatialPyramidPooling([1, 2, 4])(X)

    X = GlobalAveragePooling2D()(X)
    X = Dropout(0.5)(X)
    X = Dense(19, activation='softmax')(X)
    return Model([input], X)


proba_x = np.zeros((7292, 19))
proba_t = np.zeros((7500, 19))
for fold, (xx, yy) in enumerate(kfold.split(x, y)):
    print("{}train {}th fold{}".format('==' * 20, fold + 1, '==' * 20))
    y_ = to_categorical(y, num_classes=19)
    model = Net(normalization='bn')
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    model.summary()
    plateau = ReduceLROnPlateau(monitor="val_acc",
                                verbose=1,
                                mode='max',
                                factor=0.5,
                                patience=8)
    early_stopping = EarlyStopping(monitor='val_acc',
                                   verbose=1,
                                   mode='max',
                                   patience=18)
    checkpoint = ModelCheckpoint(f'models/fold{fold}.h5',
                                 monitor='val_acc',
                                 verbose=0,
                                 mode='max',
                                 save_best_only=True)
    model.fit(x[xx], y_[xx],
              epochs=500,
              batch_size=256,
              verbose=2,
              shuffle=True,
              validation_data=(x[yy], y_[yy]),
              callbacks=[plateau, early_stopping, checkpoint])
    model.load_weights(f'models/fold{fold}.h5')
    proba_x=model.predict(x[yy],verbose=0,batch_size=1024)
    proba_t += model.predict(t, verbose=0, batch_size=1024) / 5.

    oof_y=np.argmax(proba_x,axis = 1)
    score1=accuracy_score(y[yy],oof_y)
    # print('accuracy_score',score1)
    score=sum(acc_combo(y_true,y_pred)for y_true, y_pred in zip(y[yy], oof_y)) / oof_y.shape[0]
    print('accuracy_score',score1,'acc_combo',score)
sub.behavior_id = np.argmax(proba_t, axis=1)
sub.to_csv('ln.csv', index=False)