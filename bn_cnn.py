from scipy.signal import resample
import pandas as pd
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


def Net():
    input = Input(shape=(60, 8, 1))
    X = Conv2D(filters=64,
               kernel_size=(3, 3),
               activation='relu',
               padding='same')(input)
    X = BatchNormalization()(X)
    X = Conv2D(filters=128,
               kernel_size=(3, 3),
               activation='relu',
               padding='same')(X)
    X = BatchNormalization()(X)

    X = MaxPooling2D()(X)
    X = Dropout(0.2)(X)
    X = Conv2D(filters=256,
               kernel_size=(3, 3),
               activation='relu',
               padding='same')(X)
    X = BatchNormalization()(X)

    X = Dropout(0.3)(X)
    X = Conv2D(filters=512,
               kernel_size=(3, 3),
               activation='relu',
               padding='same')(X)
    X = BatchNormalization()(X)
    X = GlobalAveragePooling2D()(X)
    X = Dropout(0.5)(X)
    X = BatchNormalization()(Dropout(0.2)(Dense(128, activation='relu')(Flatten()(X))))
    X = Dense(19, activation='softmax')(X)
    return Model([input], X)


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

    csv_logger = CSVLogger('logs/log.csv', separator=',', append=True)
    model.fit(x[xx], y_[xx],
              epochs=500,
              batch_size=256,
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