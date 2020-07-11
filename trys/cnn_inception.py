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
from load_data import load_data
from utils import acc_combo

X, y, X_test, seq_len, num_cols = load_data()

sub = pd.read_csv('../data/提交结果示例.csv')
kfold = StratifiedKFold(5, shuffle=True)
features = pd.read_csv('../data/features.csv')
features = features[['accg_mean',
                     'phi_acc_y_yz_std',
                     'accg_median',
                     'phi_acc_x_xz_std',
                     'acc_x_map_acc_y_map_mean',
                     'acc_xg_acc_yg_6',
                     'acc_xg_acc_yg_1',
                     'phi_acc_x_xy_std',
                     'acc_xg_acc_yg_acc_zg_7',
                     'acc_xg_acc_yg_4',
                     'acc_xg_acc_yg_2',
                     'acc_z_bool_mean',
                     'acc_xg_acc_yg_5',
                     'acc_xg_acc_yg_acc_zg_1',
                     'acc_y_map_acc_z_map_mean',
                     'acc_xg_acc_yg_3',
                     'acc_xg_acc_yg_7',
                     'acc_x_map_acc_z_map_mean',
                     'acc_xg_acc_yg_acc_zg_2',
                     'acc_z_mean',
                     'acc_y_mean',
                     'phi_acc_y_yz_mean',
                     'acc_xg_acc_yg_acc_zg_5',
                     'acc_x_acc_y_5',
                     'acc_xg_acc_yg_acc_zg_6',
                     'acc_xg_acc_yg_acc_zg_3',
                     'acc_x_mean',
                     'acc_xg_acc_yg_acc_zg_4',
                     'acc_xg_acc_yg_acc_zg_0',
                     'acc_yz_ygzy_gdirect_median',
                     'acc_y_bool_mean',
                     'acc_yz_ygzy_gdirect_std',
                     'acc_x_acc_y_3',
                     'acc_yz_ygzy_gdirect_mean',
                     'acc_xy_thea1_std',
                     'acc_xg_div_acc_x_std',
                     'two_dim_acc_yg_acc_zg_mean',
                     'acc_x_acc_y_acc_z_7',
                     'acc_xg_diff_acc_x_std',
                     'phi_acc_x_xz_mean',
                     'acc_xg_map_acc_yg_min',
                     'acc_xg_acc_yg_0',
                     'acc_x_acc_y_acc_z_5',
                     'acc_yz_ygzy_gdirect_min',
                     'acc_xy_thea1_min',
                     'acc_xg_map_acc_zg_mean',
                     'acc_yg_map_acc_zg_mean',
                     'acc_yg_map_acc_zg_min',
                     'phi_acc_xg_xy_std',
                     'phi_acc_x_xy_mean']]
x_w2v = features[:X.shape[0]].values
t_w2v = features[X.shape[0]:].values

fea_size = x_w2v.shape[1]


def Net():
    input = Input(shape=(60, num_cols))
    pred = Conv1D(filters=32,
                  kernel_size=2,
                  strides=1,
                  padding='same',
                  activation='relu')(input)
    pred = MaxPooling1D(pool_size=2, strides=2, padding='same')(pred)
    pred = Conv1D(filters=32,
                  kernel_size=2,
                  strides=1,
                  padding='same',
                  activation='relu')(pred)
    pred = MaxPooling1D(pool_size=2, strides=2, padding='same')(pred)
    pred = Conv1D(filters=64,
                  kernel_size=2,
                  strides=1,
                  padding='same',
                  activation='relu')(pred)
    pred = MaxPooling1D(pool_size=2, strides=2, padding='same')(pred)
    pred = Conv1D(filters=64,
                  kernel_size=2,
                  strides=1,
                  padding='same',
                  activation='relu')(pred)
    pred = MaxPooling1D(pool_size=2, strides=2, padding='same')(pred)
    pred = BatchNormalization()(pred)
    pred = Dropout(0.5)(pred)

    conv1_11 = Conv1D(filters=64,
                      kernel_size=2,
                      strides=1,
                      padding='same',
                      activation='relu')(pred)
    conv1_21 = Conv1D(filters=64,
                      kernel_size=2,
                      strides=1,
                      padding='same',
                      activation='relu')(pred)
    conv1_31 = Conv1D(filters=64,
                      kernel_size=2,
                      strides=1,
                      padding='same',
                      activation='relu')(pred)
    avg_pool_41 = MaxPooling1D(pool_size=2, strides=1,
                               padding='same', )(pred)

    conv2_22 = Conv1D(filters=64,
                      kernel_size=2,
                      strides=1,
                      padding='same',
                      activation='relu')(conv1_21)
    conv4_32 = Conv1D(filters=64,
                      kernel_size=2,
                      strides=1,
                      padding='same',
                      activation='relu')(conv1_31)
    conv1_42 = Conv1D(filters=64,
                      kernel_size=2,
                      strides=1,
                      padding='same',
                      activation='relu')(avg_pool_41)

    pred = Concatenate(axis=2)([conv1_11, conv2_22, conv4_32, conv1_42])
    pred = BatchNormalization()(pred)
    pred = Dropout(0.5)(pred)
    pred = BatchNormalization()(Dropout(0.2)(Dense(128, activation='relu')(Flatten()(pred))))

    fea_input = Input(shape=(fea_size, 1))
    X = Conv1D(filters=32,
               kernel_size=2,
               strides=1,
               padding='same',
               activation='relu')(fea_input)
    X = MaxPooling1D(pool_size=2, strides=2, padding='same')(X)
    X = Conv1D(filters=32,
               kernel_size=2,
               strides=1,
               padding='same',
               activation='relu')(X)
    X = MaxPooling1D(pool_size=2, strides=2, padding='same')(X)
    X = Conv1D(filters=64,
               kernel_size=2,
               strides=1,
               padding='same',
               activation='relu')(X)
    X = MaxPooling1D(pool_size=2, strides=2, padding='same')(X)
    X = Conv1D(filters=64,
               kernel_size=2,
               strides=1,
               padding='same',
               activation='relu')(X)
    X = MaxPooling1D(pool_size=2, strides=2, padding='same')(X)
    X = BatchNormalization()(X)
    X = Dropout(0.5)(X)
    X = BatchNormalization()(Dropout(0.2)(Dense(128, activation='relu')(Flatten()(X))))

    pred = Concatenate(axis=-1)([pred, X])
    pred = BatchNormalization()(Dropout(0.2)(Dense(128, activation='relu')(Flatten()(pred))))

    pred = Dense(19, activation='softmax')(pred)
    return Model([input, fea_input], pred)


acc_scores = []
combo_scores = []
proba_t = np.zeros((7500, 19))
for fold, (train_index, valid_index) in enumerate(kfold.split(X, y)):
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
    model.fit([X[train_index], x_w2v[train_index]], y_[train_index],
              epochs=500,
              batch_size=64,
              verbose=2,
              shuffle=True,
              validation_data=([X[valid_index], x_w2v[valid_index]], y_[valid_index]),
              callbacks=[plateau, early_stopping, checkpoint, csv_logger])
    model.load_weights(f'models/fold{fold}.h5')
    proba_x = model.predict([X[valid_index], x_w2v[valid_index]], verbose=0, batch_size=1024)
    proba_t += model.predict([X_test, t_w2v], verbose=0, batch_size=1024) / 5.

    oof_y = np.argmax(proba_x, axis=1)
    score1 = accuracy_score(y[valid_index], oof_y)
    # print('accuracy_score',score1)
    score = sum(acc_combo(y_true, y_pred) for y_true, y_pred in zip(y[valid_index], oof_y)) / oof_y.shape[0]
    print('accuracy_score', score1, 'acc_combo', score)
    acc_scores.append(score1)
    combo_scores.append(score)
print("5kflod mean acc score:{}".format(np.mean(acc_scores)))
print("5kflod mean combo score:{}".format(np.mean(combo_scores)))
sub.behavior_id = np.argmax(proba_t, axis=1)
sub.to_csv('result/submit_acc{}_combo{}.csv'.format(np.mean(acc_scores), np.mean(combo_scores)), index=False)
