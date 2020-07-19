#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: har.py 
@time: 2020-07-12 02:12
@description:
"""

import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.model_selection import *
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.layers import *
from disout_tf2 import *
from load_data import *
from load_inv_data import *
from utils import *

train_features, _, test_features = load_features_data(feature_id=2)

train_lstm, y1, test_lstm, seq_len, _ = load_lstm_data()
train_lstm_inv, _, test_lstm_inv, _, _ = load_lstm_inv_data()
y = load_y()


def multi_conv2d(input_forward):
    input = Reshape((60, train_lstm.shape[2], 1), input_shape=(60, train_lstm.shape[2]))(input_forward)
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
    # X = BatchNormalization()(X)

    X = Dropout(0.3)(X)
    X = Conv2D(filters=512,
               kernel_size=(3, 3),
               activation='relu',
               padding='same')(X)
    # X = BatchNormalization()(X)
    X = GlobalAveragePooling2D()(X)
    X = Dropout(0.5)(X)
    # X = BatchNormalization()(Dropout(0.2)(Dense(128, activation='relu')(Flatten()(X))))
    return X


def Net():
    # 前向与后向卷积
    input_forward = Input(shape=(60, train_lstm.shape[2]), name='input_forward')
    input_backward = Input(shape=(60, train_lstm.shape[2]), name='input_backward')
    X_forward = multi_conv2d(input_forward)
    X_backward = multi_conv2d(input_backward)

    # 特征输入
    feature_input = Input(shape=(train_features.shape[1],), name='feature_input')
    dense = Dense(32, activation='relu')(feature_input)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.2)(dense)
    dense = Dense(64, activation='relu')(dense)
    dense = Dropout(0.2)(dense)
    dense = Dense(128, activation='relu')(dense)
    dense = Dropout(0.2)(dense)
    dense = Dense(256, activation='relu')(dense)
    dense = BatchNormalization()(dense)

    output = Concatenate(axis=-1)([X_forward, X_backward, dense])
    output = BatchNormalization()(Dropout(0.2)(Dense(640, activation='relu')(Flatten()(output))))

    output = Dense(19, activation='softmax')(output)
    return Model([input_forward, input_backward, feature_input], output)


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
    model = Net()
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
    history = model.fit([train_lstm[train_index],
                         train_lstm_inv[train_index],
                         train_features[train_index]],
                        y_[train_index],
                        epochs=500,
                        batch_size=256,
                        verbose=1,
                        shuffle=True,
                        class_weight=(1 - class_weight) ** 3,
                        validation_data=([train_lstm[valid_index],
                                          train_lstm_inv[valid_index],
                                          train_features[valid_index]],
                                         y_[valid_index]),
                        callbacks=[plateau, early_stopping, checkpoint, csv_logger])
    model.load_weights(f'models/fold{fold}.h5')
    proba_x = model.predict([train_lstm[valid_index], train_lstm_inv[valid_index], train_features[valid_index]],
                            verbose=0, batch_size=1024)
    proba_t += model.predict([test_lstm, test_lstm_inv, test_features], verbose=0, batch_size=1024) / 5.

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

sub = pd.read_csv('data/提交结果示例.csv')
sub.behavior_id = np.argmax(proba_t, axis=1)
sub.to_csv('result/har_acc{}_combo{}.csv'.format(np.mean(acc_scores), np.mean(combo_scores)), index=False)
pd.DataFrame(proba_t, columns=['pred_{}'.format(i) for i in range(19)]).to_csv(
    'result/har_proba_t_{}.csv'.format(np.mean(acc_scores)), index=False)
