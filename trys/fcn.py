# -*- coding: utf-8 -*-
"""
@author: quincyqiang
@software: PyCharm
@file: fcn.py
@time: 2020/7/25 23:48
@description：
"""

from sklearn.metrics import *
from sklearn.model_selection import *
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.layers import *

from load_data import *
from load_inv_data import *
from utils import *

train_lstm, y1, test_lstm, seq_len, _ = load_lstm_data()

y = load_y()

# x_train_mean = train_lstm.mean()
# x_train_std = train_lstm.std()
# train_lstm = (train_lstm - x_train_mean) / (x_train_std)
#
# test_lstm = (test_lstm - x_train_mean) / (x_train_std)
np.random.seed(813306)


def build_fcn():
    input = keras.layers.Input(shape=(train_lstm.shape[1:]))
    x = Reshape((60, train_lstm.shape[2], 1), input_shape=(60, train_lstm.shape[2]))(input)

    #    drop_out = Dropout(0.2)(x)
    conv1 = keras.layers.Conv2D(128, 3, 1, padding='same')(x)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation('relu')(conv1)

    #    drop_out = Dropout(0.2)(conv1)
    conv2 = keras.layers.Conv2D(256, 3, 1, padding='same')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    #    drop_out = Dropout(0.2)(conv2)
    conv3 = keras.layers.Conv2D(128, 3, 1, padding='same')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)

    full = keras.layers.GlobalAveragePooling2D()(conv3)
    # out = BatchNormalization()(Dropout(0.2)(Dense(128, activation='relu')(Flatten()(out))))
    out = keras.layers.Dense(19, activation='softmax')(full)
    model = keras.models.Model(inputs=input, outputs=out)
    return model


acc_scores = []
combo_scores = []
final_x = np.zeros((7292, 19))
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
    model = build_fcn()
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
    checkpoint = ModelCheckpoint(f'models/fold{fold}_fcn.h5',
                                 monitor='val_acc',
                                 verbose=0,
                                 mode='max',
                                 save_best_only=True)

    csv_logger = CSVLogger('../logs/log.csv', separator=',', append=True)
    history = model.fit(train_lstm[train_index],
                        y_[train_index],
                        epochs=500,
                        batch_size=256,
                        verbose=1,
                        shuffle=True,
                        class_weight=dict(enumerate((1 - class_weight) ** 3)),
                        validation_data=(train_lstm[valid_index],
                                         y_[valid_index]),
                        callbacks=[plateau, early_stopping, checkpoint, csv_logger])
    model.load_weights(f'models/fold{fold}_fcn.h5')
    proba_x = model.predict(train_lstm[valid_index],
                            verbose=0, batch_size=1024)
    proba_t += model.predict(test_lstm, verbose=0, batch_size=1024) / 5.
    final_x[valid_index] += proba_x

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
sub.to_csv('result/fcn_acc{}_combo{}.csv'.format(np.mean(acc_scores), np.mean(combo_scores)), index=False)
pd.DataFrame(proba_t, columns=['pred_{}'.format(i) for i in range(19)]).to_csv(
    'result/fcn_proba_t_{}.csv'.format(np.mean(acc_scores)), index=False)
pd.DataFrame(final_x, columns=['pred_{}'.format(i) for i in range(19)]).to_csv(
    'result/fcn_proba_x_{}.csv'.format(np.mean(acc_scores)), index=False)
