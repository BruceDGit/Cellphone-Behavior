# -*- coding: utf-8 -*-
"""
@author: quincyqiang
@software: PyCharm
@file: resnet.py
@time: 2020/7/25 22:31
@description：https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline
"""

from sklearn.metrics import *
from sklearn.model_selection import *
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.layers import *

from load_data import *
from load_inv_data import *
from utils import *

train_features, _, test_features = load_features_data(feature_id=2)

train_lstm, y1, test_lstm, seq_len, _ = load_lstm_data()
y = load_y()

np.random.seed(813306)


def build_resnet(input_shape, n_feature_maps, nb_classes):
    print('build conv_x')
    input = keras.layers.Input(shape=(input_shape))
    x = Reshape((60, train_lstm.shape[2], 1), input_shape=(60, train_lstm.shape[2]))(input)

    conv_x = keras.layers.BatchNormalization()(x)
    conv_x = keras.layers.Conv2D(n_feature_maps, 8, 1, padding='same')(conv_x)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    print('build conv_y')
    conv_y = keras.layers.Conv2D(n_feature_maps, 5, 1, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    print('build conv_z')
    conv_z = keras.layers.Conv2D(n_feature_maps, 3, 1, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    is_expand_channels = not (input_shape[-1] == n_feature_maps)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv2D(n_feature_maps, 1, 1, padding='same')(x)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.BatchNormalization()(x)
    print('Merging skip connection')
    y = keras.layers.Add()([shortcut_y, conv_z])
    y = keras.layers.Activation('relu')(y)

    print('build conv_x')
    x1 = y
    conv_x = keras.layers.Conv2D(n_feature_maps * 2, 8, 1, padding='same')(x1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    print('build conv_y')
    conv_y = keras.layers.Conv2D(n_feature_maps * 2, 5, 1, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    print('build conv_z')
    conv_z = keras.layers.Conv2D(n_feature_maps * 2, 3, 1, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    is_expand_channels = not (input_shape[-1] == n_feature_maps * 2)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv2D(n_feature_maps * 2, 1, 1, padding='same')(x1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.BatchNormalization()(x1)
    print('Merging skip connection')
    y = keras.layers.Add()([shortcut_y, conv_z])
    y = keras.layers.Activation('relu')(y)

    print('build conv_x')
    x1 = y
    conv_x = keras.layers.Conv2D(n_feature_maps * 2, 8, 1, padding='same')(x1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    print('build conv_y')
    conv_y = keras.layers.Conv2D(n_feature_maps * 2, 5, 1, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    print('build conv_z')
    conv_z = keras.layers.Conv2D(n_feature_maps * 2, 3, 1, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    is_expand_channels = not (input_shape[-1] == n_feature_maps * 2)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv2D(n_feature_maps * 2, 1, 1, padding='same')(x1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.BatchNormalization()(x1)
    print('Merging skip connection')
    y = keras.layers.Add()([shortcut_y, conv_z])
    y = keras.layers.Activation('relu')(y)

    full = keras.layers.GlobalAveragePooling2D()(y)
    out = keras.layers.Dense(nb_classes, activation='softmax')(full)
    print('        -- model was built.')
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
    model = build_resnet(train_lstm.shape[1:], 64, 19)
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
    checkpoint = ModelCheckpoint(f'models/fold{fold}_resnet.h5',
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
                        class_weight=dict(enumerate((1 - class_weight) ** 3)),
                        validation_data=(train_lstm[valid_index],
                                         y_[valid_index]),
                        callbacks=[plateau, early_stopping, checkpoint, csv_logger])
    model.load_weights(f'models/fold{fold}.h5')
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
sub.to_csv('result/resnet_acc{}_combo{}.csv'.format(np.mean(acc_scores), np.mean(combo_scores)), index=False)
pd.DataFrame(proba_t, columns=['pred_{}'.format(i) for i in range(19)]).to_csv(
    'result/resnet_proba_t_{}.csv'.format(np.mean(acc_scores)), index=False)
pd.DataFrame(final_x, columns=['pred_{}'.format(i) for i in range(19)]).to_csv(
    'result/resnet_proba_x_{}.csv'.format(np.mean(acc_scores)), index=False)
