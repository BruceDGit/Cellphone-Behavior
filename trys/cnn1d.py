import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers

from load_data import load_lstm_data
from utils import acc_combo

X_train, y_train, test_lstm, seq_len, _ = load_lstm_data()

kfold = StratifiedKFold(5, shuffle=True)


def zeropad(x):
    y = K.zeros_like(x)
    return K.concatenate([x, y], axis=2)


def zeropad_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3
    shape[2] *= 2
    return tuple(shape)


def resnet_block(layer, num_filters, subsample_length, block_index, conv_increase_channels_at, conv_num_skip):
    shortcut = MaxPooling1D(pool_size=subsample_length)(layer)
    zero_pad = (block_index % conv_increase_channels_at) == 0 and block_index > 0

    if zero_pad is True:
        shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)

    for i in range(conv_num_skip):
        if not (block_index == 0 and i == 0):
            layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)
            layer = Dropout(0.2)(layer)

        layer = Conv1D(filters=num_filters, kernel_size=16, strides=subsample_length if i == 0 else 1,
                       padding='same',
                       kernel_initializer='he_normal')(layer)
    layer = Add()([shortcut, layer])
    return layer


def densenet_block(layer, num_filters, subsample_length, block_index, zero_pad):
    shortcut = MaxPooling1D(pool_size=subsample_length)(layer)
    if zero_pad is True:
        shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)

    layer = Conv1D(filters=num_filters, kernel_size=16, strides=subsample_length,
                   padding='same',
                   kernel_initializer='he_normal')(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.2)(layer)
    layer = Conv1D(filters=num_filters, kernel_size=16, strides=1,
                   padding='same',
                   kernel_initializer='he_normal')(layer)
    layer = tf.keras.layers.concatenate([shortcut, layer])

    # transition
    layer = BatchNormalization()(layer)
    layer = Conv1D(filters=num_filters, kernel_size=1, strides=1)(layer)
    return layer


def build_model():
    inputs = Input(shape=X_train.shape[1:], dtype='float32', name='inputs')

    # add densenet layer
    layer = Conv1D(filters=32, kernel_size=16, strides=1, padding='same', kernel_initializer='he_normal')(inputs)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    conv_subsample_lengths = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    for index, subsample_length in enumerate(conv_subsample_lengths):
        num_filters = 2 ** (index // 4) * 32
        layer = densenet_block(layer, num_filters, subsample_length, index, (index > 0) and (index % 4 == 0))
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    layer = Flatten()(layer)
    layer_densenet = Dense(32, activation='relu')(layer)

    # add resnet layer
    layer = Conv1D(filters=32, kernel_size=32, strides=1, padding='same', kernel_initializer='he_normal')(inputs)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    conv_subsample_lengths = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    for index, subsample_length in enumerate(conv_subsample_lengths):
        num_filters = 2 ** (index // 4) * 32
        layer = resnet_block(layer, num_filters, subsample_length, index, 4, 2)

    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    layer = Flatten()(layer)
    layer = Dense(32, activation='relu')(layer)

    # Age and Gender
    inputs_age = Input(shape=(1,), dtype='int32', name='age_input')
    layer_age = Embedding(output_dim=32, input_dim=9, input_length=1)(inputs_age)
    layer_age = Flatten()(layer_age)

    inputs_gender = Input(shape=(1,), dtype='int32', name='gender_input')
    layer_gender = Embedding(output_dim=32, input_dim=3, input_length=1)(inputs_gender)
    layer_gender = Flatten()(layer_gender)

    # Concat all layers
    layer = tf.keras.layers.concatenate([layer_densenet, layer])

    # add output layer
    layer = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(layer)
    outputs = Dense(19, activation='sigmoid')(layer)

    # model = Model(inputs=[inputs, inputs_age, inputs_gender], outputs=[outputs])
    model = Model(inputs=[inputs], outputs=[outputs])

    return model


def Net():
    input = Input(X_train.shape[1:])

    # model = Conv1D(32, 3, activation='relu', )(input)
    # model = BatchNormalization()(model)
    # model = MaxPooling1D(pool_size=2)(model)
    #
    # model = Conv1D(64, 3, activation='relu')(model)
    # model = BatchNormalization()(model)
    # model = MaxPooling1D(pool_size=2)(model)
    #
    # model = Conv1D(128, 3, activation='relu')(model)
    # model = BatchNormalization()(model)
    # model = MaxPooling1D(pool_size=2)(model)
    #
    # model = Conv1D(256, 3, activation='relu')(model)
    # model = BatchNormalization()(model)
    # model = MaxPooling1D(pool_size=2)(model)
    #
    # model = Flatten()(model)
    # model = Dense(512, activation='relu')(model)
    # output = Dense(19, activation='softmax')(model)
    # ecg
    x = Conv1D(64, 2, activation='relu')(input)
    x = MaxPool1D()(x)
    x = BatchNormalization()(x)

    x = Conv1D(128, 2, activation='relu')(x)
    x = MaxPool1D()(x)
    x = BatchNormalization()(x)

    x = Conv1D(128, 3, activation='relu')(x)
    x = MaxPool1D()(x)
    x = BatchNormalization()(x)

    x = Conv1D(256, 4, activation='relu')(x)
    x = MaxPool1D()(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu', name='dense_1')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(2048, activation='relu', name='dense_2')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(19, activation='softmax', name='predictions')(x)
    return Model([input], outputs)


acc_scores = []
combo_scores = []
proba_t = np.zeros((7500, 19))
for fold, (xx, yy) in enumerate(kfold.split(X_train, y_train)):
    print("{}train {}th fold{}".format('==' * 20, fold + 1, '==' * 20))
    y_ = to_categorical(y_train, num_classes=19)
    model = build_model()
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
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
    model.fit(X_train[xx], y_[xx],
              epochs=500,
              batch_size=256,
              verbose=2,
              shuffle=True,
              validation_data=(X_train[yy], y_[yy]),
              callbacks=[plateau, early_stopping, checkpoint])
    model.load_weights(f'models/fold{fold}.h5')

    proba_x = model.predict(X_train[yy], verbose=0, batch_size=1024)
    proba_t += model.predict(test_lstm, verbose=0, batch_size=1024) / 5.

    oof_y = np.argmax(proba_x, axis=1)
    score1 = accuracy_score(y_train[yy], oof_y)
    # print('accuracy_score',score1)
    score = sum(acc_combo(y_true, y_pred) for y_true, y_pred in zip(y_train[yy], oof_y)) / oof_y.shape[0]
    print('accuracy_score', score1, 'acc_combo', score)
    acc_scores.append(score1)
    combo_scores.append(score)
print("5kflod mean acc score:{}".format(np.mean(acc_scores)))
print("5kflod mean combo score:{}".format(np.mean(combo_scores)))
sub = pd.read_csv('data/提交结果示例.csv')
sub.behavior_id = np.argmax(proba_t, axis=1)
sub.to_csv('result/cnn_acc{}_combo{}.csv'.format(np.mean(acc_scores), np.mean(combo_scores)), index=False)
