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
from sklearn.preprocessing import MinMaxScaler
from utils import acc_combo
from load_data import load_lstm_data, load_cnn_data, load_features_data, load_y

train_lstm, _, test_lstm, seq_len, _ = load_lstm_data()
train_cnn, _, test_cnn, dim_height, dim_width, dim_channel = load_cnn_data()
train_features, _, test_features = load_features_data()
y = load_y()


def Net():
    input = Input(shape=(dim_height, dim_width, dim_channel))
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
    X = AveragePooling2D()(X)
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

    # feainput = Input(shape=(w2v_fea))
    # dense = Dense(32, activation='relu')(feainput)
    # dense = BatchNormalization()(dense)
    # dense = Dropout(0.2)(dense)
    # dense = Dense(64, activation='relu')(dense)
    # dense = Dropout(0.2)(dense)
    # dense = Dense(128, activation='relu')(dense)
    # dense = Dropout(0.2)(dense)
    # dense = Dense(256, activation='relu')(dense)
    # dense = BatchNormalization()(dense)
    fea_input = Input(shape=(train_features.shape[1], 1))
    dense = Conv1D(filters=32,
                   kernel_size=2,
                   strides=1,
                   padding='same',
                   activation='relu')(fea_input)
    dense = MaxPooling1D(pool_size=2, strides=2, padding='same')(dense)
    dense = Conv1D(filters=64,
                   kernel_size=2,
                   strides=1,
                   padding='same',
                   activation='relu')(dense)
    dense = MaxPooling1D(pool_size=2, strides=2, padding='same')(dense)
    dense = Conv1D(filters=64,
                   kernel_size=2,
                   strides=1,
                   padding='same',
                   activation='relu')(dense)
    dense = MaxPooling1D(pool_size=2, strides=2, padding='same')(dense)
    dense = Conv1D(filters=64,
                   kernel_size=2,
                   strides=1,
                   padding='same',
                   activation='relu')(dense)
    dense = MaxPooling1D(pool_size=2, strides=2, padding='same')(dense)
    dense = BatchNormalization()(Dropout(0.2)(Dense(128, activation='relu')(Flatten()(dense))))

    X = Concatenate(axis=-1)([X, dense])
    X = Dense(19, activation='softmax')(X)
    return Model([input, fea_input], X)


acc_scores = []
combo_scores = []
proba_t = np.zeros((7500, 19))
kfold = StratifiedKFold(5, shuffle=True, random_state=42)

for fold, (train_index, valid_index) in enumerate(kfold.split(train_cnn, y)):
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
                                   patience=40)
    checkpoint = ModelCheckpoint(f'models/fold{fold}.h5',
                                 monitor='val_acc',
                                 verbose=0,
                                 mode='max',
                                 save_best_only=True)

    csv_logger = CSVLogger('logs/log.csv', separator=',', append=True)
    model.fit([train_cnn[train_index], train_features[train_index]], y_[train_index],
              epochs=500,
              batch_size=64,
              verbose=2,
              shuffle=True,
              validation_data=([train_cnn[valid_index],
                                train_features[valid_index]],
                               y_[valid_index]),
              callbacks=[plateau, early_stopping, checkpoint, csv_logger])
    model.load_weights(f'models/fold{fold}.h5')
    proba_x = model.predict([train_cnn[valid_index], train_features[valid_index]], verbose=0, batch_size=1024)
    proba_t += model.predict([test_cnn, test_features], verbose=0, batch_size=1024) / 5.

    oof_y = np.argmax(proba_x, axis=1)
    score1 = accuracy_score(y[valid_index], oof_y)
    # print('accuracy_score',score1)
    score = sum(acc_combo(y_true, y_pred) for y_true, y_pred in zip(y[valid_index], oof_y)) / oof_y.shape[0]
    print('accuracy_score', score1, 'acc_combo', score)
    acc_scores.append(score1)
    combo_scores.append(score)
print("5kflod mean acc score:{}".format(np.mean(acc_scores)))
print("5kflod mean combo score:{}".format(np.mean(combo_scores)))

sub = pd.read_csv('data/提交结果示例.csv')

sub.behavior_id = np.argmax(proba_t, axis=1)
sub.to_csv('result/multiinput_acc{}_combo{}.csv'.format(np.mean(acc_scores), np.mean(combo_scores)), index=False)
