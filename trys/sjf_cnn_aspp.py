# 网络结构
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
# 加载数据和模型
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import UpSampling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

from trys.sjf_class_weight import DatasetLoader
from utils import score, acc_combo

class ASPP(object):
    def __init__(self, filters, kernel_size, activation, dilation_rates=[1, 2, 5, 9]):
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rates = dilation_rates
        self.activation = activation

    def branch(self, tensor, dilation_rate):
        x = Conv1D(self.filters, self.kernel_size, 1, padding="same", kernel_regularizer=l2(0.011),
                   activation=self.activation, dilation_rate=dilation_rate)(tensor)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        return x

    def pool(self, tensor):
        shape = tensor.shape.as_list()  # [batch,lenth,channels]
        scaler = GlobalAveragePooling1D()(tensor)  # [batch,channels]
        scaler = Reshape([1] + [shape[2]])(scaler)
        scaler = UpSampling1D(shape[1])(scaler)
        return scaler

    def __call__(self, tensor):
        output = [Conv1D(self.filters, 1, 1, padding="same", activation=self.activation, kernel_regularizer=l2(0.011))(
            tensor)]
        output.append(self.pool(tensor))
        for rate in self.dilation_rates:
            output.append(self.branch(tensor, rate))
        output = tf.keras.layers.concatenate(output)
        return output


def CNN(inputs, num_classes):
    x = Conv1D(64, 3, 1, padding="same", kernel_regularizer=l2(0.01))(inputs)
    x = BatchNormalization()(x)
    x = Activation(tf.nn.relu)(x)

    x = Conv1D(128, 3, 1, padding="same", kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Activation(tf.nn.relu)(x)

    x = MaxPooling1D(2, padding="same")(x)

    x = Conv1D(256, 3, 1, padding="same", kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Activation(tf.nn.relu)(x)

    x = ASPP(128, 3, activation=tf.nn.relu)(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.4)(x)

    x = Dense(num_classes, kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Softmax()(x)
    return x


inputs = Input(shape=[200, 8])
outputs = CNN(inputs, num_classes=19)

train_csv_file = "../data/sensor_train.csv"
test_csv_file = "../data/sensor_test.csv"
batch_size = 24

# train
if not os.path.exists('data/x.pkl'):
    dataset = DatasetLoader(train_csv_file, with_label=True, num_classes=19)
    dataset = dataset.make_numpy()
    dataset = dataset.resample(num_interpolation=64)
    x, y = dataset.apply_data()
    class_weight = dataset.apply_class_weights()

    with open('data/x.pkl', 'wb') as f:
        pickle.dump(x, f)
    with open('data/y.pkl', 'wb') as f:
        pickle.dump(y, f)
    with open('data/class_weight.pkl', 'wb') as f:
        pickle.dump(class_weight, f)
else:
    with open('data/x.pkl', 'rb') as f:
        x = pickle.load(f)
    with open('data/y.pkl', 'rb') as f:
        y = pickle.load(f)
    with open('data/class_weight.pkl', 'rb') as f:
        class_weight = pickle.load(f)
# test
if not os.path.exists('data/x_val.pkl'):
    dataset = DatasetLoader(test_csv_file, with_label=False)
    data = dataset.make_numpy()
    data = dataset.resample(num_interpolation=64)
    x_val = data.apply_data()
    with open('data/x_val.pkl', 'wb') as f:
        pickle.dump(x_val, f)
else:
    with open('data/x_val.pkl', 'rb') as f:
        x_val = pickle.load(f)

kfold = StratifiedKFold(5, shuffle=True, random_state=20001026)
proba_t = np.zeros((7500, 19))

acc_scores = []
combo_scores = []

for fold, (xx, yy) in enumerate(kfold.split(x, y)):
    inputs = Input(shape=[64, 8])
    outputs = CNN(inputs, num_classes=19)
    model = Model(inputs=inputs, outputs=outputs)
    _y = to_categorical(y, 19)
    plateau = ReduceLROnPlateau(monitor="val_score",
                                verbose=1,
                                mode='max',
                                factor=0.5,
                                patience=6)
    early_stopping = EarlyStopping(monitor='val_acc',
                                   verbose=1,
                                   mode='max',
                                   patience=15)
    checkpoint = ModelCheckpoint(f'fold{fold}.h5',
                                 monitor='val_score',
                                 verbose=1,
                                 mode='max',
                                 save_best_only=True)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["acc", score])
    trained_model = model.fit(
        x[xx],
        _y[xx],
        batch_size=batch_size,
        class_weight=(1 - class_weight) ** 3,
        shuffle=True,
        validation_data=(x[yy], _y[yy]),
        epochs=300,
        callbacks=[plateau, early_stopping, checkpoint])
    model.load_weights(f'fold{fold}.h5')
    proba_t += model.predict(x_val, verbose=0, batch_size=1024) / 5.

    proba_x = model.predict(x[yy],
                            verbose=0, batch_size=1024)
    oof_y = np.argmax(proba_x, axis=1)
    score1 = accuracy_score(y[yy], oof_y)
    # print('accuracy_score',score1)
    score = sum(acc_combo(y_true, y_pred) for y_true, y_pred in zip(y[yy], oof_y)) / oof_y.shape[0]
    print('accuracy_score', score1, 'acc_combo', score)
    acc_scores.append(score1)
    combo_scores.append(score)

print("5kflod mean acc score:{}".format(np.mean(acc_scores)))
print("5kflod mean combo score:{}".format(np.mean(combo_scores)))

sub = pd.read_csv('../data/提交结果示例.csv')
sub.behavior_id = np.argmax(proba_t, axis=1)
sub.to_csv('result/har_acc{}_combo{}.csv'.format(np.mean(acc_scores), np.mean(combo_scores)), index=False)
pd.DataFrame(proba_t, columns=['pred_{}'.format(i) for i in range(19)]).to_csv(
    'result/cnn_aspp_proba_t_{}.csv'.format(np.mean(acc_scores)), index=False)
