# 网络结构
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from extend import WeightDecayScheduler, AdamW
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.model_selection import *
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.layers import *

from load_data import *
from load_inv_data import *
from utils import *

train_features, _, test_features = load_features_data(feature_id=2)

train_lstm, y1, test_lstm, seq_len, _ = load_lstm_data()
train_lstm_inv, _, test_lstm_inv, _, _ = load_lstm_inv_data()
y = load_y()


class SoftThreshold(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SoftThreshold, self).__init__(**kwargs)

    def call(self, inputs):
        tensor = inputs[0]
        threshold = inputs[1]
        threshold = tf.abs(threshold)
        less = tf.cast(tf.less(tensor, -threshold), tf.float32)
        greater = tf.cast(tf.greater(tensor, threshold), tf.float32)
        tensor = (tensor - threshold) * greater + (tensor + threshold) * less
        return tensor


class SKNet(object):
    def __init__(self, filters, kernel_size, activation, dilation_rates=[1, 1, 1, 1]):
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rates = dilation_rates
        self.activation = activation
        self.sk_rate = 16

    def branch(self, tensor, dilation_rate):
        x = Conv2D(self.filters, self.kernel_size, 1, padding="same", kernel_regularizer=l2(0.00),
                   activation=self.activation, dilation_rate=dilation_rate)(tensor)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        return x

    def attention(self, tensor):
        add = tf.keras.layers.add(tensor)
        shape = add.shape.as_list()
        squeeze = GlobalAveragePooling2D()(add)
        squeeze = Dense(shape[-1] // self.sk_rate, activation=tf.nn.relu)(squeeze)
        extract = Dense(shape[-1], activation=tf.nn.softmax)(squeeze)
        extract = tf.expand_dims(extract, axis=1)
        extract = tf.expand_dims(extract, axis=1)
        output = []
        for t in tensor:
            threshold = tf.keras.layers.multiply([t, extract])
            output.append(SoftThreshold()([t, threshold]))
        return output

    def pool(self, tensor):
        tensor = Conv2D(self.filters, self.kernel_size, 1, padding="same", kernel_regularizer=l2(0.00))(tensor)
        shape = tensor.shape.as_list()  # [batch,lenth,channels]
        scaler = AveragePooling2D((shape[1], shape[2]))(tensor)  # [batch,channels]
        scaler = UpSampling2D((shape[1], shape[2]))(scaler)
        return scaler

    def __call__(self, tensor):
        x = Conv2D(self.filters, 1, 1, padding="same",
                   activation=self.activation, kernel_regularizer=l2(0.00))(tensor)
        x = BatchNormalization()(x)
        x = Activation(None)(x)
        output = [x]
        output.append(self.pool(tensor))
        for rate in self.dilation_rates:
            output.append(self.branch(tensor, rate))
        output = self.attention(output)
        output = tf.keras.layers.add(output)
        output = Conv2D(self.filters, 3, 1, padding="same", kernel_regularizer=l2(0.00))(output)
        output = BatchNormalization()(output)
        output = Activation(self.activation)(output)
        return output


def CNN(input_forward):
    input = Reshape((60, train_lstm.shape[2], 1), input_shape=(60, train_lstm.shape[2]))(input_forward)
    x = Conv2D(32, 3, 1, padding="same", kernel_regularizer=l2(0.00))(input)
    x = BatchNormalization()(x)
    x = Activation(tf.nn.relu)(x)

    x = Conv2D(64, 3, 1, padding="same", kernel_regularizer=l2(0.00))(x)
    x = BatchNormalization()(x)
    x = Activation(tf.nn.relu)(x)

    x = Conv2D(128, 3, 1, padding="same", kernel_regularizer=l2(0.00))(x)
    x = BatchNormalization()(x)
    x = Activation(tf.nn.relu)(x)

    x = MaxPooling2D(2, padding="same")(x)
    x = Dropout(0.5)(x)

    x = Conv2D(192, 3, 1, padding="same", kernel_regularizer=l2(0.00))(x)
    x = BatchNormalization()(x)
    x = Activation(tf.nn.relu)(x)
    x = Dropout(0.5)(x)

    x = SKNet(384, 3, activation=tf.nn.relu)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)

    return x


def Net():
    input_forward = Input(shape=(60, train_lstm.shape[2]))
    input_backward = Input(shape=(60, train_lstm.shape[2]))
    X_forward = CNN(input_forward)
    X_backward = CNN(input_backward)

    feainput = Input(shape=(train_features.shape[1],))
    dense = Dense(32, activation='relu')(feainput)
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
    return Model([input_forward, input_backward, feainput], output)


acc_scores = []
combo_scores = []
proba_t = np.zeros((7500, 19))
kfold = StratifiedKFold(5, shuffle=True, random_state=42)

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
sub.to_csv('result/sk_acc{}_combo{}.csv'.format(np.mean(acc_scores), np.mean(combo_scores)), index=False)
pd.DataFrame(proba_t, columns=['pred_{}'.format(i) for i in range(19)]).to_csv(
    'result/sk_proba_t_{}.csv'.format(np.mean(acc_scores)), index=False)
