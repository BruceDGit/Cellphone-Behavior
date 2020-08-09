# 网络结构
import pickle

from sklearn.metrics import *
from sklearn.model_selection import *
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import *
# 加载数据和模型
from tensorflow.keras.regularizers import l2

from load_data import *
from load_inv_data import *
from trys.sjf_class_weight import DatasetLoader
from trys.sjf_extend import *
from utils import *


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


def CNN(inputs, num_classes):
    x = Conv2D(32, 3, 1, padding="same", kernel_regularizer=l2(0.00))(inputs)
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

    x = Dense(num_classes, kernel_regularizer=l2(0.00))(x)
    x = BatchNormalization()(x)
    x = Softmax()(x)
    return x


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

kfold = StratifiedKFold(5, shuffle=True, random_state=12255877)
proba_t = np.zeros((7500, 19))
acc_scores = []
combo_scores = []
final_x = np.zeros((7292, 19))

for fold, (train_index, valid_index) in enumerate(kfold.split(x, y)):
    inputs = Input(shape=[64, 8, 1])
    outputs = CNN(inputs, num_classes=19)
    model = Model(inputs=inputs, outputs=outputs)
    _y = to_categorical(y, 19)
    plateau = ReduceLROnPlateau(monitor="val_score",
                                verbose=1,
                                mode='max',
                                factor=0.5,
                                patience=30)
    early_stopping = EarlyStopping(monitor='val_acc',
                                   verbose=1,
                                   mode='max',
                                   patience=60)
    checkpoint = ModelCheckpoint(f'models/sknet_baseline{fold}.h5',
                                 monitor='val_score',
                                 verbose=1,
                                 mode='max',
                                 save_best_only=True)
    weight_decay = WeightDecayScheduler(init_lr=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=AdamW(lr=0.001, weight_decay=6e-4), metrics=["acc", score])
    trained_model = model.fit(
        x[train_index],
        _y[train_index],
        batch_size=batch_size,
        class_weight=dict(enumerate((1 - class_weight) ** 2)),
        shuffle=True,
        validation_data=(x[valid_index], _y[valid_index]),
        epochs=800,
        callbacks=[plateau, early_stopping, checkpoint, weight_decay])
    model.load_weights(f'models/sknet_baseline{fold}.h5')
    proba_x = model.predict(x[valid_index],
                            verbose=0, batch_size=1024)
    proba_t += model.predict(x_val, verbose=0, batch_size=1024) / 5.
    final_x[valid_index] += proba_x

    oof_y = np.argmax(proba_x, axis=1)
    score1 = accuracy_score(y[valid_index], oof_y)
    # print('accuracy_score',score1)
    score = sum(acc_combo(y_true, y_pred) for y_true, y_pred in zip(y[valid_index], oof_y)) / oof_y.shape[0]
    print('accuracy_score', score1, 'acc_combo', score)
    acc_scores.append(score1)
    combo_scores.append(score)

sub = pd.read_csv('data/提交结果示例.csv')
sub.to_csv('result/sk_acc{}_combo{}.csv'.format(np.mean(acc_scores), np.mean(combo_scores)), index=False)
pd.DataFrame(proba_t, columns=['pred_{}'.format(i) for i in range(19)]).to_csv(
    'result/sk_proba_t_{}.csv'.format(np.mean(acc_scores)), index=False)
pd.DataFrame(final_x, columns=['pred_{}'.format(i) for i in range(19)]).to_csv(
    'result/sk_proba_x_{}.csv'.format(np.mean(acc_scores)), index=False)
