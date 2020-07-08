import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from hyperopt import Trials, STATUS_OK, tpe, rand
from hyperas import optim
from hyperas.distributions import choice, uniform
from sklearn.model_selection import train_test_split


def data():
    train = pd.read_csv('../data/sensor_train.csv')

    y = train.groupby('fragment_id')['behavior_id'].min()

    train['mod'] = (train.acc_x ** 2 + train.acc_y ** 2 + train.acc_z ** 2) ** .5
    train['modg'] = (train.acc_xg ** 2 + train.acc_yg ** 2 + train.acc_zg ** 2) ** .5

    x = np.zeros((7292, 60, 8, 1))
    for i in tqdm(range(7292)):
        tmp = train[train.fragment_id == i][:60]
        x[i, :, :, 0] = resample(tmp.drop(['fragment_id', 'time_point', 'behavior_id'],
                                          axis=1), 60, np.array(tmp.time_point))[0]
    y = to_categorical(y, num_classes=19)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def create_model(X_train, X_test, y_train, y_test):
    input = Input(shape=(60, 8, 1))
    X = Conv2D(filters={{choice([32, 64, 128])}},
               kernel_size={{choice([(3, 3), (4, 4), (5, 5)])}},
               activation={{choice(['relu', 'sigmoid', 'tanh'])}},
               padding='same')(input)
    X = Conv2D(filters={{choice([256, 512, 1024, 1216])}},
               kernel_size={{choice([(3, 3), (4, 4), (5, 5)])}},
               activation={{choice(['relu', 'sigmoid', 'tanh'])}},
               padding='same')(X)
    X = MaxPooling2D()(X)
    X = Dropout({{uniform(0, 1)}})(X)

    X = Conv2D(filters={{choice([256, 512, 1024])}},
               kernel_size={{choice([(3, 3), (4, 4), (5, 5)])}},
               activation={{choice(['relu', 'sigmoid', 'tanh'])}},
               padding='same')(X)
    X = Dropout({{uniform(0, 1)}})(X)

    X = Conv2D(filters={{choice([512, 1024,1600 ])}},
               kernel_size={{choice([(3, 3), (4, 4), (5, 5)])}},
               activation={{choice(['relu', 'sigmoid', 'tanh'])}},
               padding='same')(X)
    X = GlobalMaxPooling2D()(X)
    X = Dropout({{uniform(0, 1)}})(X)
    X = Dense(19, activation='softmax')(X)
    model = Model([input], X)
    model.compile(loss='categorical_crossentropy',
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}},
                  metrics=['acc'])
    plateau = ReduceLROnPlateau(monitor="val_acc",
                                verbose=0,
                                mode='max',
                                factor=0.1,
                                patience=6)
    early_stopping = EarlyStopping(monitor='val_acc',
                                   verbose=0,
                                   mode='max',
                                   patience=10)
    result = model.fit(X_train, y_train,
                       epochs=100,
                       batch_size={{choice([64, 128])}},
                       verbose=2,
                       shuffle=True,
                       validation_split=0.1,
                       callbacks=[plateau, early_stopping]
                       )

    # get the highest validation accuracy of the training epochs
    validation_acc = np.amax(result.history['val_acc'])
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    X_train, X_test, y_train, y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
