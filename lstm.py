import argparse
import gc
import keras
import os
from scipy.signal import resample
import pandas as pd
from keras import Sequential, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import *
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow_addons as tfa
from tensorflow.keras.layers import add, Flatten
from tensorflow_addons.layers import *



data_path = 'data/'
train = pd.read_csv(data_path+'sensor_train.csv')
test = pd.read_csv(data_path+'sensor_test.csv')

train_sequences=list()
base_fea=['acc_x','acc_y','acc_z','acc_xg','acc_yg','acc_zg']

for index,group in train.groupby(by='fragment_id'):
    train_sequences.append(group[base_fea].values)
train_sequences[0]
y_train=train.drop_duplicates(subset=['fragment_id']).reset_index(drop=True)['behavior_id'].values
16