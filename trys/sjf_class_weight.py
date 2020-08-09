# 数据预处理
import tensorflow as tf
import numpy as np
import os
from time import time
from scipy.signal import welch
from scipy.interpolate import interp1d as interp
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import Progbar
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 加载数据和模型
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

class Preprocessing(object):
    """
    对序列进行提取特征等预处理
    为了可能处理变长序列,因此输入为list
    """

    def __init__(self, with_label=True):
        self.with_label = with_label

    def __call__(self, sequence):
        for i in range(len(sequence)):
            time_point = np.expand_dims(sequence[i][0], axis=0)
            if self.with_label:
                label = np.expand_dims(sequence[i][-1], axis=0)
                new_sequence = self.for_each(sequence[i][1:-1])
                sequence[i] = np.concatenate([time_point, new_sequence, label], axis=0)
            else:
                new_sequence = self.for_each(sequence[i][1:])
                sequence[i] = np.concatenate([time_point, new_sequence], axis=0)
        sequence = self.for_all(sequence)
        return sequence

    def smooth(self, array, decay_rate=0.9):
        _smooth = np.zeros(array.shape)
        for i in range(1, len(array) - 1):
            decay = min(decay_rate, (i + 1) / (i + 10))
            _smooth[i] = _smooth[i - 1] * decay + (1 - decay) * array[i]
        return _smooth

    def for_all(self, sequence):  # 变长序列可以numpy吗#
        return sequence
        mean = np.zeros(shape=sequence[0].shape[0])
        std = np.zeros(shape=sequence[0].shape[0])
        lenth = len(sequence)
        for index in range(lenth):
            if self.with_label:
                _range = range(1, sequence[index].shape[0] - 1)
            else:
                _range = range(1, sequence[index].shape[0])
            for i in _range:
                mean[i] += sequence[index][i].mean() / lenth
                std[i] += sequence[index][i].std() / lenth
        for index in range(lenth):
            if self.with_label:
                _range = range(1, sequence[index].shape[0] - 1)
            else:
                _range = range(1, sequence[index].shape[0])
            for i in _range:
                sequence[index][i] = (sequence[index][i] - mean[i]) / std[i]
        return sequence

    def for_each(self, sequence):
        acc = (sequence[0] ** 2 + sequence[1] ** 2 + sequence[2] ** 2) ** 0.5
        acc = np.expand_dims(acc, axis=0)
        acc_g = (sequence[3] ** 2 + sequence[4] ** 2 + sequence[5] ** 2) ** 0.5
        acc_g = np.expand_dims(acc_g, axis=0)
        sequence = np.concatenate([sequence, acc, acc_g], axis=0)
        return sequence


# 加载数据
class DatasetLoader(object):
    def __init__(self, csv_file, with_label=True, num_classes=19):
        self.csv_file = csv_file
        self.with_label = with_label
        self.format = "channel_last"
        self.split = False
        self.names = self.get_feature_names()
        self.num_classes = num_classes
        self.data_split = False

    def get_feature_names(self):
        with open(self.csv_file) as f:
            examples = {}
            names = f.readline().split(',')[1:]
            names[-1] = names[-1][:-1]
            return names

    def make_numpy(self, num_interpolation=200, with_label=True):
        '''将数据读取并保存为Numpy数组
               Args:
                 num_interpolation:差值法采样点个数
                 with_label：是否带标签
               Returns:
                 A list,shape=[num_examples,keys,length]
        '''
        # 数据读取
        if self.csv_file is None:
            raise ValueError("sub dataset cannnot get numpy data")
        print("Loading date...")
        line = {}
        with open(self.csv_file) as f:
            examples = {}
            names = f.readline().split(',')[1:]
            names[-1] = names[-1][:-1]
            while True:
                try:
                    line = f.readline().split(",")
                    if line is None:
                        break
                    for i in range(len(line)):
                        line[i] = eval(line[i])
                    if not line[0] in examples:
                        examples[line[0]] = []
                    examples[line[0]].append(line[1:])
                except:
                    break
        print("done")
        # 格式转换
        for i in range(len(examples)):
            examples[i] = np.array(examples[i]).transpose([1, 0])
        self.examples = examples = list(examples.values())
        return self

    def resample(self, num_interpolation=200):
        examples = self.examples
        print("interpolate")
        bar = Progbar(len(examples))  # 进度条
        if num_interpolation and num_interpolation is not None:
            for i in range(len(examples)):
                range_len = examples[i][0][-1] - examples[i][0][0]
                range_start = examples[i][0][0]
                range_interval = range_len / num_interpolation
                interp_x = [range_start + range_interval * i for i in range(num_interpolation)]
                interp_data = [interp_x]
                for feature_id in range(1, len(self.names)):
                    try:
                        interp_f = interp(examples[i][0], examples[i][feature_id], kind="cubic")
                        interp_data.append([interp_f(x) for x in interp_x])
                    except:
                        raise ValueError("%d %d" % (i, feature_id), examples[i])
                bar.update(i)
                examples[i] = np.array(interp_data)
        print("\ndone")
        # 数据预处理
        preprocession = Preprocessing(with_label=self.with_label)
        examples = preprocession(examples)
        self.examples = np.array(examples, dtype="float32")
        if self.with_label:
            self.y = self.examples[::, -1, 0].tolist()
            self.x = self.examples[::, 1:-1, ::]
        else:
            self.x = self.examples[::, 1:, ::]
        return self

    def apply_class_weights(self):
        weights = np.zeros([self.num_classes])
        for i in range(len(weights)):
            weights[i] = (self.examples[::, -1:, 0] == i).sum()
        return weights / weights.sum()

    def data_format(self,string="channel_last"):
        if not string in ["channel_first", "channel_last"]:
            raise ValueError("either channel_last or channel_first are supported")
        self.format = string

    def apply_data(self):
        if self.with_label:
            if self.split:
                if self.format == "channel_first":
                    return self.x_train, self.y_train, self.x_test, self.y_test
                else:
                    return self.x_train.transpose([0, 2, 1]), self.y_train, self.x_test.transpose(
                        [0, 2, 1]), self.y_test

            else:
                if self.format == "channel_first":
                    return self.x, self.y
                else:
                    return self.x.transpose([0, 2, 1]), self.y
        else:
            if self.format == "channel_first":
                return self.x
            else:
                return self.x.transpose([0, 2, 1])

