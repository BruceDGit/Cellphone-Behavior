import tensorflow as tf
import numpy as np
from tensorflow import keras
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
import gc

import tensorflow as tf
import numpy as np


def data_enhance(method, train_data, train_labels):
    if method == 'noise':
        # noise = train_data + np.random.normal(0, 0.1, size=train_data.shape)
        # return noise, train_labels
        noise1 = train_data + np.random.normal(0, 0.1, size=train_data.shape)
        noise2 = train_data + np.random.uniform(-0.3, 0.3, train_data.shape)
        noise = np.r_[noise1, noise2]
        train_labels = np.r_[train_labels, train_labels]
        return noise, train_labels

    elif method == 'mixup':
        index = [i for i in range(len(train_labels))]
        np.random.shuffle(index)

        x_mixup = np.zeros(train_data.shape)
        y_mixup = np.zeros(train_labels.shape)

        for i in range(len(train_labels)):
            x1 = train_data[i]
            x2 = train_data[index[i]]
            y1 = train_labels[i]
            y2 = train_labels[index[i]]

            factor = np.random.beta(0.2, 0.2)

            x_mixup[i] = x1 * factor + x2 * (1 - factor)
            y_mixup[i] = y1 * factor + y2 * (1 - factor)

        return x_mixup, y_mixup


def save_results(path, output):
    print('saving...')

    df_r = pd.DataFrame(columns=['fragment_id', 'behavior_id'])
    for i in range(len(output)):
        behavior_id = output[i]
        df_r = df_r.append(
            {'fragment_id': i, 'behavior_id': behavior_id}, ignore_index=True)
    df_r.to_csv(path, index=False)


def single_score(y, y_pred):
    # 数值ID与行为编码的对应关系
    mapping = {0: 'A_0', 1: 'A_1', 2: 'A_2', 3: 'A_3',
               4: 'D_4', 5: 'A_5', 6: 'B_1', 7: 'B_5',
               8: 'B_2', 9: 'B_3', 10: 'B_0', 11: 'A_6',
               12: 'C_1', 13: 'C_3', 14: 'C_0', 15: 'B_6',
               16: 'C_2', 17: 'C_5', 18: 'C_6'}
    # 将行为ID转为编码
    code_y, code_y_pred = mapping[y], mapping[y_pred]
    if code_y == code_y_pred:  # 编码完全相同得分1.0
        return 1.0
    elif code_y.split("_")[0] == code_y_pred.split("_")[0]:  # 编码仅字母部分相同得分1.0/7
        return 1.0 / 7
    elif code_y.split("_")[1] == code_y_pred.split("_")[1]:  # 编码仅数字部分相同得分1.0/3
        return 1.0 / 3
    else:
        return 0.0


def py_score(y_true, y_pred):
    y_true = np.argmax(y_true, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)
    scores = []
    for i in range(len(y_true)):
        scores.append(single_score(y_true[i], y_pred[i]))
    mean_score = np.array(scores, dtype="float32").mean()
    return mean_score, mean_score


def score(y_true, y_pred):
    """线上评测所使用的评测方法
    Args:
      y_true:one_hot编码的标签
      y_pred:网络类别置信度预测
    Returns:
      Tensor标量
    """
    mean_score = tf.py_function(py_score, [y_true, y_pred], [tf.float32, tf.float32])[0]
    return tf.reshape(mean_score, shape=())


def acc_combo(y, y_pred):
    # 数值ID与行为编码的对应关系
    mapping = {0: 'A_0', 1: 'A_1', 2: 'A_2', 3: 'A_3',
               4: 'D_4', 5: 'A_5', 6: 'B_1', 7: 'B_5',
               8: 'B_2', 9: 'B_3', 10: 'B_0', 11: 'A_6',
               12: 'C_1', 13: 'C_3', 14: 'C_0', 15: 'B_6',
               16: 'C_2', 17: 'C_5', 18: 'C_6'}
    # 将行为ID转为编码
    code_y, code_y_pred = mapping[y], mapping[y_pred]
    if code_y == code_y_pred:
        # 编码完全相同得分1.0
        return 1.0
    elif code_y.split("_")[0] == code_y_pred.split("_")[0]:  # 编码仅字母部分相同得分1.0/7
        return 1.0 / 7
    elif code_y.split("_")[1] == code_y_pred.split("_")[1]:  # 编码仅数字部分相同得分1.0/3
        return 1.0 / 3
    else:
        return 0.0
