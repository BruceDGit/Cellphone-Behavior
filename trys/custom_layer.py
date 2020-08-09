# -*- coding: utf-8 -*-
"""
@author: quincyqiang
@software: PyCharm
@file: custom_layer.py
@time: 2020/7/25 17:53
@description：
"""

# 网络结构
import tensorflow as tf
# 加载数据和模型
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import UpSampling1D
from tensorflow.keras.regularizers import l2


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