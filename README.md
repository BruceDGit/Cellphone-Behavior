## 模型

## CNN
1.Baseline  0.711左右的分数

https://mp.weixin.qq.com/s/r7Ai8FVSPRB71PVghYk75A
更换优化算法 0.7296349206349205
```text
model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
```
线上0.7296349206349205

2. 开源Baseline ot_code代码 线上0.74

```text
23/23 [==============================] - 1s 43ms/step - loss: 0.1183 - acc: 0.9578 - val_loss: 1.1392 - val_acc: 0.7455 - lr: 7.8125e-06
Epoch 178/500
23/23 [==============================] - 1s 43ms/step - loss: 0.1159 - acc: 0.9597 - val_loss: 1.1376 - val_acc: 0.7476 - lr: 7.8125e-06
Epoch 00178: early stopping

```
线上0.7456984126984126

将优化算法调整为rmsprop线上分数0.7463015873015874
```text
Epoch 161/500
23/23 [==============================] - 1s 44ms/step - loss: 0.1667 - acc: 0.9412 - val_loss: 0.9308 - val_acc: 0.7510 - lr: 7.8125e-06
Epoch 162/500
23/23 [==============================] - 1s 44ms/step - loss: 0.1751 - acc: 0.9403 - val_loss: 0.9285 - val_acc: 0.7510 - lr: 7.8125e-06
Epoch 00162: early stopping
```

3. bp_cnn
在卷积层之后添加批归一化
```text
X = Conv2D(filters=128,
               kernel_size=(3, 3),
               activation='relu',
               padding='same')(X)
    X = BatchNormalization()(X)
```
训练效果有明显提升
```text
Epoch 153/500
23/23 - 1s - loss: 0.0745 - acc: 0.9794 - val_loss: 0.8036 - val_acc: 0.7716 - lr: 7.8125e-06
Epoch 154/500
23/23 - 1s - loss: 0.0777 - acc: 0.9775 - val_loss: 0.8064 - val_acc: 0.7709 - lr: 7.8125e-06
Epoch 00154: early stopping

```
4. 尝试不同的归一化方式
- Group Normalization (TensorFlow Addons)
-Instance Normalization (TensorFlow Addons)
- Layer Normalization (TensorFlow Core)


## 失败的尝试
- 添加特征
- lstm：也不算失败，目前网络结构比较简单，个人感觉应该LSTM的效果比CNN要好
- 超参数调优