## 模型

## CNN
1.Baseline  `0.711左右的分数`

https://mp.weixin.qq.com/s/r7Ai8FVSPRB71PVghYk75A
更换优化算法 `0.7296349206349205`
```text
model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
```
`线上0.7296349206349205`

2. 开源Baseline ot_code代码 `线上0.74`

```text
23/23 [==============================] - 1s 43ms/step - loss: 0.1183 - acc: 0.9578 - val_loss: 1.1392 - val_acc: 0.7455 - lr: 7.8125e-06
Epoch 178/500
23/23 [==============================] - 1s 43ms/step - loss: 0.1159 - acc: 0.9597 - val_loss: 1.1376 - val_acc: 0.7476 - lr: 7.8125e-06
Epoch 00178: early stopping

```
线上0.7456984126984126

将优化算法调整为rmsprop线上`分数0.7463015873015874`
```text
Epoch 161/500
23/23 [==============================] - 1s 44ms/step - loss: 0.1667 - acc: 0.9412 - val_loss: 0.9308 - val_acc: 0.7510 - lr: 7.8125e-06
Epoch 162/500
23/23 [==============================] - 1s 44ms/step - loss: 0.1751 - acc: 0.9403 - val_loss: 0.9285 - val_acc: 0.7510 - lr: 7.8125e-06
Epoch 00162: early stopping
```
从上面可以看出来rmsprop要优于Adam算法
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
`线上 0.7661587301587302`


4. 尝试不同的归一化方式
- Group Normalization (TensorFlow Addons)
```text
23/23 - 2s - loss: 0.0185 - acc: 0.9971 - val_loss: 0.8559 - val_acc: 0.8073 - lr: 1.5625e-05
Epoch 157/500
23/23 - 2s - loss: 0.0173 - acc: 0.9971 - val_loss: 0.8587 - val_acc: 0.8107 - lr: 7.8125e-06
Epoch 158/500
23/23 - 2s - loss: 0.0186 - acc: 0.9964 - val_loss: 0.8590 - val_acc: 0.8073 - lr: 7.8125e-06
Epoch 00158: early stopping
```
- Instance Normalization (TensorFlow Addons)
```text
23/23 - 2s - loss: 2.8817 - acc: 0.1030 - val_loss: 2.8588 - val_acc: 0.1021 - lr: 0.0010
Epoch 2/500
23/23 - 2s - loss: 2.8613 - acc: 0.0945 - val_loss: 2.8557 - val_acc: 0.1021 - lr: 0.0010
Epoch 3/500
23/23 - 2s - loss: 2.8608 - acc: 0.0965 - val_loss: 2.8554 - val_acc: 0.1021 - lr: 0.0010
Epoch 4/500
23/23 - 2s - loss: 2.8594 - acc: 0.0989 - val_loss: 2.8557 - val_acc: 0.1008 - lr: 0.0010
```
没有成功
- Layer Normalization (TensorFlow Core)


## 失败的尝试
- 添加特征
- lstm：也不算失败，目前网络结构比较简单，个人感觉应该LSTM的效果比CNN要好
- 超参数调优