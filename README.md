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

删除两个特征之后，`线上0.7644`

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
`线上分数0.764825396825397`

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
```text
Epoch 00112: ReduceLROnPlateau reducing learning rate to 6.25000029685907e-05.
23/23 - 2s - loss: 0.0067 - acc: 0.9995 - val_loss: 0.9610 - val_acc: 0.7970 - lr: 1.2500e-04
Epoch 113/500
23/23 - 2s - loss: 0.0052 - acc: 0.9998 - val_loss: 0.9456 - val_acc: 0.8038 - lr: 6.2500e-05
Epoch 114/500
23/23 - 2s - loss: 0.0045 - acc: 0.9995 - val_loss: 0.9422 - val_acc: 0.8018 - lr: 6.2500e-05
Epoch 00114: early stopping

```
`线上分数0.7607777777777778`
## 失败的尝试
- 添加特征
- lstm：也不算失败，目前网络结构比较简单，个人感觉应该LSTM的效果比CNN要好
- 超参数调优
- 加上归一化之后 效果比较差
```text
print('Scaler....')
for col in ['acc_x','acc_y','acc_z','acc_xg','acc_yg','acc_zg','mod','modg']:
    scaler = MinMaxScaler().fit(data[[col]])
    train[[col]] = scaler.transform(train[[col]])
```

- 加入特征
```text
角度特征

# 角度
# data['acc_x_cos(x)'] = data['acc_x'] / data['mod']
# data['acc_y_cos(y)'] = data['acc_y'] / data['mod']
# data['acc_z_cos(z)'] = data['acc_z'] / data['mod']
# data['acc_x_angle(x)'] = np.arccos(data['acc_x_cos(x)'])
# data['acc_y_angle(y)'] = np.arccos(data['acc_y_cos(y)'])
# data['acc_z_angle(z)'] = np.arccos(data['acc_z_cos(z)'])
```

```text
# 2020.7.8
data['mod2'] = data.acc_x ** 2 + data.acc_y ** 2 + data.acc_z ** 2
data['modg2'] = data.acc_xg ** 2 + data.acc_yg ** 2 + data.acc_zg ** 2

92/92 - 2s - loss: 0.0624 - acc: 0.9803 - val_loss: 0.8870 - val_acc: 0.7840 - lr: 3.1250e-05
Epoch 00117: early stopping
accuracy_score 0.7839506172839507 acc_combo 0.8136063753347688
```
`线上0.7614603174603174`

- 数据增强
```text
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True  # randomly flip imag
)

```
`线下0.792` `线上0.7577619047619049`