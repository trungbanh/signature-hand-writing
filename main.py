#! /usr/bin/python3
import cv2
import pickle

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from src.preprocess import readImage
from src.preprocess import dataModel, label

# listImage, _ = readImage()
# pickle.dump(listImage, open("../dataModel-15-66.pkl", "wb"))

xTrain, xTest, yTrain, yTest = train_test_split(
    dataModel(), label(), test_size=0.1, random_state=20)

xTrain = xTrain.astype('float32')
xTest = xTest.astype('float32')
xTrain /= 255
xTest /= 255
yTrain = keras.utils.to_categorical(yTrain, 15)
yTest = keras.utils.to_categorical(yTest, 15)

if K.image_data_format() == 'channels_first':
    xTrain = xTrain.reshape(xTrain.shape[0], 1, 128, 64)
    xTest = xTest.reshape(xTest.shape[0], 1, 128, 64)
    print(x_train.shape, " ", labels.shape)
    input_shape = (1, 128, 64)
else:
    xTrain = xTrain.reshape(xTrain.shape[0], 128, 64, 1)
    xTest = xTest.reshape(xTest.shape[0], 128, 64, 1)
    print(xTrain.shape, " ", yTrain.shape)
    input_shape = (128, 64, 1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                 activation='relu', input_shape=input_shape))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dense(15, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.005),
              metrics=['accuracy'])

his = model.fit(xTrain, yTrain,
          batch_size=50,
          epochs=90,
          verbose=1,
          validation_data=(xTest, yTest))

model.save("CNNmodel.h5")

plt.plot(his.history['acc'])
plt.plot(his.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(his.history['loss'])
plt.plot(his.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
