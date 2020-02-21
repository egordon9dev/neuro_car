#!/usr/bin/env python3


import data.load_data as load_data
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K
import numpy as np

batch_size = 128
epochs = 1

img_rows, img_cols = 72, 128

ncdata = load_data.NeuroCarData("./data")

n_training_frames = 2000
n_test_frames = 100

train_frames = [ncdata.next_data_frame(0) for i in range(n_training_frames)]
x_train = np.array([np.array(f.image) for f in train_frames])
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
y_train = np.array([np.array(f.ranges[0]) for f in train_frames])

test_frames = [ncdata.next_data_frame(0) for i in range(n_test_frames)]
x_test = np.array([np.array(f.image) for f in test_frames])
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
y_test = np.array([np.array(f.ranges[0]) for f in test_frames])

model = Sequential()

model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(img_rows,img_cols, 1)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))

model.compile(loss=keras.losses.mse,
              optimizer=keras.optimizers.RMSprop(.00000001),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])










# import numpy as np
# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from keras.datasets import mnist
# from keras import backend as K
 
# # 4. Load pre-shuffled MNIST data into train and test sets
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
 
# # 5. Preprocess input data
# X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
# X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255
 
# # 7. Define model architecture
# model = Sequential()
 
# # model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(1,28,28)))
# # model.add(Conv2D(32, (3, 3), activation='relu'))
# # model.add(MaxPooling2D(pool_size=(2,2)))
# # model.add(Dropout(0.25))
 
# # model.add(Flatten())
# # model.add(Dense(128, activation='relu'))
# # model.add(Dropout(0.5))
# # model.add(Dense(10, activation='softmax'))
 
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=(28,28,1)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))




# # 8. Compile model
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
 
# # 9. Fit model on training data
# model.fit(X_train, Y_train, 
#           batch_size=32, nb_epoch=10, verbose=1)
 
# # 10. Evaluate model on test data
# score = model.evaluate(X_test, Y_test, verbose=0)
