#!/usr/bin/env python3
# from keras.applications.resnet50 import ResNet50
# from keras.applications.mobilenet import MobileNet
# from keras.applications.vgg16 import VGG16
import data.load_data as load_data
import keras
from keras.constraints import maxnorm
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np

batch_size = 128
epochs = 2

img_rows, img_cols = 72, 128

ncdata = load_data.NeuroCarData("./data")

n_training_iterations = 5
n_training_frames = 5000
n_test_frames = 10000
min_range, max_range = (0.08, 100.0)

def preprocess_labels(labels):
    # clamp labels to within bounds
    labels = [min(max(l, min_range), max_range) for l in labels]
    # normalize
    labels = [(l-min_range)/(max_range-min_range) for l in labels]

    return labels

#copy channels across all RGB values since these images are binary
def to_three_channels(images):
    return np.array([[[[px[0], px[0], px[0]] for px in row] for row in img] for img in images])

training_enabled = False
n_channels=1
x_train = None
x_test = None
y_train = None
y_test = None
def get_more_training_data():
    global x_train, y_train
    train_frames = [ncdata.next_data_frame(0) for i in range(n_training_frames)]
    n_ranges = len(train_frames[0].ranges)
    x_train = np.array([np.array(f.image) for f in train_frames])
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    y_train = np.array([np.array(f.ranges[0]) for f in train_frames])
    y_train = preprocess_labels(y_train)
    if n_channels==3:
        x_train = to_three_channels(x_train)

def get_test_data():
    global x_test, y_test
    test_frames = [ncdata.next_data_frame(0) for i in range(n_test_frames)]
    x_test = np.array([np.array(f.image) for f in test_frames])
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    y_test = np.array([np.array(f.ranges[0]) for f in test_frames])
    y_test = preprocess_labels(y_test)
    if n_channels==3:
        x_test = to_three_channels(x_test)

# pre_net = MobileNet(include_top=False, weights='imagenet', input_shape=(img_rows,img_cols,3))
# output = pre_net.layers[-1].output
# output = keras.layers.Flatten()(output)
# pre_net = Model(pre_net.input, output=output)
# for layer in pre_net.layers:
#     layer.trainable = False
model = None
if training_enabled:
    model = Sequential()
    # model.add(pre_net)
    model.add(Conv2D(4, (3, 3), activation='relu', input_shape=(img_rows,img_cols, n_channels)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((5,5)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(1))

    model.compile(loss=keras.losses.mse,
                optimizer="rmsprop",
                metrics=["mae", "acc"])

    for i in range(n_training_iterations):
        get_more_training_data()
        model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1)

    print("saving model...")
    model.save("neurocar_model.h5")
    print("done saving model.")
else:
    model = keras.models.load_model("neurocar_model.h5")

for i, m in enumerate(model.metrics):
    print(model.metrics_names[i] + ": " + str(m))

get_test_data()
metrics = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)

for i, m in enumerate(metrics):
    print(model.metrics_names[i] + ": " + str(m))

# ----- best networks: 
# 
# ------ mae = 0.0939 (after 6 epochs) -----
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_rows,img_cols, 1)))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1))

