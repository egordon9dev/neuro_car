#!/usr/bin/env python3
import data.load_data as load_data
import keras
from keras.constraints import maxnorm
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPooling2D, Input, Add, AveragePooling2D
from keras import backend as K
import numpy as np
from model import resnet_v1, vgg
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau

# This script trains ResNet to approximate the laser range 
# values given the input image (as a 128 X 72 grayscale image)
# NOTE: training this model requires at least 18 GB of memory

batch_size = 128
epochs = 200

img_rows, img_cols = 72, 128

n_training_frames = 5000
n_test_frames = 5000
min_range, max_range = (0.08, 100.0)

def preprocess_label(l):
    # clamp labels to within bounds
    l = min(max(l, min_range), max_range)
    # normalize
    l = (l-min_range)/(max_range-min_range)
    # de-nan/inf
    if np.isnan(l) or np.isinf(l):
        l = 1
    return l

# load training data
x_train = None
y_train = None
def get_training_data(ncdata, range_idxs):
    global x_train, y_train
    x_train = []
    y_train = []
    for i in range(n_training_frames):
        frame = ncdata.next_data_frame(0)
        image = np.array(frame[0])[:,:,None].repeat(3, axis=2)
        x_train.append(image)
        y_train.append([preprocess_label(frame[1][i]) for i in range_idxs])
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    # print statistics
    print("loaded %d images for training."%(len(x_train)))
    print("mean %.3f, stddev %.3f min: %.3f Q1: %.3f Q2: %.3f Q3: %.3f max: %.3f"%(np.mean(y_train), np.std(y_train), np.min(y_train), np.percentile(y_train,25), np.median(y_train), np.percentile(y_train,75), np.max(y_train)))


x_test = None
y_test = None
def get_testing_data(ncdata, range_idxs):
    global x_test, y_test
    # load testing data
    x_test = []
    y_test = []
    for i in range(n_test_frames):
        frame = ncdata.next_data_frame(0)
        image = np.array(frame[0])[:,:,None].repeat(3, axis=2)
        x_test.append(image)
        y_test.append([preprocess_label(frame[1][i]) for i in range_idxs])
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 0
    if epoch <= 25:
      lr = 1e-3
    elif epoch <= 65:
      lr = 1e-4
    elif epoch <= 125:
      lr = 1e-5
    else:
      lr = 5e-6
    print('Learning rate: ', lr)
    return lr


def train_model(range_idxs):
    filepath = "neurocar_model.h5"
    base_model = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(72,128,3), pooling="avg")
    x = Dense(500)(base_model.output)
    x = Dense(len(range_idxs))(x)
    model = Model(inputs=base_model.inputs, outputs = x)

    model.compile(loss=keras.losses.mse,
                optimizer=keras.optimizers.Adam(lr=lr_schedule(0)),
                metrics=["acc", "mae"])
    model.summary()
    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                verbose=1,
                                save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                cooldown=0,
                                patience=5,
                                min_lr=0.5e-4)

    callbacks = [checkpoint, lr_reducer, lr_scheduler]

    ncdata = load_data.NeuroCarData(["./data/grayscale_data.td"])
    print("loading training data...")
    get_training_data(ncdata, range_idxs)
    print("loading testing data...")
    get_testing_data(ncdata, range_idxs)
    print("fitting model...")
    model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test, y_test),
                shuffle=True,
                callbacks=callbacks)

def test_model(range_idxs):
    ncdata = load_data.NeuroCarData(["./data/grayscale_data.td"])
    print("loading testing data...")
    get_testing_data(ncdata, range_idxs)
    filepath = "ncmodel.h5"
    model = keras.models.load_model(filepath)
    metrics = model.evaluate(x_test, y_test, verbose=1)
    for i, m in enumerate(metrics):
        print(model.metrics_names[i] + ": " + str(m))

range_idxs = [0,3,33,36]
train_model(range_idxs)