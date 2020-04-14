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

batch_size = 32
epochs = 10

img_rows, img_cols = 72, 128

n_training_frames = 5000
n_test_frames = 5000
min_range, max_range = (0.08, 100.0)

def preprocess_labels(labels):
    # clamp labels to within bounds
    labels = [min(max(l, min_range), max_range) for l in labels]
    # normalize
    labels = [(l-min_range)/(max_range-min_range) for l in labels]
    # de-nan/inf
    for i in range(len(labels)):
        if np.isnan(labels[i]) or np.isinf(labels[i]):
            labels[i] = 1
    return labels

# load training data
x_train = None
y_train = None
def get_training_data(ncdata, range_idx):
    global x_train, y_train
    x_train = []
    y_train = []
    for i in range(n_training_frames):
        frame = ncdata.next_data_frame(0)
        image = frame[0]
        image = image.reshape((image.shape[0], image.shape[1], 1))
        x_train.append(image)
        y_train.append(frame[1][range_idx])
    y_train = preprocess_labels(y_train)
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    # print statistics
    print("loaded %d images for training."%(len(x_train)))
    print("mean %.3f, stddev %.3f min: %.3f Q1: %.3f Q2: %.3f Q3: %.3f max: %.3f"%(np.mean(y_train), np.std(y_train), np.min(y_train), np.percentile(y_train,25), np.median(y_train), np.percentile(y_train,75), np.max(y_train)))

x_test = None
y_test = None
def get_testing_data(ncdata, range_idx):
    global x_test, y_test
    # load testing data
    x_test = []
    y_test = []
    for i in range(n_test_frames):
        frame = ncdata.next_data_frame(0)
        image = frame[0]
        image = image.reshape((image.shape[0], image.shape[1], 1))
        x_test.append(image)
        y_test.append(frame[1][range_idx])
    y_test = preprocess_labels(y_test)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 10:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def train_model(range_idx):
    filepath = "neurocar_model_range_" + str(range_idx) + ".h5"
    model = vgg((img_rows,img_cols,1), depth=10, num_filters=8)

    model.compile(loss=keras.losses.mse,
                optimizer=keras.optimizers.Adam(lr=lr_schedule(0)),
                metrics=["acc", "mae"])
    model.summary()
    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                monitor='val_acc',
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
    get_training_data(ncdata, range_idx)
    print("loading testing data...")
    get_testing_data(ncdata, range_idx)
    print("fitting model...")
    model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test, y_test),
                shuffle=True,
                callbacks=callbacks)


def test_model(range_idx):
    ncdata = load_data.NeuroCarData(["./data/grayscale_data.td"])
    print("loading testing data...")
    get_testing_data(ncdata, range_idx)
    filepath = "neurocar_model_range_" + str(range_idx) + ".h5"
    model = keras.models.load_model(filepath)
    metrics = model.evaluate(x_test, y_test, verbose=1)
    for i, m in enumerate(metrics):
        print(model.metrics_names[i] + ": " + str(m))

range_idxs = [0,1,35,36]
training_enabled = False
for range_idx in range_idxs:
    if training_enabled:
        train_model(range_idx)
    else:
        test_model(range_idx)
