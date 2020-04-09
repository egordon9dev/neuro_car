#!/usr/bin/env python3
import data.load_data as load_data
import keras
from keras.constraints import maxnorm
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPooling2D, Input, Add, AveragePooling2D
from keras import backend as K
import numpy as np
from model import resnet_v1
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau

training_enabled = True
batch_size = 128
epochs = 200

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
    
    
    #TODO fill the data evenly across different range values



    n_bins = 10
    for i in range(n_training_frames):
        frame = ncdata.next_data_frame(0)
        x_train.append(frame.image)
        y_train.append(frame.ranges[range_idx])
    y_train = preprocess_labels(y_train)
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    print(str(x_train)[:300])
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
        x_test.append(frame.image)
        y_test.append(frame.ranges[range_idx])
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
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def train_model(range_idx):
    model = None
    filepath = "neurocar_model_range_" + str(range_idx) + ".h5"
    if training_enabled:
        # model = keras.models.load_model(filepath)
        model = resnet_v1((img_rows,img_cols,1), 50)

        model.compile(loss=keras.losses.mse,
                    optimizer=keras.optimizers.SGD(lr=lr_schedule(0),momentum=.9),
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
                                    min_lr=0.5e-6)

        callbacks = [checkpoint, lr_reducer, lr_scheduler]

        ncdata = load_data.NeuroCarData("./data")
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
        # print("testing model...")
        # # Score trained model.
        # metrics = model.evaluate(x_test, y_test, verbose=1)
        # for i, m in enumerate(metrics):
        #     print(model.metrics_names[i] + ": " + str(m))

range_idxs = [31]
for range_idx in range_idxs:
    train_model(range_idx)