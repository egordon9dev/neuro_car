#!/usr/bin/env python3
import data.load_data as load_data
import keras
from keras.constraints import maxnorm
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np

training_enabled = True
batch_size = 128
epochs = 2
init_file_idx = 0
init_training_idx = 0

img_rows, img_cols = 72, 128
n_channels=1
ncdata = load_data.NeuroCarData("./data", init_file_idx, init_training_idx)

n_training_iterations = 10000
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


x_train = None
x_test = None
y_train = None
y_test = None
def get_more_training_data(range_idx):
    global x_train, y_train
    train_frames = [ncdata.next_data_frame(0) for i in range(n_training_frames)]
    n_ranges = len(train_frames[0].ranges)
    print("ranges num: " + str(n_ranges))
    x_train = np.array([np.array(f.image) for f in train_frames])
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    y_train = np.array([np.array(f.ranges[range_idx]) for f in train_frames])
    y_train = preprocess_labels(y_train)
    if n_channels==3:
        x_train = to_three_channels(x_train)

def get_test_data(range_idx):
    global x_test, y_test
    test_frames = [ncdata.next_data_frame(0) for i in range(n_test_frames)]
    x_test = np.array([np.array(f.image) for f in test_frames])
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    y_test = np.array([np.array(f.ranges[range_idx]) for f in test_frames])
    y_test = preprocess_labels(y_test)
    if n_channels==3:
        x_test = to_three_channels(x_test)

def train_model(range_idx):
    # pre_net = MobileNet(include_top=False, weights='imagenet', input_shape=(img_rows,img_cols,3))
    # output = pre_net.layers[-1].output
    # output = keras.layers.Flatten()(output)
    # pre_net = Model(pre_net.input, output=output)
    # for layer in pre_net.layers:
    #     layer.trainable = False

    model = None
    if training_enabled:
        try:
            model = keras.models.load_model("neurocar_model_range_" + str(range_idx) + ".h5")
        except:
            model = Sequential()
            # model.add(pre_net)
            model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_rows,img_cols, n_channels)))
            model.add(MaxPooling2D((2,2)))

            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2,2)))

            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2,2)))

            model.add(Conv2D(256, (3, 3), activation='relu'))
            model.add(Conv2D(256, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2,2)))
            
            model.add(Flatten())
            model.add(Dense(400, activation='relu'))
            model.add(Dropout(.5))
            model.add(Dense(400, activation='relu'))
            model.add(Dropout(.5))
            model.add(Dense(100, activation='relu'))
            model.add(Dense(1))

        model.compile(loss=keras.losses.mse,
                    optimizer=keras.optimizers.SGD(learning_rate=.01, momentum=.9),
                    metrics=["mae"])

        i = 0
        while i < n_training_iterations:
            n_iterations_segment = input("how many iterations should we run?")
            if n_iterations_segment == 0:
                break
            for i in range(n_iterations_segment):
                if(not i < n_training_iterations):
                    break
                get_more_training_data(range_idx)
                model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1)
            
        print("saving model...")
        print("data location: file_idx = %d  training_idx = %d"%(ncdata.file_idx, ncdata.training_idx))
        model.save("neurocar_model_range_" + str(range_idx) + ".h5")
        print("done saving model.")
    else:
        model = keras.models.load_model("neurocar_model_range_" + str(range_idx) + ".h5")

    for i, m in enumerate(model.metrics):
        print(model.metrics_names[i] + ": " + str(m))

    get_test_data(range_idx)
    metrics = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)

    for i, m in enumerate(metrics):
        print(model.metrics_names[i] + ": " + str(m))


range_idxs = [31]
for range_idx in range_idxs:
    header = "="*20 + " "*10 + "range " + str(range_idx) + " "*10 + "="*20;
    line = "="*len(header)
    print(line + "\n" + header + "\n" + line)
    train_model(range_idx)

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