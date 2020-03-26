#!/usr/bin/env python3
import data.load_data as load_data
import keras
from keras.constraints import maxnorm
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPooling2D, Input, Add, AveragePooling2D
from keras import backend as K
import numpy as np

training_enabled = True
batch_size = 128
epochs = 100
init_file_idx = 0
init_training_idx = 0

img_rows, img_cols = 72, 128
n_channels=1
ncdata = load_data.NeuroCarData("./data", init_file_idx, init_training_idx)

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
    x_train = np.array([np.array(f.image) for f in train_frames])
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    y_train = np.array([np.array(f.ranges[range_idx]) for f in train_frames])
    y_train = preprocess_labels(y_train)
    if n_channels==3:
        x_train = to_three_channels(x_train)
    # print("loaded %d images for training."%(len(train_frames)))
    # print("mean %.3f, stddev %.3f min: %.3f Q1: %.3f Q2: %.3f Q3: %.3f max: %.3f"%(np.mean(y_train), np.std(y_train), np.min(y_train), np.percentile(y_train,25), np.median(y_train), np.percentile(y_train,75), np.max(y_train)))

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
    model = None
    if training_enabled:
        try:
            model = keras.models.load_model("neurocar_model_range_" + str(range_idx) + ".h5")
        except:
            # 15 Layer Resnet-like network
            input_layer = Input(shape=(img_rows,img_cols, n_channels))
            x = Conv2D(16,(3,3), strides=2)(input_layer)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            y = Conv2D(16,(3,3),padding="same")(x)
            y = BatchNormalization()(y)
            y = Activation("relu")(y)
            y = Conv2D(16,(3,3),padding="same")(y)
            y = BatchNormalization()(y)
            y = Add()([x,y])
            y = Activation("relu")(y)

            y = Conv2D(32,(3,3), strides=2)(y)
            y = BatchNormalization()(y)
            y = Activation("relu")(y)


            z = Conv2D(32,(3,3),padding="same")(y)
            z = BatchNormalization()(z)
            z = Activation("relu")(z)
            z = Conv2D(32,(3,3),padding="same")(z)
            z = BatchNormalization()(z)
            z = Add()([y,z])
            z = Activation("relu")(z)

            z = Conv2D(64,(3,3), strides=2)(z)
            z = BatchNormalization()(z)
            z = Activation("relu")(z)


            w = Conv2D(64,(3,3),padding="same")(z)
            w = BatchNormalization()(w)
            w = Activation("relu")(w)
            w = Conv2D(64,(3,3),padding="same")(w)
            w = BatchNormalization()(w)
            w = Add()([z,w])
            w = Activation("relu")(w)

            w = Conv2D(128,(3,3), strides=2)(w)
            w = BatchNormalization()(w)
            w = Activation("relu")(w)

            u = Conv2D(128,(3,3),padding="same")(w)
            u = BatchNormalization()(u)
            u = Activation("relu")(u)
            u = Conv2D(128,(3,3),padding="same")(u)
            u = BatchNormalization()(u)
            u = Add()([w,u])
            u = Activation("relu")(u)

            u = Flatten()(u)
            u = Dense(200, activation="relu")(u)
            output_layer = Dense(1)(u)

            model = Model(inputs=input_layer, outputs=output_layer)


        model.compile(loss=keras.losses.mse,
                    optimizer=keras.optimizers.SGD(lr=.1,momentum=.9),
                    metrics=["mae"])

        while True:
            n_iterations = int(input("how many iterations should we run?"))
            if n_iterations == 0:
                break
            for i in range(n_iterations):
                maes = []
                for j in range(81):
                    print("data location: file_idx = %d  training_idx = %d"%(ncdata.file_idx, ncdata.training_idx))
                    get_more_training_data(range_idx)
                    history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=1,
                            verbose=1)
                    maes.append(float(history.history["mae"][-1]))
                with open("status.txt", "a") as f:
                    f.write("Done.\n")
                    f.write("Finished iteration: %d out of %d    mae: %.6f\n\n"%(i, n_iterations, np.mean(maes)))
            
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

with open("status.txt", "w") as f:
    f.write("")
range_idxs = [31]
for range_idx in range_idxs:
    header = "="*20 + " "*10 + "range " + str(range_idx) + " "*10 + "="*20;
    line = "="*len(header)
    print(line + "\n" + header + "\n" + line)
    with open("status.txt", "a") as f:
        f.write(header+"\n")
    train_model(range_idx)

# ----- best networks: 
# 


# one iteration of 100 epochs, mae = 0.004
# model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(img_rows,img_cols, n_channels)))
# model.add(MaxPooling2D((2,2)))

# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2,2)))

# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2,2)))

# model.add(Flatten())
# model.add(Dense(400, activation='relu'))
# model.add(Dropout(.3))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(1))




# ------ mae = 0.0939 (after 6 epochs) -----
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_rows,img_cols, 1)))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1))