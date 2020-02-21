from urllib.request import urlretrieve

import nengo
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import msgpack
import cv2

import nengo_dl

train = "distance"
file_paths = ["neurocardataset2/training_data%d.td" %(i) for i in range(1,34)]
def trainData(indices):
    images = []
    labels = []
    for index in indices:
        file = open(file_paths[index], "rb")
        data = np.array(msgpack.load(file))
    
        for i in range(len(data)):
            # image transformation from 72x128 to 28x28
            image = np.reshape(np.array(data[i][0]), (72, 128))
            image = np.vstack((np.zeros((56, 128)), image))
            image = cv2.resize(image, (28,28), interpolation=cv2.INTER_CUBIC)
            images.append(image.flatten())
            # data[i][x] where x=1 is laser distances, x=2 is velocity, x=3 is direction
            if train == "distance":
                labels.append(round(min(data[i][1]) * 5))
            elif train == "velocity":
                labels.append(data[i][2])
            elif train == "direction":
                labels.append(max(0, int(data[i][3] * 10 + 15))) #(data[i][2], data[i][3]))
    
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

train_images, train_labels = trainData(range(12))
test_images, test_labels = trainData([12])

""" plt.figure()
plt.imshow(np.reshape(train_images[0], (28, 28)),
               cmap="gray")
plt.axis('off')
plt.title(str(train_labels[0]));
plt.show() """

with nengo.Network(seed=0) as net:
    # set some default parameters for the neurons that will make
    # the training progress more smoothly
    net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
    net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    net.config[nengo.Connection].synapse = None
    neuron_type = nengo.LIF(amplitude=0.01)

    # this is an optimization to improve the training speed,
    # since we won't require stateful behaviour in this example
    nengo_dl.configure_settings(stateful=False)

    # the input node that will be used to feed in input images
    inp = nengo.Node(np.zeros(28 * 28))

    # add the first convolutional layer
    x = nengo_dl.Layer(tf.keras.layers.Conv2D(
        filters=32, kernel_size=3))(inp, shape_in=(28, 28, 1)) # (72, 128, 1)
    x = nengo_dl.Layer(neuron_type)(x)

    # add the second convolutional layer
    x = nengo_dl.Layer(tf.keras.layers.Conv2D(
        filters=64, strides=2, kernel_size=3))(x, shape_in=(26, 26, 32)) # 5/8 - 282240 (70, 126, 32)
    x = nengo_dl.Layer(neuron_type)(x)

    # add the third convolutional layer
    x = nengo_dl.Layer(tf.keras.layers.Conv2D(
        filters=128, strides=2, kernel_size=3))(x, shape_in=(12, 12, 64)) # 3/8 134912 (34, 62, 64)
    x = nengo_dl.Layer(neuron_type)(x)

    # linear readout
    unitRange = max(train_labels) + 1
    out = nengo_dl.Layer(tf.keras.layers.Dense(units=unitRange))(x)

    # we'll create two different output probes, one with a filter
    # (for when we're simulating the network over time and
    # accumulating spikes), and one without (for when we're
    # training the network using a rate-based approximation)
    out_p = nengo.Probe(out, label="out_p")
    out_p_filt = nengo.Probe(out, synapse=0.1, label="out_p_filt")


minibatch_size = 500
sim = nengo_dl.Simulator(net, minibatch_size=minibatch_size)


# add single timestep to training data
train_images = train_images[:, None, :]
train_labels = train_labels[:, None, None]

# when testing our network with spiking neurons we will need to run it
# over time, so we repeat the input/target data for a number of
# timesteps.
n_steps = 30
test_images = np.tile(test_images[:, None, :], (1, n_steps, 1))
test_labels = np.tile(test_labels[:, None, None], (1, n_steps, 1))


def classification_accuracy(y_true, y_pred):
    return tf.metrics.sparse_categorical_accuracy(
        y_true[:, -1], y_pred[:, -1])

train = True
if train:
    sim.compile(loss={out_p_filt: classification_accuracy})
    print("accuracy before training:", sim.evaluate(test_images, {out_p_filt: test_labels}, verbose=0)["loss"])
    sim.compile(optimizer=tf.optimizers.RMSprop(0.001), loss={out_p: tf.losses.SparseCategoricalCrossentropy(from_logits=True)})
    sim.fit(train_images, {out_p: train_labels}, epochs=10)

    sim.save_params("./neuro_params")
else:
    sim.load_params("./neuro_params")


sim.compile(loss={out_p_filt: classification_accuracy})
print("accuracy after training:", sim.evaluate(test_images, {out_p_filt: test_labels}, verbose=0)["loss"])


data = sim.predict(test_images[:minibatch_size])

""" for i in [0]:#[0, 50, 100, 150, 199]:
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(test_images[i, 0].reshape((28, 28)), cmap="gray")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.plot(data[out_p_filt][i])
    plt.legend([str(i) for i in range(10)], loc="upper left")
    plt.xlabel("timesteps")
plt.show() """

sim.close()
print("Done")