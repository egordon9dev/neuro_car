#!/usr/bin/env python3
import numpy as np
import math
import nengo
import pickle

training_data = []
with open("training_data", "rb") as data_file:
    training_data = pickle.load(data_file)

if len(training_data) == 0:
    sys.exit("file reading failed")
#length of training_data: 933
#length of img_arr: 9216
print(len(training_data))
print(str(len(training_data[500][0])))

optimal_action = (0,0)
last_action = (0,0)
training_idx = 0
def get_next_data(t):
    global optimal_action, training_idx, training_data
    img_arr = training_data[training_idx][0]
    optimal_action = training_data[training_idx][1]
    training_idx = (training_idx+1) % len(training_data)
    return img_arr

def move(t, x):
    global last_action
    last_action = x

model = nengo.Network(seed=8)
with model:
    movement = nengo.Ensemble(n_neurons=100, dimensions=2, radius=1.4)

    movement_node = nengo.Node(move, size_in=2, label='reward')
    nengo.Connection(movement, movement_node)

    stim_ensemble = nengo.Ensemble(n_neurons=5000, dimensions=9216, radius=4)
    stim_camera = nengo.Node(get_next_data)
    nengo.Connection(stim_camera, stim_ensemble)

    bg = nengo.networks.actionselection.BasalGanglia(3)
    thal = nengo.networks.actionselection.Thalamus(3)
    nengo.Connection(bg.output, thal.input)

    def u_fwd(x):
        return 0.8

    def u_left(x):
        return 0.6

    def u_right(x):
        return 0.7

    conn_fwd = nengo.Connection(stim_ensemble, bg.input[0], function=u_fwd, learning_rule_type=nengo.PES())
    conn_left = nengo.Connection(stim_ensemble, bg.input[1], function=u_left, learning_rule_type=nengo.PES())
    conn_right = nengo.Connection(stim_ensemble, bg.input[2], function=u_right, learning_rule_type=nengo.PES())

    nengo.Connection(thal.output[0], movement, transform=[[1], [0]])
    nengo.Connection(thal.output[1], movement, transform=[[0], [1]])
    nengo.Connection(thal.output[2], movement, transform=[[0], [-1]])

    errors = nengo.networks.EnsembleArray(n_neurons=50, n_ensembles=3)
    nengo.Connection(bg.output[0], errors.ensembles[0].neurons, transform=np.ones((50, 1)) * 4)
    nengo.Connection(bg.output[1], errors.ensembles[1].neurons, transform=np.ones((50, 1)) * 4)
    nengo.Connection(bg.output[2], errors.ensembles[2].neurons, transform=np.ones((50, 1)) * 4)
    nengo.Connection(bg.input, errors.input, transform=1)

    nengo.Connection(errors.ensembles[0], conn_fwd.learning_rule)
    nengo.Connection(errors.ensembles[1], conn_left.learning_rule)
    nengo.Connection(errors.ensembles[2], conn_right.learning_rule)
simulator = nengo.Simulator(model)
simulator.run(1000)
for i in range(10):
    simulator.step()
    print("idx: %d last action: %s,  optimal action: %s" % (training_idx, str(last_action), str(optimal_action)))
simulator.close()
