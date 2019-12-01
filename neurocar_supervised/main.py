#!/usr/bin/env python3
import numpy as np
import math
import nengo
import msgpack
import random
from collections import deque

# training_data1 = None
# training_data2 = None
# with open("training_data_1000s_56k.td", "rb") as data_file:
#     training_data1 = msgpack.load(data_file)
# with open("training_data_10s_runs_26k.td", "rb") as data_file:
#     training_data2 = msgpack.load(data_file)
# training_data = training_data1 + training_data2
# print("%d + %d = %d ?" % (len(training_data1), len(training_data2), len(training_data)))
training_data = []
file_idx = 0
file_paths = ["neurocardataset1/training_data%d.td"%(i) for i in range(1,31)] + ["neurocardataset2/training_data%d"%(i) for i in range(1,34)]

def load_new_data_file():
    global training_data, file_idx
    with open(file_paths[file_idx], "rb") as file:
        training_data = msgpack.unpack(file)
    file_idx = (file_idx+1) % len(file_paths)

load_new_data_file()
    

if len(training_data) == 0:
    sys.exit("file reading failed")
#length of training_data: 933
#length of training_data_1000s_56k: 56228
#length of img_arr: 9216
print(len(training_data))
print(str(len(training_data[500][0])))
n_recent_actions = 1000
recent_actions = deque([(0,0)]*n_recent_actions)
recent_optimal_actions = deque([(0,0)]*n_recent_actions)
average_error = 0
optimal_action = (0,0)
last_action = (0,0)
training_idx = 0
its = 0
def get_next_data(t):
    global training_idx, training_data, its, average_error
    its += 1
    if its % 100 == 0:
        print("its: " + str(its) + " avg error: " + str(average_error))
    img_arr = training_data[training_idx][0]
    training_idx += 1
    if training_idx >= len(training_data):
        load_new_data_file()
        training_idx = 0
    return img_arr

def get_optimal_action(x):
    global training_data, training_idx, optimal_action
    optimal_action = downscale_action(np.array([training_data[training_idx][2], training_data[training_idx][3]]))
    return optimal_action

def mag(x):
    return (x[0]**2 + x[1]**2) ** 0.5

def move(t, x):
    global last_action, optimal_action, average_error
    last_action = upscale_action(x)

    current_error = mag((optimal_action[0]-x[0], optimal_action[1] - x[1]))
    alpha = 0.999
    average_error = (1-alpha) * current_error + alpha * average_error

    return x

def downscale_action(x):
    return np.array([x[0]/4, x[1]/1.5])
def upscale_action(x):
    return np.array([x[0]*4, x[1]*1.5])

class Explicit(nengo.solvers.Solver):
    def __init__(self, value, weights=False):
        super(Explicit, self).__init__(weights=weights)
        self.value = value
            
    def __call__(self, A, Y, rng=None, E=None):
        return self.value, {}
n_stim_neurons = 8000
model = nengo.Network(seed=8)
with model:
    movement = nengo.Ensemble(n_neurons=8000, dimensions=2)

    movement_node = nengo.Node(move, size_in=2, size_out=2, label='Movement')
    nengo.Connection(movement, movement_node)

    stim_ensemble = nengo.Ensemble(n_neurons=n_stim_neurons, dimensions=9216)
    stim_camera = nengo.Node(get_next_data)
    nengo.Connection(stim_camera, stim_ensemble)
    try:
        weights_trans = np.load("weights_trans.npy")
    except IOError:
        print("failed to load weights_trans file")
        weights_trans = np.zeros((n_stim_neurons, 1))
    try:
        weights_rot = np.load("weights_rot.npy")
    except IOError:
        print("failed to load weights_rot file")
        weights_rot = np.zeros((n_stim_neurons, 1))

    conn_trans = nengo.Connection(stim_ensemble, movement, function=lambda x: 0.7, learning_rule_type=nengo.PES(), transform=[[1], [0]], solver = Explicit(weights_trans))
    conn_rot = nengo.Connection(stim_ensemble, movement, function=lambda x: 0, learning_rule_type=nengo.PES(), transform=[[0], [1]], solver = Explicit(weights_rot))
    weights_trans_probe = nengo.Probe(conn_trans, "weights", sample_every=1.0)
    weights_rot_probe = nengo.Probe(conn_rot, "weights", sample_every=1.0)

    error = nengo.Ensemble(n_neurons=800, dimensions=2)
    #error = current - target
    nengo.Connection(movement_node, error)
    nengo.Connection(stim_ensemble, error, function=get_optimal_action, transform=-1)
    nengo.Connection(error[0], conn_trans.learning_rule, transform=[[1], [0]])
    nengo.Connection(error[1], conn_rot.learning_rule, transform=[[0], [1]])
simulator = nengo.Simulator(model)
simulator.run(25)
for i in range(10):
    simulator.step()
    print("idx: %d last action: %s,  optimal action: %s" % (training_idx, str(last_action), str(upscale_action(optimal_action))))
np.save("weights_trans.npy", simulator.data[weights_trans_probe][-1][0].T)
np.save("weights_rot.npy", simulator.data[weights_rot_probe][-1][1].T)
simulator.close()
