#!/usr/bin/env python3
import numpy as np
import math
import nengo
import msgpack

training_data = []
with open("training_data_1000s_56k", "rb") as data_file:
    training_data = msgpack.load(data_file)

if len(training_data) == 0:
    sys.exit("file reading failed")
#length of training_data: 933
#length of training_data_1000s_56k: 56228
#length of img_arr: 9216
print(len(training_data))
print(str(len(training_data[500][0])))

optimal_action = (0,0)
last_action = (0,0)
training_idx = 0
its = 0
def get_next_data(t):
    global optimal_action, training_idx, training_data, its
    its += 1
    if its % 100 == 0:
        print("its: " + str(its))
    img_arr = training_data[training_idx][0]
    optimal_action = training_data[training_idx][1]
    training_idx = (training_idx+1) % len(training_data)
    return img_arr

def move(t, x):
    global last_action
    last_action = x

model = nengo.Network(seed=8)
with model:
    movement = nengo.Ensemble(n_neurons=200, dimensions=2, radius=1.4)

    movement_node = nengo.Node(move, size_in=2, label='reward')
    nengo.Connection(movement, movement_node)

    stim_ensemble = nengo.Ensemble(n_neurons=10000, dimensions=9216, radius=4)
    stim_camera = nengo.Node(get_next_data)
    nengo.Connection(stim_camera, stim_ensemble)

    conn_trans = nengo.Connection(stim_ensemble, movement, function=lambda x: 0.5, learning_rule_type=nengo.PES(), transform=[[1], [0]])
    conn_rot = nengo.Connection(stim_ensemble, movement, function=lambda x: 0, learning_rule_type=nengo.PES(), transform=[[0], [1]])

    error = nengo.Ensemble(n_neurons=1000, dimensions=2)
    #error = current - target
    nengo.Connection(movement[0], movement_node, error)
    optimal_node
    nengo.Connectino()
simulator = nengo.Simulator(model)
simulator.run(60)
for i in range(10):
    simulator.step()
    print("idx: %d last action: %s,  optimal action: %s" % (training_idx, str(last_action), str(optimal_action)))
simulator.close()
