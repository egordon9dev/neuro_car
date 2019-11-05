import numpy as np
import nengo
import nengo_fpga
from nengo_fpga.networks import FpgaPesEnsembleNetwork

lastAction = (0.0, 0.0)

def act(t, x):
    # t is an unused time parameter
    # x refers to a tuple of (acceleration action, turn action)
    # values are -1, 0, 1 for each
    # (in terms of acceleration: deccelerate, none, accelerate) 
    # (in terms of turning: left, none, right)
    # should return the reward of the action committed
    acceleration_modifier, turning_modifier = x

    # temporarily print out chosen action
    global lastAction
    lastAction = x

    ## MAY NEED TO CHANGE THESE
    acceleration_rate = 1.0
    turning_rate = 1.0

    acceleration = acceleration_modifier * acceleration_rate
    turning = turning_modifier * turning_rate

    ## MAKE THE CAR DO THESE THINGS

    ## GET REWARD FOR THE STATE AND RETURN IT
    # temporarily, train the network to accelerate and turn left
    #if x[0] == 1 and x[1] == -1:
    #    return 1
    #else:
    #    return 0
    return x[0] * 10 + x[1] * -10

class InputManager:
    """
    The InputManager maintains the current input data for a neural network.
    To update the data, please call set_input_data and ensure that
    the data dimensionality is exactly the same as the previous data.
    """

    def __init__(self, data, dimensionality):
        self.data = data
        self.dimensionality = dimensionality

    def set_input_data(self, data):
        self.data = data

    def get_input_data(self, t):
        return self.data

class NeuralNet:
    """
    A NeuralNet contains a Nengo network with an FPGA implementation.
    It intends to receive data through an InputManager.
    It will select one of a few actions and execute a given function.
    """

    def __init__(self, input_manager, act_function, learning_active=1, board="pynq", learn_rate=1e-4, learn_synapse=0.030, action_threshold=0.1, init_transform=[1, 1, 1, 1, 1, 1, 1, 1, 1]):
        self.model = nengo.Network()
        self.input_manager = input_manager
        self.learning_active = learning_active
        self.board = board
        # parameters for the learning model (i.e. i don't know what they really do)
        self.learn_rate = learn_rate
        self.learn_synapse = learn_synapse
        self.action_threshold = action_threshold
        self.init_transform = init_transform

        with self.model:
            # Set up the input
            input_node = nengo.Node(self.input_manager.get_input_data)

            # Set up the movement node
            movement = nengo.Ensemble(n_neurons=100, dimensions=2)
            movement_node = nengo.Node(act_function, size_in=2, label="reward")
            nengo.Connection(movement, movement_node)

            # Create the action selection networks
            basal_ganglia = nengo.networks.actionselection.BasalGanglia(9)
            thalamus = nengo.networks.actionselection.Thalamus(9)
            nengo.Connection(basal_ganglia.output, thalamus.input)

            # Convert the selection actions to act transforms

            # Deccelerate, turn left
            nengo.Connection(thalamus.output[0], movement, transform=[[-1], [-1]])
            # Deccelerate, no turn
            nengo.Connection(thalamus.output[1], movement, transform=[[-1], [0]])
            # Deccelerate, turn right
            nengo.Connection(thalamus.output[2], movement, transform=[[-1], [1]])
            # No acceleration, turn left
            nengo.Connection(thalamus.output[3], movement, transform=[[0], [-1]])
            # No acceleration, no turn
            nengo.Connection(thalamus.output[4], movement, transform=[[0], [0]])
            # No acceleration, turn right
            nengo.Connection(thalamus.output[5], movement, transform=[[0], [1]])
            # Accelerate, turn left
            nengo.Connection(thalamus.output[6], movement, transform=[[1], [-1]])
            # Accelerate, no turn
            nengo.Connection(thalamus.output[7], movement, transform=[[1], [0]])
            # Accelerate, turn right
            nengo.Connection(thalamus.output[8], movement, transform=[[1], [1]])

            # Generate the training (error) signal
            def error_func(t, x):
                actions = np.array(x[:9])
                utils = np.array(x[9:18])
                r = x[19]
                activate = x[20]

                max_action = max(actions)
                actions[actions < self.action_threshold] = 0
                actions[actions != max_action] = 0
                actions[actions == max_action] = 1

                return activate * (
                    np.multiply(actions, (utils - r) * (1 - r) ** 5)
                    + np.multiply((1 - actions), (utils - 1) * (1 - r) ** 5)
                )

            errors = nengo.Node(error_func, size_in=21, size_out=9)
            nengo.Connection(thalamus.output, errors[:9])
            nengo.Connection(basal_ganglia.input, errors[9:18])
            nengo.Connection(movement_node, errors[19])

            # Learning on the FPGA
            adaptive_ensemble = FpgaPesEnsembleNetwork(
                self.board,
                n_neurons=100 * self.input_manager.dimensionality,
                dimensions=self.input_manager.dimensionality,
                learning_rate=self.learn_rate,
                function=lambda x: self.init_transform,
                label="pes ensemble"
            )
            
            nengo.Connection(input_node, adaptive_ensemble.input, synapse=self.learn_synapse)
            nengo.Connection(errors, adaptive_ensemble.error)
            nengo.Connection(adaptive_ensemble.output, basal_ganglia.input)

            learn_on = nengo.Node(self.learning_active)
            nengo.Connection(learn_on, errors[20])

        self.simulator = nengo_fpga.Simulator(self.model)

    def run_network(self, number_of_seconds):
        #with self.simulator:
        self.simulator.run(number_of_seconds)

    def step_network(self):
        #with self.simulator:
        self.simulator.step()

    def close_simulator(self):
        self.simulator.close()

#print("Syntax correct.")
input_manager = InputManager(np.array([1]), 1)
network = NeuralNet(input_manager, act)
network.run_network(60)
accel_total = 0.0
turn_total = 0.0
for x in range(10):
    network.step_network()
    print(lastAction)
    accel_total += lastAction[0]
    turn_total += lastAction[1]
network.close_simulator()
print("Average Acceleration Decision: ", accel_total / 10.0)
print("Average Turning Decision: ", turn_total / 10.0)