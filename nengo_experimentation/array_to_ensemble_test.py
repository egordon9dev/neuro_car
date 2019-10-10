import nengo

test_array = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

def get_output():
    return test_array

with nengo.Network() as net:
    main_ensemble = nengo.Ensemble(10, dimensions=1, size_in=1)
    input_node = nengo.Node(output=get_output())
    nengo.Connection(input_node, main_ensemble)