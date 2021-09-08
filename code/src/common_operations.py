"""
This module contains spiking functions that are used
in both Algorithms from Kuhn and Wattenhofer (2005).
The complexities of every instruction are written as follows:
time complexity: T-O()
space complexity: S-O()
energy complexity for von Neumann instructions: NE-O()
energy complexity for SNN instructions: SE-O()
All complexities are given in terms of the amount of nodes in the input graph.
"""

from code.src.pySimulator.networks import Network
from code.src.pySimulator.simulators import Simulator


def spiking_degree(nodes, edges, delta, test=False):
    """Calculate the degree of every neuron.
    Complexity = T-O(n^2) S-O(n^2) NE-O(n^2) SE-O(n^2)
    Returns a list of neurons of length n.

    :param nodes: list of nodes of the input graph G
    :param edges: list of edges of the input graph G
    :param delta: maximum degree in input graph G
    """

    # T-O(1) S-O(1) NE-O(1) SE-O(1)
    net = Network()
    # T-O(1) S-O(1) NE-O(1) NE-O(1)
    sim = Simulator(net)

    # T-O(n) S-O(n) NE-O(n) SE-O(n^2)
    # Neurons for all nodes
    neurons = {
        n: net.createLIF(ID=n, m=1.0, V_init=delta + 2, thr=delta + 1) for n in nodes
    }

    # T-O(n^2) S-O(n^2) NE-O(n^2)
    # Synapses for all edges (bi-directional)
    _synapses = {
        e[0] + e[1]: net.createSynapse(neurons[e[0]], neurons[e[1]], w=1.0, d=1)
        for e in edges
    }

    if test:
        return net, sim
    else:
        # T-O(1) S-O(1) NE-O(1) SE-O(n^2)
        # Execute the network
        sim.run(2)
        # T-O(n) S-O(n) NE-O(n)
        # Voltage represents the degree of all nodes
        return {n: neurons[n].V for n in neurons}


def spiking_max(first, second, test=False):
    """calculation of the maximum of result and the step function in the loop in alg2.
    Calculation of maximum without a conditional.
    ((a > b) * a) + ((b > a) * b)
    (where a = first, and b = second)
    The value between the brackets is 1 or 0, and thus determines the maximum. 1 = spike
    The two parts of the calculation are represented by respectively
    'first_input_neuron' and 'second_input_neuron'
    the final maximum is then represented by the voltage of the 'max_neuron'

    The equality neuron checks whether the incoming values are equal. If they
    are equal, the max resets and the value is sent directly to the maximum neuron.

    Complexity = T-O(1) S-O(1) NE-O(1) SE-O(1)

    :param first: first value that will be compared with another value to calculate the maximum of the two
    :param second: second value that will be compared with another value to calculate the maximum of the two

    """

    if first < 0 or second < 0:
        raise Exception("Please enter a positive value.")

    # T-O(1) S-O(1) NE-O(1)
    net = Network()
    # T-O(1) S-O(1) NE-O(1)
    sim = Simulator(net)

    # T-O(1) S-O(1) NE-O(1)
    singleton = net.createInputTrain([1], loop=False, ID="singleton")

    # The first_fire and second_fire neurons spike with their respective values.
    # Their values are sent to the input neurons
    # T-O(1) S-O(1) NE-O(1) SE-O(1)
    first_fire_neuron = net.createLIF(
        ID="first_fire", m=1.0, amplitude=first + 1, thr=1
    )
    # T-O(1) S-O(1) NE-O(1) SE-O(1)
    second_fire_neuron = net.createLIF(
        ID="second_fire", m=1.0, amplitude=second + 1, thr=1
    )

    # T-O(1) S-O(1) NE-O(1)
    _synapse_first_fire = net.createSynapse(singleton, first_fire_neuron, w=1, d=1)
    # T-O(1) S-O(1) NE-O(1)
    _synapse_second_fire = net.createSynapse(singleton, second_fire_neuron, w=1, d=1)

    # The input neurons only fire if the incoming value is greater than the value of the other. So, first_input
    # only fires if the incoming spike is higher than the 'second' value. If it fires, the max_neuron receives the
    # spike with a weight of the 'first' value. This means that the max-neuron will always have the voltage of the
    # highest value of the two (as long as it does not spike, we can measure the voltage and return it).
    # T-O(1) S-O(1) NE-O(1) SE-O(1)
    first_input_neuron = net.createLIF(
        ID="first_input", m=1.0, V_init=0, thr=second + 1, amplitude=1
    )
    # T-O(1) S-O(1) NE-O(1) SE-O(1)
    second_input_neuron = net.createLIF(
        ID="second_input", m=1.0, V_init=0, thr=first + 1, amplitude=1
    )

    # Max neuron should never spike (need to be able to request the Voltage)
    # to prevent spiking when the values are equal, the threshold is set to first + second + 2. If the values are
    # equal, the max neuron needs to be reset, so that the equal neuron can send its value to the max. But, that
    # creates some problems, because if a value is zero, the equality and the max  will always reach its threshold.
    # Therefore, the addition of 1 in the weights from the fire neurons to the max and eq are necessary, so that the
    # threshold can be set higher.
    # T-O(1) S-O(1) NE-O(1) SE-O(1)
    max_neuron = net.createLIF(
        ID=f"max_neuron", m=1.0, V_init=0, thr=first + second + 2
    )

    # T-O(1) S-O(1) NE-O(1) SE-O(1)
    equality = net.createLIF(ID="equality", m=1.0, amplitude=1, thr=first + second + 2)

    # T-O(1) S-O(1) NE-O(1)
    _synapse_first = net.createSynapse(first_fire_neuron, first_input_neuron, w=1, d=1)
    # T-O(1) S-O(1) NE-O(1)
    _synapse_second = net.createSynapse(
        second_fire_neuron, second_input_neuron, w=1, d=1
    )
    # T-O(1) S-O(1) NE-O(1)
    _synapse_maximum_first = net.createSynapse(
        first_input_neuron, max_neuron, w=first + 1, d=1
    )
    # T-O(1) S-O(1) NE-O(1)
    _synapse_maximum_second = net.createSynapse(
        second_input_neuron, max_neuron, w=second + 1, d=1
    )
    # T-O(1) S-O(1) NE-O(1)
    _synapse_first_fire_equality = net.createSynapse(
        first_input_neuron, equality, w=first + 1, d=1
    )
    # T-O(1) S-O(1) NE-O(1)
    _synapse_second_fire_equality = net.createSynapse(
        second_input_neuron, equality, w=second + 1, d=1
    )

    # T-O(1) S-O(1) NE-O(1)
    _synapse_equality_max = net.createSynapse(equality, max_neuron, w=first + 1, d=1)

    if test:
        # T-O(1) S-O(1) NE-O(1)
        return net, sim
    else:
        # T-O(1) S-O(1) NE-O(1) SE-O(1)
        sim.run(5)
        return max_neuron.V - 1
