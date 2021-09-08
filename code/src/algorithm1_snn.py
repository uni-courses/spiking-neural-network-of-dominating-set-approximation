"""
This module contains our spiking implementation of
Algorithm 1 from Kuhn and Wattenhofer (2005).
The complexities of every instruction are written as follows:
time complexity: T-O()
space complexity: S-O()
energy complexity for von Neumann instructions: NE-O()
energy complexity for SNN instructions: SE-O()
All complexities are given in terms of the amount of nodes in the input graph.
"""

from code.src.pySimulator.networks import Network
from code.src.pySimulator.simulators import Simulator
from code.src.common_operations import spiking_degree

import numpy as np


def algorithm1(graph, x_alphas):
    """Returns a dictionary containing the dominating set status of all neurons
    Complexity = T-O(n^2) S-O(n^2) NE-O(n^2) SE-O(n^2)

    :param graph: input graph G
    :param x_alphas: list of numbers with length number of nodes n, alpha-approximation LP_mds of algorithm 2.

    """
    # T-O(1) S-O(1) NE-O(1)
    if graph.is_directed():
        raise Exception("parameter graph should be undirected")
    # T-O(1) S-O(1) NE-O(1)
    if not isinstance(x_alphas, dict):
        raise Exception(
            "parameter x_alphas should be a dictionary with node keys and x values"
        )

    # T-O(n^2) S-O(n) NE-O(n^2)
    delta = max([degree for node, degree in graph.degree()])
    # T-O(n^2) S-O(1) NE-O(n^2)
    graph = graph.to_directed()
    # T-O(n) S-O(n) NE-O(n)
    nodes = list(graph.nodes)
    # T-O(n^2) S-O(n^2) NE-O(n^2)
    edges = list(graph.edges)

    # T-O(n^2) S-O(n^2) NE-O(n^2) SE-O(n^2)
    degrees = spiking_degree(nodes=nodes, edges=edges, delta=delta)
    # T-O(n) S-O(1) NE-O(n)
    degrees = nonspiking_to_integer(degrees=degrees)

    # T-O(n^2) S-O(n^2) NE-O(n^2) SE-O(n^2)
    delta1s = spiking_max_degree(
        nodes=nodes, edges=edges, degrees=degrees, value=degrees, delta=delta
    )
    # T-O(n) S-O(1) NE-O(n)
    delta1s = nonspiking_to_integer(degrees=delta1s)

    # T-O(n^2) S-O(n^2) NE-O(n^2) SE-O(n^2)
    delta2s = spiking_max_degree(
        nodes=nodes, edges=edges, degrees=degrees, value=delta1s, delta=delta
    )

    # T-O(n) S-O(n) NE-O(n) SE-O(n)
    delta2_logs = nonspiking_log(delta2s=delta2s)

    # T-O(n) S-O(n) NE-O(n) SE-O(n*log n)
    products = spiking_multiplication(
        nodes=nodes, delta2_logs=delta2_logs, x_alphas=x_alphas
    )

    # T-O(n) S-O(n) NE-O(n) SE-O(n*(log n)^2)
    probabilities = spiking_minimum(products)
    # T-O(n) S-O(n) NE-O(n) SE-O(1)
    x_ds = spiking_sampling(probabilities=probabilities)
    # T-O(n^2) S-O(n^2) NE-O(n^2) SE-O(n^2)
    x_ds = spiking_summation(edges=edges, x_ds=x_ds)
    return x_ds


def nonspiking_to_integer(degrees):
    """returns dictionary of integers that represent the degree of each node.
    Complexity = T-O(n) S-O(n) E-O(n)

    :param degrees: list with the degree of each node

    """
    # T-O(n) S-O(n) E-O(n)
    return {d: int(degrees[d]) for d in degrees}


def spiking_max_degree(nodes, edges, degrees, value, delta, test=False):
    """Calculate the max degree of its 2-step neighborhood (delta2)
    Complexity = T-O(n^2) S-O(n^2) NE-O() SE-O(1)
    Returns a list of neurons of length n.

    :param nodes: list with all nodes of input graph G
    :param edges: list with all edges of input graph G
    :param degrees: list with degrees for all nodes in graph G
    :param value: list with degrees
    :param delta: maximum degree in graph G

    """
    # T-O(1) S-O(1) NE-O(1)
    net = Network()
    # T-O(1) S-O(1) NE-O(1)
    sim = Simulator(net)

    # T-O(1) S-O(1) NE-O(1)
    singleton = net.createInputTrain([1], loop=False)
    # T-O(1) S-O(1) NE-O(1) SE-O(1)
    const_fire = net.createLIF(ID="const", m=1.0, V_reset=1.0)
    # T-O(1) S-O(1) NE-O(1)
    net.createSynapse(singleton, const_fire, 1.0, 1)

    # T-O(n) S-O(n) NE-O(n) SE-O(1) Neurons for all nodes, connected to the incoming edges of the node
    in_neurons = {
        n: net.createLIF(ID=f"{n}_in", m=1.0, V_init=0, thr=degrees[n] + 1)
        for n in nodes
    }

    # T-O(n) S-O(n) NE-O(n) SE-O(n^2) Neurons for all nodes, connected to the outgoing edges of the node
    out_neurons = {
        n: net.createLIF(
            ID=f"{n}_out", m=1.0, V_init=degrees[n] + 2, thr=degrees[n] + 1
        )
        for n in nodes
    }

    # T-O(n^2) S-O(n^2) NE-O(n^2) Synapses for all edges (bi-directional)
    _synapses = {
        e[0]
        + e[1]: net.createSynapse(
            out_neurons[e[0]], in_neurons[e[1]], w=1.0, d=value[e[0]]
        )
        for e in edges
    }

    # T-O(n) S-O(n) NE-O(n) Recurrent synapses
    _recurrent_synapses = {
        n: net.createSynapse(out_neurons[n], in_neurons[n], w=1.0, d=value[n])
        for n in nodes
    }

    # T-O(n) S-O(n) NE-O(n) Counter nodes to convert temporal representation to voltage representation
    # Threshold high enough, so it never spikes
    counter_neurons = {
        n: net.createLIF(ID=f"{n}_counter", m=1.0, V_init=0, thr=delta + 2)
        for n in nodes
    }

    # T-O(n) S-O(n) NE-O(1) SE-O(1) Inhibition nodes to stop the counter
    inhibition_neurons = {
        n: net.createLIF(ID=f"{n}_inhibition", m=1.0, V_init=0, V_reset=1.0)
        for n in nodes
    }

    # T-O(n) S-O(n) NE-O(n) Synapses to start the counter
    _synapses_fire = {
        n: net.createSynapse(const_fire, counter_neurons[n], w=1.0, d=1) for n in nodes
    }

    # T-O(n) S-O(n) NE-O(n) Synapses to inhibit the counter
    _synapses_inhibit = {
        n: net.createSynapse(inhibition_neurons[n], counter_neurons[n], w=-1.0, d=1)
        for n in nodes
    }

    # T-O(n) S-O(n) NE-O(n) Synapses to start inhibiting
    _synapses_stop_counter = {
        n: net.createSynapse(in_neurons[n], inhibition_neurons[n], w=1.0, d=1)
        for n in nodes
    }

    if test:
        return net, sim
    else:
        # T-O(n) S-O(1) NE-O(n) SE-O(n^2)
        sim.run(2 + delta)
        # T-O(n) S-O(n) NE-O(n) Voltage represents the resulting product
        return {n: counter_neurons[n].V for n in nodes}


def nonspiking_log(delta2s):
    """Calculate the logarithm of the delta2 values.
    Complexity = T-O(n) S-O(n) NE-O(n)
    Returns a list of length number of neurons n

    :param delta2s: A parameter that represents the delta_2 values in Kuhns algorithm 1.

    """
    # T-O(n) S-O(n) NE-O(n)
    return {d: np.log(delta2s[d] + 1) for d in delta2s}


def spiking_multiplication(nodes, delta2_logs, x_alphas, test=False):
    """Perform an elementwise multiplication of delta2_logs and x_alphas
    Complexity = T-O(n) S-O(n) NE-O(n) SE-O(n*log n)
    Retruns a list of length number of nodes n.

    :param nodes: list with all nodes of input graph G
    :param delta2_logs:
    :param x_alphas: list of numbers with length number of nodes n, alpha-approximation LP_mds of algorithm 2.

    """
    # T-O(1) S-O(1) NE-O(1)
    net = Network()
    # T-O(1) S-O(1) NE-O(1)
    sim = Simulator(net)

    # T-O(n) S-O(n) NE-O(n) SE-O(n) Neurons representing all x values
    x_neurons = {
        x: net.createLIF(ID=x, m=1.0, V_init=1, thr=1, amplitude=x_alphas[x])
        for x in x_alphas
    }

    # T-O(n) S-O(n) NE-O(n) SE-O(1) Neurons representing all multiplication results
    mul_neurons = {n: net.createLIF(ID=n, m=1.0, V_init=0, thr=2) for n in nodes}

    # T-O(n) S-O(n) NE-O(n) Synapse performing the actual multiplication
    _synapses = {
        n + n: net.createSynapse(x_neurons[n], mul_neurons[n], w=delta2_logs[n], d=1)
        for n in nodes
    }

    if test:
        return net, sim
    else:
        # T-O(1) S-O(1) NE-O(1) SE-O(n*log n)
        # Execute the network
        sim.run(4)
        # T-O(n) S-O(n) NE-O(n)
        # Voltage represents the resulting product
        return {n: mul_neurons[n].V for n in nodes}


def spiking_minimum(multiplications, test=False):
    """Elementwise calculation of the minimum of the array elements and 1.
    Calculation of minimum without a conditional.
    ((1 > value) * value) + ((value > 1) * 1)
    value between the brackets is 1 or 0, and thus determines the minimum.
    the spikes of the mult_value_neurons represent the incoming
    multiplication values.
    two parts of the calculation are represented by respectively
    'first_neurons' and 'second_neurons'.
    the final minimum is then represented by the voltage of 'min_neurons'
    To prevent the network from continuing to spike when the threshold of a neuron is set to 0,
    which occurs if multiplications[n] = 0, the value of 1 is added to both the amplitude of the
    mult_value_neurons, as well as to the threshold of the receiving prob_second neuron.

    Complexity = T-O(n) S-O(n) NE-O(n) SE-O(n*(log n)^2)
    Returns list of length number of neurons n.

    :param multiplications: output of function spiking_multiplication

    """
    # T-O(1) S-O(1) NE-O(1)
    net = Network()
    # T-O(1) S-O(1) NE-O(1)
    sim = Simulator(net)

    # T-O(1) S-O(1) NE-O(1)
    singleton = net.createInputTrain([1], loop=False)
    # T-O(n) S-O(n) NE-O(n) SE-O(1)
    mult_value_neurons = {
        n: net.createLIF(ID=n, m=1.0, amplitude=multiplications[n] + 1)
        for n in multiplications
    }

    # T-O(n) S-O(n) NE-O(n)
    _synapses_fire = {
        n: net.createSynapse(singleton, mult_value_neurons[n], w=1.0, d=2)
        for n in mult_value_neurons
    }

    # T-O(n) S-O(n) NE-O(n) SE-O(1)
    first_neurons = {
        n: net.createLIF(ID=n, m=1.0, V_init=0, thr=multiplications[n])
        for n in multiplications
    }

    # T-O(n) S-O(n) NE-O(n) SE-O(1)
    second_neurons = {
        n: net.createLIF(ID=n, m=1.0, V_init=0, thr=2) for n in multiplications
    }

    # T-O(n) S-O(n) NE-O(n) SE-O(1)
    # never spike (threshold > 1), voltage represents the minimum value.
    min_neurons = {
        n: net.createLIF(ID=n, m=1.0, V_init=0, thr=1.1) for n in multiplications
    }

    # T-O(n) S-O(n) NE-O(n)
    _synapses_first = {
        n + n: net.createSynapse(singleton, first_neurons[n], w=1, d=1)
        for n in first_neurons
    }
    # T-O(n) S-O(n) NE-O(n)
    _synapses_second = {
        n + n: net.createSynapse(mult_value_neurons[n], second_neurons[n], w=1, d=1)
        for n in second_neurons
    }

    # T-O(n) S-O(n) NE-O(n)
    _synapse_minimum_first = {
        n
        + n: net.createSynapse(
            first_neurons[n], min_neurons[n], w=multiplications[n], d=1
        )
        for n in multiplications
    }

    # T-O(n) S-O(n) NE-O(n)
    _synapse_minimum_second = {
        n + n: net.createSynapse(second_neurons[n], min_neurons[n], w=1, d=1)
        for n in multiplications
    }

    if test:
        return net, sim
    else:
        # T-O(1) S-O(1) NE-O(1) SE-O(n*(log n)^2)
        # Execute the network
        sim.run(4)
        # T-O(n) S-O(n) NE-O(n)
        return {n: min_neurons[n].V for n in multiplications}


def spiking_sampling(probabilities, test=False):
    """create random spikers and observe whether they print a spike
    Complexity = T-O(n) S-O(n) NE-O(n) SE-O(1)
    Returns a list of raster spikes of length number of neurons n.

    :param probabilities: list of probabilities of length number of nodes n, output of spiking_minimum()

    """
    # T-O(1) S-O(1) NE-O(1)
    net = Network()
    # T-O(1) S-O(1) NE-O(1)
    sim = Simulator(net)

    # T-O(n) S-O(n) NE-O(n)
    # Neurons representing all probabilities
    neurons = {
        p: net.createRandomSpiker(ID=p, p=probabilities[p]) for p in probabilities
    }

    # T-O(n) S-O(n) NE-O(n)
    # Add nodes to the spike detector
    sim.raster.addTargets(neurons.values())

    if test:
        return net, sim
    else:
        # T-O(1) S-O(1) NE-O(n) SE-O(1)
        # Execute the network
        sim.run(4)
        # T-O(n) S-O(n) NE-O(n)
        # Spike at t=0 represents the xDS value
        return {n: sim.raster.spikes[0, idx] for idx, n in enumerate(neurons)}


def spiking_summation(edges, x_ds, test=False):
    """Send xDS to all nodes and update its value

    Complexity = T-O(n^2) S-O(n^2) NE-O(n^2) SE-O(n^2)
    Returns a list of spike rasters of length number of neurons n

    :param edges: list of edges of input graph G
    :param x_ds: values that determine whether neuron is in the dominating set.

    """
    # T-O(1) S-O(1) NE-O(1)
    net = Network()
    # T-O(1) S-O(1) NE-O(1)
    sim = Simulator(net)

    # T-O(n) S-O(n) NE-O(n) SE-O(1)
    # Neurons for all nodes
    neurons = {
        x: (
            net.createInputTrain(ID=x, train=[1], loop=True)
            if x_ds[x]
            else net.createLIF(ID=x, m=1.0, thr=1, I_e=1.0)
        )
        for x in x_ds
    }

    # T-O(n^2) S-O(n^2) NE-O(n^2)
    # Synapses for all edges (bi-directional)
    _synapses = {
        e[0]
        + e[1]: net.createSynapse(
            neurons[e[0]], neurons[e[1]], w=-1.0 * x_ds[e[0]], d=1
        )
        for e in edges
    }

    # T-O(n) S-O(n) NE-O(n)
    # Add nodes to the spike detector
    sim.raster.addTargets(neurons.values())

    if test:
        return net, sim
    else:
        #  T-O(1) S-O(1) NE-O(1) SE-O(n^2)
        #  Execute the network
        sim.run(2)
        # T-O(n) S-O(n) NE-O(n) Spike at t=1 represents the updated xDS value
        return {x: sim.raster.spikes[1, idx] for idx, x in enumerate(x_ds)}
