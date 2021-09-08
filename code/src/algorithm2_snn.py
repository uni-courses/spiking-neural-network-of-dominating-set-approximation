"""
This module contains our spiking implementation of
Algorithm 2 from Kuhn and Wattenhofer (2005).
The complexities of every instruction are written as follows:
time complexity: T-O()
space complexity: S-O()
energy complexity for von Neumann instructions: NE-O()
energy complexity for SNN instructions: SE-O()
All complexities are given in terms of the amount of nodes in the input graph.
"""

from code.src.pySimulator.networks import Network
from code.src.pySimulator.simulators import Simulator
from code.src.common_operations import spiking_max, spiking_degree


def algorithm2(graph, k=10, test=False):
    """Executes the methods for algorithm 2 in accompanying order
    Complexity = T-O(k^2*n^2) S-O(n^2) NE-O(K^2*n^2) SE-O(K^2*n^2)

    :param graph: input graph G
    :param k: number of iterations (Default value = 10)

    """
    # T-O(1) S-O(1) E-O(1)
    if graph.is_directed():
        raise Exception("parameter graph should be undirected")

    # T-O(n^2) S-O(n) NE-O(n^2)
    delta = max([degree for _, degree in graph.degree()])
    # T-O(n) S-O(n) NE-O(n)
    nodes = list(graph.nodes)
    # T-O(n^2) S-O(1) NE-O(n^2)
    graph = graph.to_directed()
    # T-O(n^2) S-O(n^2) NE-O(n^2)
    bi_edges = list(graph.edges)

    # T-O(n^2) S-O(n^2) SE-O(n^2)
    # Calculate the size of the neighbourhood for every neuron
    deg = spiking_degree(nodes, bi_edges, delta)

    # T-O(n) S-O(n) NE-O(n)
    # add 1 to deg to convert to dyn vals
    dyn_deg_vals = {n: deg[n] + 1 for n in nodes}

    # T-O(n) S-O(n) NE-O(n)
    # initialize x_vals
    x_vals = {n: 0 for n in nodes}

    # T-O(n) S-O(n) NE-O(n)
    # initialize color_vals
    color_vals = {n: 1 for n in nodes}

    # T-O(k^2*n^2) S-O(n^2) NE-O(K^2*n^2) SE-O(K^2*n^2)
    # Perform the for loop as described in algorithm 2
    for l in range(k - 1, 0, -1):
        # T-O(k*n^2) S-O(n^2) NE-O(k*n^2) SE-O(k*n^2)
        for m in range(k - 1, 0, -1):
            # T-O(1) S-O(1) NE-O(1) SE-O(1)
            lk = (delta + 1) ** (l / k)
            # T-O(1) S-O(1) NE-O(1) SE-O(1)
            mk = 1 / ((delta + 1) ** (m / k))
            # T-O(n^2) S-O(n^2) NE-O(n^2) SE-O(n^2)
            dyn_deg_vals, x_vals, color_vals = spiking_update(
                bi_edges,
                nodes,
                lk,
                mk,
                dyn_deg_vals,
                x_vals,
                color_vals,
                delta,
            )

    return (dyn_deg_vals, x_vals, color_vals) if test else x_vals


def spiking_update(edges, nodes, lk, mk, dyna_deg_vals, x_vals, color_vals, delta, test=False):
    """'
    updates the values of x, then stays the same or gets increased
    degrees are adjusted (most likely) gradually decrease
    colours (start all white, then can change to gray but canNOT revert to white
    Complexity = T-O(n^2) S-O(n^2) NE-O(n^2) SE-O(n^2)

    :param edges: list of edges of the input graph G
    :param nodes: list of nodes of the input graph G
    :param lk: value of (delta + 1) ** (l / k)
    :param mk: 1 / ((delta + 1) ** (m / k))
    :param dyna_deg_vals: dynamic degree
    :param x_vals: x values of all nodes that will determine the colour of the node
    :param color_vals: x values that determine whether the node is in DS
    :param delta: maximum degree of nodes in graph G

    """

    # T-O(1) S-O(1) NE-O(1)
    net = Network()
    # T-O(1) S-O(1) NE-O(1)
    sim = Simulator(net)

    # T-O(1) S-O(1) NE-O(1)
    # create singleton neuron
    singleton = net.createInputTrain([1], loop=False)

    # T-O(n) S-O(n) NE-O(n) SE-O(1)
    # Create nodes for all the variables
    x = {n: net.createLIF(ID=n, m=1.0, V_init=0, thr=delta + 1) for n in nodes}
    # T-O(n) S-O(n) NE-O(n) SE-O(1)
    dyna_deg = {n: net.createLIF(ID=n, m=1.0, V_init=0, thr=delta + 2) for n in nodes}
    # T-O(n) S-O(n) NE-O(n) SE-O(1)
    # Color is represented as a ConstFire neuron if white and a silent neuron if grey
    color = {n: net.createLIF(ID=n, m=1.0, thr=1, I_e=color_vals[n]) for n in nodes}

    # T-O(n) S-O(n) NE-O(n) SE-O(n^2)
    # Check if-statement, and activate/inhibit the correct x calculator neuron
    dyna_deg_check = {
        n: net.createLIF(ID=n, m=1.0, V_init=dyna_deg_vals[n], thr=lk) for n in nodes
    }

    # T-O(n) S-O(n) NE-O(n) SE-O(n)
    # Update x
    calc_x = {
        n: net.createLIF(
            ID=n, m=1.0, V_init=0, thr=1, amplitude=spiking_max(x_vals[n], mk)
        )
        for n in nodes
    }

    # T-O(n) S-O(n) NE-O(n) SE-O(1)
    # Keep old x
    old_x = {
        n: net.createLIF(ID=n, m=1.0, V_init=0, thr=1, amplitude=x_vals[n])
        for n in nodes
    }
    # T-O(n) S-O(n) NE-O(n)
    _singleton_old_x_synapses = {
        n: net.createSynapse(singleton, old_x[n], w=1.0, d=1) for n in nodes
    }

    # T-O(n) S-O(n) NE-O(n)
    # Activate the update x neurons if the if-statement is met
    _dyna_deg_check_calc_x_synapses = {
        n: net.createSynapse(dyna_deg_check[n], calc_x[n], w=1.0, d=1) for n in nodes
    }

    # T-O(n) S-O(n) NE-O(n)
    # Inhibit the old x neurons if the if-statement is met
    _dyna_deg_check_old_x_synapses = {
        n: net.createSynapse(dyna_deg_check[n], old_x[n], w=-1.0, d=1) for n in nodes
    }

    # T-O(n) S-O(n) NE-O(n) save value of x_i
    _calc_x_x_synapses = {
        n: net.createSynapse(calc_x[n], x[n], w=1, d=1) for n in nodes
    }
    # T-O(n) S-O(n) NE-O(n)
    _old_x_x_synapses = {n: net.createSynapse(old_x[n], x[n], w=1, d=1) for n in nodes}

    # T-O(n^2) S-O(n^2) NE-O(n^2)
    # Calculate the dynamic degrees by adding the colors
    _color_dyn_deg_synapses = {
        e[0] + e[1]: net.createSynapse(color[e[0]], dyna_deg[e[1]], w=1.0, d=1)
        for e in edges
    }
    # T-O(n) S-O(n) NE-O(n)
    _recurrent_color_dyn_deg_synapses = {
        n: net.createSynapse(color[n], dyna_deg[n], w=1.0, d=1) for n in nodes
    }

    # T-O(n) S-O(n) NE-O(n) SE-O(1)
    # Calculate if the color needs to be updated by adding x's and firing if they add up to 1
    color_check = {n: net.createLIF(ID=n, m=1.0, V_init=0, thr=1) for n in nodes}

    # T-O(n^2) S-O(n^2) NE-O(n^2)
    _calc_x_color_check_synapses = {
        e[0] + e[1]: net.createSynapse(calc_x[e[0]], color_check[e[1]], w=1.0, d=1)
        for e in edges
    }
    # T-O(n^2) S-O(n^2) NE-O(n^2)
    _old_x_color_check_synapses = {
        e[0] + e[1]: net.createSynapse(old_x[e[0]], color_check[e[1]], w=1.0, d=1)
        for e in edges
    }

    # T-O(n) S-O(n) NE-O(n)
    _recurrent_calc_x_color_check_synapses = {
        n: net.createSynapse(calc_x[n], color_check[n], w=1.0, d=1) for n in nodes
    }
    # T-O(n) S-O(n) NE-O(n)
    _recurrent_calc_x_color_check_synapses = {
        n: net.createSynapse(old_x[n], color_check[n], w=1.0, d=1) for n in nodes
    }

    # T-O(n) S-O(n) NE-O(n) SE-O(1)
    # Create nodes that silences the color (and thus turning a node grey)
    color_silencer = {n: net.createLIF(ID=n, m=1.0, V_reset=1, thr=1) for n in nodes}

    # T-O(n) S-O(n) NE-O(n)
    _color_check_color_silencer_synapses = {
        n: net.createSynapse(color_check[n], color_silencer[n], w=1.0, d=1)
        for n in nodes
    }

    # T-O(n) S-O(n) NE-O(n) Update the color
    _color_silencer_color_synapses = {
        n: net.createSynapse(color_silencer[n], color[n], w=-1.0, d=1) for n in nodes
    }

    if test:
        return net, sim
    else:
        # T-O(1) S-O(1) NE-O(1) SE-O(n^2)
        # run 2 steps for dynamic degree updating
        sim.run(2)
        # T-O(n) S-O(n) NE-O(n)
        dyna_deg_vals = {n: dyna_deg[n].V for n in nodes}
        # T-O(1) S-O(1) NE-O(1) SE-O(n^2)
        # run another 3 steps for color updating
        sim.run(3)
        # T-O(n) S-O(n) NE-O(n)
        x_vals = {n: x[n].V for n in nodes}
        # T-O(n) S-O(n) NE-O(n)
        color_vals = {n: color[n].out for n in nodes}
        return dyna_deg_vals, x_vals, color_vals
