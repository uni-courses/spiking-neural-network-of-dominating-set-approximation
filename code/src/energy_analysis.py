from code.src.common_operations import spiking_max, spiking_degree
from code.src.algorithm2_snn import spiking_update
from code.src.algorithm1_snn import *

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle

import multiprocessing as mp

def my_func(inp):
    (n, k) = inp
    spikes_list_temp = []
    spikes_voltage_list_temp = []
    for p in np.linspace(0.5, 1, 2):  # probability of edge creation
        for seed in range(
            0, 2
        ):  # random seed to get the same random graph for testing
            graph = nx.fast_gnp_random_graph(n, p, seed=seed)
            if nx.is_connected(graph):
                spikes, spikes_voltage = energy_analysis_manual_graph(graph, k)
                spikes_list_temp.append(spikes)
                spikes_voltage_list_temp.append(spikes_voltage)
    return (n, k, spikes_list_temp, spikes_voltage_list_temp)

def findelement(x, n, k):
    for e in x:
        if e[0]==n and e[1]==k:
            return e


def energy_analysis_random_graphs():
    n_list = np.arange(20, 50, 1)  # number of nodes
    k_list = np.arange(2, 5, 1)  # number of iterations
    spikes_list = np.zeros((len(k_list), len(n_list)))
    spikes_voltage_list = np.zeros((len(k_list), len(n_list)))

    [(n,k) for n in n_list for k in k_list]

    pool = mp.Pool(mp.cpu_count())
    result = pool.map(my_func, [(n,k) for n in n_list for k in k_list])

    for x, n in enumerate(n_list):
        for y, k in enumerate(k_list):
            e = findelement(result, n, k)
            spikes_list_temp = e[2]
            spikes_voltage_list_temp = e[3]
            spikes_list[y, x] = np.mean(spikes_list_temp)
            spikes_voltage_list[y, x] = np.mean(spikes_voltage_list_temp)

    with open("pickled_graph_data", "wb") as fp:
        pickle.dump((n_list, k_list, spikes_list, spikes_voltage_list), fp)
    plot_3dgraph(n_list, k_list, spikes_list, spikes_voltage_list)


def plot_3dgraph(n, k, spikes, spikes_voltage):
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.view_init(elev=10)

    # Plot the surfaces
    n, k = np.meshgrid(n, k)
    spikes_surface = ax.plot_surface(
        n, k, spikes, cmap=plt.get_cmap("winter"), linewidth=0, alpha=0.75
    )
    spikes_voltage_surface = ax.plot_surface(
        n, k, spikes_voltage, cmap=plt.get_cmap("autumn"), linewidth=0, alpha=0.75
    )

    # Set axis labels:
    ax.set_xlabel("n")
    ax.set_ylabel("k")
    ax.set_zlabel("[spikes]/[voltage]")

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add a color bar which maps values to colors.
    fig.colorbar(spikes_surface, shrink=0.5, aspect=5)
    fig.colorbar(spikes_voltage_surface, shrink=0.5, aspect=5)

    plt.show()


def energy_analysis_manual_graph(G, k):
    spikes_alg2, spikes_voltage_alg2, x_vals = energy_alg2_snn(G, k=k)
    spikes_alg1, spikes_voltage_alg1 = energy_alg1_snn(G, x_vals)

    return spikes_alg1 + spikes_alg2, spikes_voltage_alg1 + spikes_voltage_alg2


def energy_alg1_snn(graph, x_alphas):
    s1, sv1 = energy_spiking_degree(graph)
    s2, sv2 = energy_spiking_max_degree(graph)
    s3, sv3 = energy_spiking_multiplication(graph, x_alphas)
    s4, sv4 = energy_spiking_minimum(graph, x_alphas)
    s5, sv5 = energy_spiking_sampling(graph, x_alphas)
    s6, sv6 = energy_spiking_summation(graph, x_alphas)

    spikes = s1 + s2 + s3 + s4 + s5 + s6
    spikes_voltage = sv1 + sv2 + sv3 + sv4 + sv5 + sv6

    return spikes, spikes_voltage


def energy_alg2_snn(graph, k):
    delta = max([degree for node, degree in graph.degree()])
    nodes = list(graph.nodes)
    graph = graph.to_directed()
    bi_edges = list(graph.edges)

    # Calculate the size of the neighbourhood for every neuron
    deg = spiking_degree(nodes, bi_edges, delta)
    spikes, spikes_voltage = energy_spiking_degree(graph)

    # add 1 to deg to convert to dyn vals
    dyn_deg_vals = {n: deg[n] + 1 for n in nodes}

    # initialize x_vals
    x_vals = {n: 0 for n in nodes}

    # initialize color_vals
    color_vals = {n: 1 for n in nodes}

    # Perform the for loop as described in algorithm 2
    for l in range(k - 1, 0, -1):
        for m in range(k - 1, 0, -1):
            lk = (delta + 1) ** (l / k)
            mk = 1 / ((delta + 1) ** (m / k))

            # Compute energy of the spiking max function
            for n in nodes:
                net, sim = spiking_max(x_vals[n], mk, test=True)
                s_max, sv_max = compute_spike_energy(net, sim, 5)
                spikes += s_max
                spikes_voltage += sv_max

            # Run update function
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

            # Compute energy of the update function
            net, sim = spiking_update(
                bi_edges,
                nodes,
                lk,
                mk,
                dyn_deg_vals,
                x_vals,
                color_vals,
                delta,
                test=True,
            )
            s, sv = compute_spike_energy(net, sim, 5)
            spikes += s
            spikes_voltage += sv

    return spikes, spikes_voltage, x_vals


def energy_spiking_degree(graph):
    delta = max([degree for node, degree in graph.degree()])

    graph = graph.to_directed()
    nodes = list(graph.nodes)
    edges = list(graph.edges)

    net, sim = spiking_degree(nodes, edges, delta, test=True)
    nr_spikes, spikes_voltage = compute_spike_energy(net, sim, 2)

    return nr_spikes, spikes_voltage


def energy_spiking_max_degree(graph):
    delta = max([degree for node, degree in graph.degree()])

    graph = graph.to_directed()
    nodes = list(graph.nodes)
    edges = list(graph.edges)

    degrees = spiking_degree(nodes, edges, delta)
    degrees = nonspiking_to_integer(degrees)

    net, sim = spiking_max_degree(nodes, edges, degrees, degrees, delta, test=True)
    nr_spikes, spikes_voltage = compute_spike_energy(net, sim, 2 + delta)

    return nr_spikes, spikes_voltage


def energy_spiking_multiplication(graph, x_alphas):
    delta = max([degree for node, degree in graph.degree()])

    graph = graph.to_directed()
    nodes = list(graph.nodes)
    edges = list(graph.edges)

    degrees = spiking_degree(nodes=nodes, edges=edges, delta=delta)
    degrees = nonspiking_to_integer(degrees=degrees)

    delta1s = spiking_max_degree(
        nodes=nodes, edges=edges, degrees=degrees, value=degrees, delta=delta
    )
    delta1s = nonspiking_to_integer(degrees=delta1s)

    delta2s = spiking_max_degree(
        nodes=nodes, edges=edges, degrees=degrees, value=delta1s, delta=delta
    )
    delta2s = nonspiking_to_integer(degrees=delta2s)
    delta2_logs = nonspiking_log(delta2s=delta2s)

    net, sim = spiking_multiplication(nodes, delta2_logs, x_alphas, test=True)
    nr_spikes, spikes_voltage = compute_spike_energy(net, sim, 4)

    return nr_spikes, spikes_voltage


def energy_spiking_minimum(graph, x_alphas):
    delta = max([degree for node, degree in graph.degree()])

    graph = graph.to_directed()
    nodes = list(graph.nodes)
    edges = list(graph.edges)

    degrees = spiking_degree(nodes=nodes, edges=edges, delta=delta)
    degrees = nonspiking_to_integer(degrees=degrees)

    delta1s = spiking_max_degree(
        nodes=nodes, edges=edges, degrees=degrees, value=degrees, delta=delta
    )
    delta1s = nonspiking_to_integer(degrees=delta1s)

    delta2s = spiking_max_degree(
        nodes=nodes, edges=edges, degrees=degrees, value=delta1s, delta=delta
    )
    delta2s = nonspiking_to_integer(degrees=delta2s)
    delta2_logs = nonspiking_log(delta2s=delta2s)

    products = spiking_multiplication(
        nodes=nodes, delta2_logs=delta2_logs, x_alphas=x_alphas
    )
    net, sim = spiking_minimum(products, test=True)
    nr_spikes, spikes_voltage = compute_spike_energy(net, sim, 4)

    return nr_spikes, spikes_voltage


def energy_spiking_sampling(graph, x_alphas):
    delta = max([degree for node, degree in graph.degree()])

    graph = graph.to_directed()
    nodes = list(graph.nodes)
    edges = list(graph.edges)

    degrees = spiking_degree(nodes=nodes, edges=edges, delta=delta)
    degrees = nonspiking_to_integer(degrees=degrees)

    delta1s = spiking_max_degree(
        nodes=nodes, edges=edges, degrees=degrees, value=degrees, delta=delta
    )
    delta1s = nonspiking_to_integer(degrees=delta1s)

    delta2s = spiking_max_degree(
        nodes=nodes, edges=edges, degrees=degrees, value=delta1s, delta=delta
    )
    delta2s = nonspiking_to_integer(degrees=delta2s)
    delta2_logs = nonspiking_log(delta2s=delta2s)

    products = spiking_multiplication(
        nodes=nodes, delta2_logs=delta2_logs, x_alphas=x_alphas
    )

    probabilities = spiking_minimum(products)

    net, sim = spiking_sampling(probabilities=probabilities, test=True)

    nr_spikes, spikes_voltage = compute_spike_energy(net, sim, 4)

    return nr_spikes, spikes_voltage


def energy_spiking_summation(graph, x_alphas):
    delta = max([degree for node, degree in graph.degree()])

    graph = graph.to_directed()
    nodes = list(graph.nodes)
    edges = list(graph.edges)

    degrees = spiking_degree(nodes=nodes, edges=edges, delta=delta)
    degrees = nonspiking_to_integer(degrees=degrees)

    delta1s = spiking_max_degree(
        nodes=nodes, edges=edges, degrees=degrees, value=degrees, delta=delta
    )
    delta1s = nonspiking_to_integer(degrees=delta1s)

    delta2s = spiking_max_degree(
        nodes=nodes, edges=edges, degrees=degrees, value=delta1s, delta=delta
    )
    delta2s = nonspiking_to_integer(degrees=delta2s)
    delta2_logs = nonspiking_log(delta2s=delta2s)

    products = spiking_multiplication(
        nodes=nodes, delta2_logs=delta2_logs, x_alphas=x_alphas
    )

    probabilities = spiking_minimum(products)

    x_ds = spiking_sampling(probabilities=probabilities)

    net, sim = spiking_summation(edges=edges, x_ds=x_ds, test=True)
    nr_spikes, spikes_voltage = compute_spike_energy(net, sim, 2)

    return nr_spikes, spikes_voltage


def compute_spike_energy(net, sim, sim_time):
    # plot values
    for n in net.nodes:
        sim.raster.addTarget(n)

    # Initialize arrays to store spikes, voltages and amperages for all nodes
    spikes = []
    voltages = []
    post_synaptic_voltages = []

    # Simulate the network
    for iteration in range(0, sim_time + 1):
        # read out the voltages and amperages at each step of the simulation and store them to lists
        for n in net.nodes:
            spikes.append(n.V)
            post_synaptic_voltages.append(n.I)
            if iteration == 0:
                voltages.append(n.V)
        sim.run(1)

    nr_spikes = sum(s != 0 for s in spikes)
    spikes_voltage = sum(post_synaptic_voltages) + sum(voltages)

    return nr_spikes, spikes_voltage
