"""
This module contains our von Neumann implementation of
Algorithm 2 from Kuhn and Wattenhofer (2005).
The complexities of every instruction are written as follows:
time complexity: T-O()
space complexity: S-O()
energy complexity for von Neumann instructions: NE-O()
energy complexity for SNN instructions: SE-O()
All complexities are given in terms of the amount of nodes in the input graph.
"""

import numpy as np
import networkx as nx


def algorithm2(graph, k=10, test=False):
    """
    Implementation of Algorithm 2
    Complexity = T-O(k^2*n^2) S-O(n^2) NE-O(k^2*n^2)

    :param G: Graph
    :param k: number of iterations
    """

    # T-O(1) S-O(1) NE-O(1)
    if graph.is_directed():
        raise Exception("parameter graph should be undirected")

    # T-O(n^2) S-O(n^2) NE-O(n^2)
    graph_copy = graph
    # T-O(n^2) S-O(n) NE-O(n^2)
    delta = max([degree for _, degree in graph_copy.degree()])

    # T-O(n) S-O(1) NE-O(n)
    for node in graph_copy.nodes:
        graph_copy.nodes[node]["x"] = 0
        graph_copy.nodes[node]["color"] = "w"
        graph_copy.nodes[node]["dynamic_degree"] = graph_copy.degree(node) + 1

    # T-O(k^2*n^2) S-O(n) NE-O(k^2*n^2)
    for l in np.arange(k - 1, 0, -1):
        # T-O(k*n^2) S-O(n) NE-O(k*n^2)
        for m in np.arange(k - 1, 0, -1):
            # T-O(n) S-O(1) NE-O(n)
            graph_copy = get_x_values(delta, graph_copy, k, l, m)
            # T-O(n^2) S-O(n) NE-O(n^2)
            graph_copy = get_dynamic_degree(graph_copy)
            # T-O(n^2) S-O(n) NE-O(n^2)
            graph_copy = get_color(graph_copy)

    # T-O(n) S-O(n) NE-O(n)
    x_alphas = {n: graph_copy.nodes[n]["x"] for n in graph_copy.nodes}
    return (x_alphas, graph_copy) if test else x_alphas


def get_x_values(delta, graph, k, l, m):
    """Synchronized Step 1
    Complexity = T-O(n) S-O(1) NE-O(n)

    :param Delta: maximum degree in graph
    :param graph: graph G
    :param k: number of iterations
    :param l: value of first for-loop
    :param m: value of second for-loop
    """
    # T-O(n) S-O(1) NE-O(n)
    for node in graph.nodes:
        # T-O(1) S-O(1) NE-O(1)
        if graph.nodes[node]["dynamic_degree"] >= (delta + 1) ** (l / k):
            # T-O(1) S-O(1) NE-O(1)
            graph.nodes[node]["x"] = max(
                graph.nodes[node]["x"], 1 / ((delta + 1) ** (m / k))
            )
    return graph


def get_dynamic_degree(graph):
    """Synchronized Step 2
    Complexity = T-O(n^2) S-O(1) NE-O(n^2)

    :param graph: input graph G
    """
    # T-O(n^2) S-O(1) NE-O(n^2)
    for node in graph.nodes:
        # T-O(1) S-O(1) NE-O(1)
        graph.nodes[node]["dynamic_degree"] = graph.nodes[node]["color"] == "w"
        # T-O(n) S-O(1) NE-O(n)
        for neighbour in nx.all_neighbors(graph, node):
            # T-O(1) S-O(1) NE-O(1)
            graph.nodes[node]["dynamic_degree"] += (
                graph.nodes[neighbour]["color"] == "w"
            )
    return graph


def get_color(graph):
    """Synchronized Step 3
    Complexity = T-O(n^2) S-O(n) NE-O(n^2)

    :param graph: input graph G
    """

    # T-O(n^2) S-O(n) NE-O(n^2)
    for node in graph.nodes:
        # T-O(n) S-O(n) NE-O(n)
        if (
            sum([(graph.nodes[n]["x"]) for n in nx.all_neighbors(graph, node)])
            + graph.nodes[node]["x"]
            >= 1
        ):
            graph.nodes[node]["color"] = "g"
    return graph
