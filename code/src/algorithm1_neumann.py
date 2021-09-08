"""
This module contains our von Neumann implementation of
Algorithm 1 from Kuhn and Wattenhofer (2003).
The complexities of every instruction are written as follows:
time complexity: T-O()
space complexity: S-O()
energy complexity for von Neumann instructions: NE-O()
energy complexity for SNN instructions: SE-O()
All complexities are given in terms of the amount of nodes in the input graph.
"""

import numpy as np
import networkx as nx


def algorithm1(graph, x_alphas, test=False):
    """
    Implementation of Algorithm 2
    Complexity = T-O(n^2) S-O(n^2) NE-O(n^2)

    :param G: Graph
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

    # T-O(n^2) S-O(n^2) NE-O(n^2)
    graph_copy = graph

    # T-O(n) S-O(1) NE-O(1)
    for node in graph_copy.nodes:
        graph_copy.nodes[node]["x"] = x_alphas[node]

    # T-O(n^2) S-O(n^2) NE-O(n^2)
    for node in graph_copy.nodes:
        # T-O(n) S-O(n) NE-O(n)
        neighbours = list(graph_copy.neighbors(node))
        # T-O(n^2) S-O(n^2) NE-O(n^2)
        second_degree_neighbours = [list(graph_copy.neighbors(n)) for n in neighbours]
        # T-O(1) S-O(1) NE-O(1)
        second_degree_neighbours.append(neighbours)
        # T-O(n^2) S-O(n) NE-O(n^2)
        second_degree_neighbours = {
            item for sublist in second_degree_neighbours for item in sublist
        }
        # T-O(n) S-O(1) NE-O(n)
        graph_copy.nodes[node]["delta_two"] = max(
            [x[1] for x in graph_copy.degree(second_degree_neighbours)]
        )
        # T-O(1) S-O(1) NE-O(1)
        graph_copy.nodes[node]["p"] = min(
            1,
            graph_copy.nodes[node]["x"]
            * np.log(graph_copy.nodes[node]["delta_two"] + 1),
        )
        # T-O(1) S-O(1) NE-O(1)
        graph_copy.nodes[node]["xds"] = (
            np.random.random_sample() <= graph_copy.nodes[node]["p"]
        )
    # T-O(n^2) S-O(n) NE-O(n^2)
    for node in graph_copy.nodes:
        # T-O(n) S-O(n) NE-O(n)
        if not sum(
            [(graph_copy.nodes[n]["xds"]) for n in nx.all_neighbors(graph_copy, node)]
        ):
            # T-O(1) S-O(1) NE-O(1)
            graph_copy.nodes[node]["xds"] = True

    # T-O(n) S-O(n) NE-O(n)
    x_ds = {n: graph_copy.nodes[n]["xds"] for n in graph_copy.nodes}
    return (x_ds, graph_copy) if test else x_ds
