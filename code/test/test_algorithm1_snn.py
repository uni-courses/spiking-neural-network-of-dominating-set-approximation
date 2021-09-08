import unittest

from code.src.algorithm1_snn import (
    algorithm1,
    nonspiking_log,
    nonspiking_to_integer,
    spiking_degree,
    spiking_max_degree,
    spiking_minimum,
    spiking_multiplication,
)
from code.test.helper import plot_graph, create_manual_test_graph

import networkx as nx
import numpy as np


class TestMain(unittest.TestCase):
    """ Object used to test the SNN implementation of alg 1 of Kuhn."""

    def __init__(self, *args, **kwargs):
        """ Initialises the test class."""
        super().__init__(*args, **kwargs)

    def test_algorithm1(self):
        """Creates a graph and generates some alpha values, then runs algorithm 1
        with that graph and verifies whether the returned nodes from a dominating set.
        """

        # create graph
        graph = create_manual_test_graph()
        plot_graph("Verified_graphs_alg1_snn", graph, len(graph.nodes), 0, 0)

        # create x_alphas
        x_alphas = {n: 0.2 for n in list(graph.nodes)}
        x_ds = algorithm1(graph=graph, x_alphas=x_alphas)
        plot_graph(
            "Verified_graphs_alg1_snn",
            graph,
            len(graph.nodes),
            0,
            0,
            x_ds,
            nx.is_dominating_set(graph, x_ds.keys()),
        )

        self.assertTrue(nx.is_dominating_set(graph, x_ds.keys()))

    def test_spiking_degree(self):
        """Creates a graph, then runs algorithm 1 with that graph and
        verifies whether the algorithm returns the correct spiking degrees.
        """
        graph = create_manual_test_graph()
        delta = max([degree for node, degree in graph.degree()])

        expected_result = {node: float(degree) for node, degree in graph.degree()}

        graph = graph.to_directed()
        nodes = list(graph.nodes)
        edges = list(graph.edges)

        result = spiking_degree(nodes, edges, delta)
        self.assertEqual(expected_result, result)

    def test_nonspiking_to_integer(self):
        """Creates a graph, then runs algorithm 1 with that graph and
        verifies whether the algorithm returns the correct
        spiking degrees ?after converrsion to integer?.
        """
        graph = create_manual_test_graph()
        delta = max([degree for node, degree in graph.degree()])

        expected_result = dict(graph.degree())

        graph = graph.to_directed()
        nodes = list(graph.nodes)
        edges = list(graph.edges)

        result = spiking_degree(nodes, edges, delta)
        self.assertEqual(expected_result, result)

    def test_spiking_max_degree(self):
        """Creates a graph, then runs algorithm 1 with that graph and
        verifies whether the algorithm returns the correct spiking
        max degrees.
        """
        graph = create_manual_test_graph()
        delta = max([degree for node, degree in graph.degree()])
        graph = graph.to_directed()
        nodes = list(graph.nodes)
        edges = list(graph.edges)

        expected_result = {n: 5 for n in nodes}
        expected_result["g"] = 2

        degrees = spiking_degree(nodes, edges, delta)
        degrees = nonspiking_to_integer(degrees)

        result = spiking_max_degree(nodes, edges, degrees, degrees, delta)
        result = nonspiking_to_integer(degrees=result)

        for n in nodes:
            self.assertEqual(expected_result[n], result[n])

    def test_nonspiking_log(self):
        """Creates a graph, then runs algorithm 1 with that graph and
        verifies whether the algorithm returns the correct
        spiking and non-spiking max degrees after conversion from
        degrees to integer and from integer to spiking max degree.
        """
        graph = create_manual_test_graph()
        delta = max([degree for node, degree in graph.degree()])
        graph = graph.to_directed()
        nodes = list(graph.nodes)
        edges = list(graph.edges)

        degrees = spiking_degree(nodes, edges, delta)
        degrees = nonspiking_to_integer(degrees)

        delta1s = spiking_max_degree(
            nodes=nodes, edges=edges, degrees=degrees, value=degrees, delta=delta
        )
        delta1s = nonspiking_to_integer(degrees=delta1s)

        delta2s = spiking_max_degree(
            nodes=nodes, edges=edges, degrees=degrees, value=delta1s, delta=delta
        )
        delta2s = nonspiking_to_integer(degrees=delta2s)

        expected_result = {n: np.log(delta2s[n] + 1) for n in nodes}

        result = nonspiking_log(delta2s=delta2s)
        self.assertEqual(expected_result, result)

    def test_spiking_multiplication(self):
        """ Verifies the spiking multiplication function is performed succesfully."""
        graph = create_manual_test_graph()
        delta = max([degree for node, degree in graph.degree()])
        graph = graph.to_directed()
        nodes = list(graph.nodes)
        edges = list(graph.edges)
        x_alphas = {n: 0.2 for n in list(graph.nodes)}

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

        expected_result = {n: delta2_logs[n] * x_alphas[n] for n in nodes}

        result = spiking_multiplication(nodes, delta2_logs, x_alphas)
        self.assertEqual(expected_result, result)

    def test_spiking_min(self):
        """ Tests the spiking minimum function."""
        graph = create_manual_test_graph()
        delta = max([degree for node, degree in graph.degree()])
        graph = graph.to_directed()
        nodes = list(graph.nodes)
        edges = list(graph.edges)
        x_alphas = {n: 0.2 for n in list(graph.nodes)}

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

        expected_result = {p: min(1, products[p]) for p in products}

        result = spiking_minimum(multiplications=products)
        self.assertEqual(expected_result, result)

    def test_algorithm1_random_graph(self):
        """Renerates random graphs and verifies the SNN implementation
        of algorithm 1 returns a dominating set for each graph."""
        for n in range(10, 15):  # number of nodes
            for p in np.linspace(0.5, 1, 2):  # probability of edge creation
                for seed in range(
                    0, 5
                ):  # random seed to get the same random graph for testing
                    graph = nx.fast_gnp_random_graph(n, p, seed=seed)
                    if nx.is_connected(graph):
                        plot_graph("Verified_graphs_alg1_snn", graph, n, p, seed)

                        x_alphas = {n: 0.2 for n in list(graph.nodes)}
                        x_ds = algorithm1(graph=graph, x_alphas=x_alphas)
                        plot_graph(
                            "Verified_graphs_alg1_snn",
                            graph,
                            n,
                            p,
                            seed,
                            x_ds,
                            nx.is_dominating_set(graph, x_ds.keys()),
                        )

                        self.assertTrue(nx.is_dominating_set(graph, x_ds.keys()))


if __name__ == "__main__":
    unittest.main()
