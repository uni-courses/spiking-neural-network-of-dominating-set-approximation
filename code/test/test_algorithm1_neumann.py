import unittest

from code.src.algorithm1_neumann import algorithm1
from code.test.helper import plot_graph, create_manual_test_graph

import networkx as nx
import numpy as np


class TestMain(unittest.TestCase):
    """ Object used to test the SNN implementation of alg 1 of Kuhn."""

    def __init__(self, *args, **kwargs):
        """ Initialises the test class """
        super().__init__(*args, **kwargs)

    def test_algorithm1(self):
        """
        Creates a graph and generates some alpha values, then runs algorithm 1
        with that graph and verifies whether the returned nodes from a dominating set.
        """

        # create graph
        graph = create_manual_test_graph()
        plot_graph("Verified_graphs_alg1_snn", graph, len(graph.nodes), 0, 0)

        # create x_alphas
        x_alphas = {n: 0.2 for n in list(graph.nodes)}
        x_ds = algorithm1(graph, x_alphas)
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

    def test_algorithm1_random_graph(self):
        """
        Generates random graphs and verifies the SNN implementation
        of algorithm 1 returns a dominating set for each graph.
        """
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
