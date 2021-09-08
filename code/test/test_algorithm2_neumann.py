import unittest

from code.src.algorithm2_neumann import algorithm2
from code.test.helper import plot_graph, create_manual_test_graph

import networkx as nx
import numpy as np


class TestMain(unittest.TestCase):
    """ """

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # integration test on Alg2
    def test_algorithm2_neumann_integration_manual_graph(self):
        """"""
        graph = create_manual_test_graph()
        dset = algorithm2(graph, k=100)
        self.assertTrue(nx.is_dominating_set(graph, dset))

    def test_algorithm2_neumann_integration_random_graph(self):
        """ """
        for n in range(10, 11):  # number of nodes
            for p in np.linspace(0.5, 1, 2):  # probability of edge creation
                for seed in range(
                    0, 2
                ):  # random seed to get the same random graph for testing
                    graph = nx.fast_gnp_random_graph(n, p, seed=seed)
                    print(f"connected={nx.is_connected(graph)}")
                    if nx.is_connected(graph):
                        plot_graph("Verified_graphs_alg2_neumann", graph, n, p, seed)

                        dset = algorithm2(graph, k=100)
                        self.assertTrue(nx.is_dominating_set(graph, dset))


if __name__ == "__main__":
    unittest.main()
