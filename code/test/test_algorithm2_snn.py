import unittest

from code.src.algorithm2_snn import algorithm2 as algorithm2_snn
from code.src.algorithm2_neumann import algorithm2 as algorithm2_neumann
from code.test.helper import plot_graph, plot_manual_graph, create_graph_abcd

import networkx as nx
import numpy as np


class TestMain(unittest.TestCase):
    """ """

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # integration test on Alg2
    def test_receipe_integration_manual_graph(self):
        """
        asserts the output of the generalized snn implementation of Algorithm 2 is identical to the output
        of the manual/particular ssn implementation of Algorithm 2
        """

        expected_dyn_deg_vals = [0, 0, 1.0, 1.0]
        expected_x_vals = [
            0.32987697769322355,
            0.32987697769322355,
            0.8705505632961241,
            0,
        ]  # unknown why this is changed
        expected_color_vals = [0, 0, 0, 1]
        IDs = ["a", "b", "c", "d"]

        # get neuman abcd graph:
        G_neumann = create_graph_abcd()
        plot_manual_graph(
            "neumann_x_values",
            "Verified_graphs_alg2_snn_generalised_random_graphs",
            G_neumann,
        )

        # pass neumann graph to generalized snn
        k = 10
        (
            dyn_deg_vals_generalised,
            x_vals_generalised,
            color_vals_generalised,
        ) = algorithm2_snn(G_neumann, k, True)

        self.assertEqual(
            expected_dyn_deg_vals,
            list(map(lambda ID: dyn_deg_vals_generalised.get(ID), IDs)),
        )
        self.assertEqual(
            expected_x_vals, list(map(lambda ID: x_vals_generalised.get(ID), IDs))
        )
        self.assertEqual(
            expected_color_vals,
            list(map(lambda ID: color_vals_generalised.get(ID), IDs)),
        )

    def test_alg2_snn_integration_random_graph_against_neumann_color(self):
        """ """
        k = 10
        for n in range(10, 11):  # number of nodes
            for p in np.linspace(0.5, 1, 2):  # probability of edge creation
                for seed in range(
                    0, 2
                ):  # random seed to get the same random graph for testing
                    graph = nx.fast_gnp_random_graph(n, p, seed=seed)
                    if nx.is_connected(graph):
                        plot_graph(
                            "Verified_graphs_alg2_snn_generalised_against_neuman",
                            graph,
                            n,
                            p,
                            seed,
                        )

                        dset, G_neumann = algorithm2_neumann(
                            graph, k, True
                        )  # TODO: pass k
                        self.assertTrue(nx.is_dominating_set(graph, dset))

                        # get the snn output of generalized algo2:
                        (_, _, color_vals_generalised) = algorithm2_snn(graph, k, True)

                        expected_color_vals = []
                        for node in G_neumann.nodes:
                            expected_color_vals.append(
                                1 if G_neumann.nodes[node]["color"] == "w" else 0
                            )

                        self.assertEqual(
                            expected_color_vals,
                            list(
                                map(
                                    lambda ID: color_vals_generalised.get(ID),
                                    list(range(len(G_neumann.nodes))),
                                )
                            ),
                        )

    def test_alg2_snn_generalized_integration_random_graph_against_neumann_x_values(
        self,
    ):
        """ """
        k = 10
        for n in range(10, 11):  # number of nodes
            for p in np.linspace(0.5, 1, 2):  # probability of edge creation
                for seed in range(
                    0, 2
                ):  # random seed to get the same random graph for testing
                    graph = nx.fast_gnp_random_graph(n, p, seed=seed)
                    if nx.is_connected(graph):
                        plot_graph(
                            "Verified_graphs_alg2_snn_generalised_against_neuman",
                            graph,
                            n,
                            p,
                            seed,
                        )

                        dset, G_neumann = algorithm2_neumann(
                            graph, k, True
                        )  # TODO: pass k
                        self.assertTrue(nx.is_dominating_set(graph, dset))

                        # get the snn output of generalized algo2:
                        (_, x_vals_generalised, _) = algorithm2_snn(graph, k, True)

                        expected_x_vals = []
                        for node in G_neumann.nodes:
                            expected_x_vals.append(G_neumann.nodes[node]["x"])

                        x_vals_generalised = list(
                            map(
                                lambda ID: x_vals_generalised.get(ID),
                                list(range(len(G_neumann.nodes))),
                            )
                        )

                        print(f"expected_x_vals={expected_x_vals}")
                        print(f"x_vals_generalised={x_vals_generalised}")
                        self.assertEqual(
                            list(np.asarray(expected_x_vals, dtype=np.float16)),
                            list(np.asarray(x_vals_generalised, dtype=np.float16)),
                        )

    def test_alg2_snn_integration_random_graph_against_neumann_dynamic_degree(self):
        """ """
        k = 10  # TODO: get k for this computation from neumann
        for n in range(10, 11):  # number of nodes
            for p in np.linspace(0.5, 1, 2):  # probability of edge creation
                for seed in range(
                    0, 2
                ):  # random seed to get the same random graph for testing
                    graph = nx.fast_gnp_random_graph(n, p, seed=seed)
                    if nx.is_connected(graph):
                        plot_graph(
                            "Verified_graphs_alg2_snn_generalised_against_neuman",
                            graph,
                            n,
                            p,
                            seed,
                        )

                        dset, G_neumann = algorithm2_neumann(
                            graph, k, True
                        )  # TODO: pass k
                        self.assertTrue(nx.is_dominating_set(graph, dset))

                        # get the snn output of generalized algo2:
                        (dyn_deg_vals_generalised, _, _) = algorithm2_snn(
                            graph, k, True
                        )

                        expected_dyn_deg_vals = []
                        for node in G_neumann.nodes:
                            expected_dyn_deg_vals.append(
                                G_neumann.nodes[node]["dynamic_degree"]
                            )

                        self.assertEqual(
                            expected_dyn_deg_vals,
                            list(
                                map(
                                    lambda ID: dyn_deg_vals_generalised.get(ID),
                                    list(range(len(G_neumann.nodes))),
                                )
                            ),
                        )


if __name__ == "__main__":
    unittest.main()
