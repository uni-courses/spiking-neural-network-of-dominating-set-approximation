import os
from pathlib import Path

from code.src.pySimulator.nodes import LIF, InputTrain, RandomSpiker

import networkx as nx
import matplotlib.pyplot as plt


def plot_graph(folder_name, G, n, p, seed, x_DS=None, isDominatingSet=None):
    """
    plot graph before algorithm with result of is dominating set

    :param folder_name:
    :param G:
    :param n:
    :param p:
    :param seed:
    :param x_DS:  (Default value = None)
    :param isDominatingSet:  (Default value = None)
    """

    create_dir_if_not_exists(
        f"{os.path.dirname(__file__)}/../../../latex/project1/Images/{folder_name}/"
    )
    options = {"with_labels": True, "node_color": "white", "edgecolors": "blue"}
    nx.draw_networkx(G, **options)
    if (x_DS is None) and (isDominatingSet is None):
        # plot graph before algorithm
        filename = f"n{n}p{p}seed{seed}"
        plt.savefig(
            f"{os.path.dirname(__file__)}/../../../latex/project1/Images/{folder_name}/{filename}.png"
        )
        plt.clf()
    else:
        # plot after algorithm
        filename = (
            f"n{n}p{p}seed{seed}is_dominating_set{isDominatingSet}:{x_DS}".replace(
                ":", "_"
            )
        )
        filename = filename.replace(",", "-")
        plt.savefig(
            os.path.dirname(__file__)
            + f"/../../../latex/project1/Images/{folder_name}/{filename}.png"
        )
        plt.clf()


def plot_manual_graph(custom_filename, folder_name, G):
    """
    plot graph before algorithm with result of is dominating set

    :param custom_filename:
    :param folder_name:
    :param G:
    """

    create_dir_if_not_exists(
        f"{os.path.dirname(__file__)}/../../../latex/project1/Images/{folder_name}/"
    )
    options = {"with_labels": True, "node_color": "white", "edgecolors": "blue"}
    nx.draw_networkx(G, **options)
    plt.savefig(
        os.path.dirname(__file__)
        + f"/../../../latex/project1/Images/{folder_name}/{custom_filename}.png"
    )
    plt.clf()


def create_dir_if_not_exists(path):
    """
    :param path:
    """

    Path(path).mkdir(parents=True, exist_ok=True)


def get_node_from_ID(ID, node_type, nodes):
    """
    :param ID:
    :param node_type:
    :param nodes:
    """

    for node in nodes:
        # isinstance(node, InputTrain)
        if isinstance(node, node_type):
            if node.ID == ID:
                return node
    raise Exception(f"Node with ID={ID} was not found")


def get_InputTrain_nodes(net):
    """
    Gets all the nodes of type InputTrain from a network

    :param net:
    """

    InputTrain_nodes = []
    for node in net.nodes:
        if isinstance(node, InputTrain):
            InputTrain_nodes.append(node)
    return InputTrain_nodes


def get_LIF_nodes(net):
    """
    Gets all the nodes of type LIF from a network

    :param net:
    """

    LIF_nodes = []
    for node in net.nodes:
        if isinstance(node, LIF):
            LIF_nodes.append(node)
    return LIF_nodes


def get_RandomSpiker_nodes(net):
    """
    Gets all the nodes of type RandomSpiker from a network

    :param net:
    """

    RandomSpiker_nodes = []
    for node in net.nodes:
        if isinstance(node, RandomSpiker):
            RandomSpiker_nodes.append(node)
    return RandomSpiker_nodes


def delete_file_if_exists(filename):
    """
    Deletes files if they exist

    :param filename: name of file that will be deleted if it exists in the root of this repository
    """

    try:
        os.remove(filename)
    except:
        print(
            f"Error while deleting file: {filename} but that is not too bad because the intention is for that byproduct to not be there."
        )


# checks if file exists
def file_exists(str):
    my_file = Path(str)
    if my_file.is_file():
        # file exist
        return True
    else:
        return False


def create_manual_test_graph():
    """
    creates manual test graph with 7 undirected nodes.
    """

    graph = nx.Graph()
    graph.add_nodes_from(
        ["a", "b", "c", "d", "e", "f", "g"],
        x=0,
        color="w",
        dynamic_degree=0,
        delta_two=0,
        p=0,
        xds=0,
    )
    graph.add_edges_from(
        [
            ("a", "b"),
            ("a", "c"),
            ("b", "c"),
            ("b", "d"),
            ("c", "d"),
            ("d", "e"),
            ("b", "e"),
            ("b", "f"),
            ("f", "g"),
        ]
    )
    return graph


def create_graph_abcd():
    """
    creates manual test graph with 7 undirected nodes.
    """
    G = nx.Graph()
    G.add_nodes_from(
        ["a", "b", "c", "d"], x=0, color="w", dynamic_degree=0, delta_two=0, p=0, xds=0
    )
    G.add_edges_from([("a", "b"), ("a", "c"), ("b", "c"), ("c", "d")])
    return G
