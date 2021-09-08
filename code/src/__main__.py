"""
Runs the main code.

It will compile the latex report of this project to pdf.

"""
import argparse

from code.src.main import (
    compile_latex_report,
    dominating_set_snn,
    dominating_set_neumann,
)
from code.test.helper import create_manual_test_graph
from code.src.energy_analysis import energy_analysis_random_graphs, plot_3dgraph
import networkx as nx
import pickle

parser = argparse.ArgumentParser()
parser.add_argument(
    "--compile_report",
    dest="compile_report",
    action="store_true",
    help="boolean flag, determines whether the notebook will be compiled",
)
parser.add_argument(
    "--n",
    dest="neumann",
    action="store_true",
    help="boolean flag, determines whether neumann implementation is used",
)
parser.add_argument(
    "--e",
    dest="energy_analysis",
    action="store_true",
    help="boolean flag, determines whether energy analysis is executed",
)
parser.add_argument(
    "--g",
    dest="graph_from_file",
    action="store_true",
    help="boolean flag, determines whether energy analysis graph is created from file ",
)
parser.add_argument(
    "--k", type=int, nargs="?", help="Provide constant k for algorithm 2"
)
parser.add_argument("infile", nargs="?", type=argparse.FileType("r"))

parser.set_defaults(
    compile_report=False,
    k=10,
    infile=None,
    neumann=False,
    energy_analysis=False,
    graph_from_file=False,
)
args = parser.parse_args()

if args.infile and not args.graph_from_file:
    try:
        graph = nx.read_graphml(args.infile.name)
    except Exception as exc:
        raise Exception(
            "Supplied input file is not a gml networkx graph object."
        ) from exc
else:
    graph = create_manual_test_graph()

if args.neumann:
    dominating_set = dominating_set_neumann(graph=graph, k=args.k)
else:
    dominating_set = dominating_set_snn(graph=graph, k=args.k)

if not nx.is_dominating_set(graph, dominating_set):
    raise Exception("An error occured during calculation, no dominating set found.")
else:
    print(dominating_set)

if args.energy_analysis:
    energy_analysis_random_graphs()

if args.graph_from_file:
    with open(args.infile.name, "rb") as fp:
        plot_3dgraph(*pickle.load(fp))

if args.infile:
    args.infile.close()

if args.compile_report:
    compile_latex_report()
