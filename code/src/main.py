from code.src.algorithm1_snn import algorithm1 as algorithm1_snn
from code.src.algorithm2_snn import algorithm2 as algorithm2_snn
from code.src.algorithm1_neumann import algorithm1 as algorithm1_neumann
from code.src.algorithm2_neumann import algorithm2 as algorithm2_neumann


def dominating_set_snn(graph=None, k=10):
    x_vals = algorithm2_snn(graph=graph, k=k)
    x_ds = algorithm1_snn(graph=graph, x_alphas=x_vals)
    dominating_set = []
    for node in x_ds:
        if x_ds[node]:
            dominating_set.append(node)
    return dominating_set


def dominating_set_neumann(graph=None, k=10):
    x_vals = algorithm2_neumann(graph=graph, k=k)
    x_ds = algorithm1_neumann(graph=graph, x_alphas=x_vals)
    dominating_set = []
    for node in x_ds:
        if x_ds[node]:
            dominating_set.append(node)
    return dominating_set
