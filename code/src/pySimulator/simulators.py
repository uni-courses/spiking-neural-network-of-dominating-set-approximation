import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import networkx as nx
from code.src.pySimulator.detectors import Raster, Multimeter


class Simulator:
    """Simulator

    Parameters
    ----------
    network : Network
        Network to simulate
    detectors : List
        List of detectors
    """

    def __init__(self, network, detectors=None, seed=None):
        self.network = network
        self.multimeter = Multimeter()
        self.raster = Raster()
        if seed != None:
            self.network.update_rng(np.random.RandomState(seed))

    def run(self, steps, plotting=False):
        """Run the simulator

        Parameters
        ----------
        steps : int
            Number of steps to simulate
        """
        self.raster.initialize(steps)
        self.multimeter.initialize(steps)

        for _ in range(steps):
            self.network.step()
            self.raster.step()
            self.multimeter.step()

        if plotting:
            self.print_detectors(steps)
            # pass

    def to_inet_string(self):
        inet_str = ""
        inet_str += self.raster.to_inet_string() + "\n\n"
        inet_str += self.multimeter.to_inet_string() + "\n\n"
        return inet_str

    def print_detectors(self, steps=0):
        rasterdata = self.raster.get_measurements()
        print(f"rasterdata={rasterdata}")
        multimeterdata = self.multimeter.get_measurements()
        print(f"multimeterdata={multimeterdata}")
        if len(self.raster.targets) and len(self.multimeter.targets):
            # names = [d.ID for t in measurements[0].targets]
            # labels = [d.get_labels() for d in self.detectors]
            ntd = len(rasterdata.T)
            nvd = len(multimeterdata.T)
            fig, _ = plt.subplots(
                constrained_layout=True, nrows=nvd + 2, figsize=(20, 20)
            )
            options = {
                "with_labels": True,
                "node_color": "white",
                "edgecolors": "blue",
                "ax": fig.axes[0],
                "node_size": 1100,
                "pos": nx.circular_layout(self.network.graph),
            }
            nx.draw_networkx(self.network.graph, **options)
            print(f"rasterdata={rasterdata}")
            print(f"multimeterdata={multimeterdata}")
            fig.axes[1].matshow(rasterdata.T, cmap="gray", aspect="auto")
            fig.axes[1].set_xticks(np.arange(-0.5, steps, 1), minor=True)
            fig.axes[1].set_yticks(np.arange(-0.5, ntd, 1), minor=True)
            fig.axes[1].grid(which="minor", color="gray", linestyle="-", linewidth=2)
            fig.axes[1].xaxis.set_major_locator(ticker.MultipleLocator(1))
            fig.axes[1].set_yticklabels(
                ["standin"] + [t.ID for t in self.raster.targets]
            )

            for i in range(nvd):
                fig.axes[i + 2].plot(multimeterdata[:, i])
                fig.axes[i + 2].set_ylabel(self.multimeter.targets[i].ID)
                fig.axes[i + 2].set_ylim(top=(max(multimeterdata.T[i]) + 0.5))
                fig.axes[i + 2].grid(b=None, which="major")
                fig.axes[i + 2].xaxis.set_major_locator(ticker.MultipleLocator(1))
            plt.show()
        elif len(self.raster.targets):
            # names = [d.ID for t in measurements[0].targets]
            # labels = [d.get_labels() for d in self.detectors]
            ntd = len(rasterdata.T)
            fig, ax = plt.subplots(constrained_layout=True, nrows=2, figsize=(20, 20))
            options = {
                "with_labels": True,
                "node_color": "white",
                "edgecolors": "blue",
                "ax": fig.axes[0],
                "node_size": 1100,
                "pos": nx.circular_layout(self.network.graph),
            }
            nx.draw_networkx(self.network.graph, **options)
            fig.axes[1].matshow(rasterdata.T, cmap="gray", aspect="auto")
            fig.axes[1].set_xticks(np.arange(-0.5, steps, 1), minor=True)
            fig.axes[1].set_yticks(np.arange(-0.5, ntd, 1), minor=True)
            fig.axes[1].grid(which="minor", color="gray", linestyle="-", linewidth=2)
            fig.axes[1].xaxis.set_major_locator(ticker.MultipleLocator(1))
            fig.axes[1].set_yticklabels(
                ["standin"] + [t.ID for t in self.raster.targets]
            )
            plt.show()
        elif len(self.multimeter.targets):
            # names = [d.ID for t in measurements[0].targets]
            # labels = [d.get_labels() for d in self.detectors]
            nvd = len(multimeterdata.T)
            fig, ax = plt.subplots(
                constrained_layout=True, nrows=nvd + 1, figsize=(20, 20)
            )
            options = {
                "with_labels": True,
                "node_color": "white",
                "edgecolors": "blue",
                "ax": fig.axes[0],
                "node_size": 1100,
                "pos": nx.circular_layout(self.network.graph),
            }
            nx.draw_networkx(self.network.graph, **options)
            for i in range(nvd):
                fig.axes[i + 1].plot(multimeterdata[:, i])
                fig.axes[i + 1].set_ylabel(self.multimeter.targets[i].ID)
                fig.axes[i + 1].set_ylim(top=(max(multimeterdata.T[i]) + 0.5))
                fig.axes[i + 1].grid(b=None, which="major")
                fig.axes[i + 1].xaxis.set_major_locator(ticker.MultipleLocator(1))
            plt.show()
