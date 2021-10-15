# Implementation of a Distributed Minimum Dominating Set Approximation Algorithm in a Spiking Neural Network
[![Python 3.8][python_badge]](https://www.python.org/downloads/release/python-382/)
[![License: GPL v3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Code Style: Black][black_badge]](https://github.com/ambv/black)

## Abstract
Neuromorphic computing is a promising new computational paradigm that may provide energy-lean solutions to algorithmic challenges such as graph problems. In particular, the class of distributed algorithms may benefit from translation to spiking neural networks. This work presents such a translation of a distributed approximation algorithm for the minimum dominating set problem, as described by Kuhn and Wattenhofer (2005), to a spiking neural network. This translation shows that neuromorphic architectures can be used to implement distributed algorithms. Subcomponents of this implementation, such as the calculation of the minimum or maximum of two numbers and degree of a node, can be reused as foundational building blocks for other (graph) algorithms. This work illustrates how leveraging neural properties for the translation of traditional algorithms relies on novel insights, thereby contributing to a growing body of knowledge on neuromorphic applications for scientific computing.

##### Authors: Victoria Bosch, Arne Diehl, Akke Toeter, Daphne Smits and Johan Kwisthout.   
##### Affiliation: School for Artificial Intelligence, Radboud University and Donders Center for Cognition, Radboud University.        
##### The journal paper can be accessed at: [TBD]

## Usage: do once

0. If you don't have pip: open Anaconda prompt and browse to the directory of this readme:
```
cd /home/<your path to the repository folder>/
```

1. To use this package, first make a new conda environment (it this automatically installs everything you need).
```
conda env create --file environment.yml
```

## Usage: do every time you start the terminal:

3. Activate the conda environment you created:
```
conda activate neumo
```

## Usage: do every run:

Navigate to the directory of this repository, you should execute the following instructions from within the directory called "spiking-neural-network-of-dominating-set-approximation"


4. Performe a run of the dominating set algorithm (in `main.py`, called from `__main__.py`)
```
python -m code.src
```

5. If you want to perform the algorithm on the [example graph](https://networkx.org/documentation/stable/reference/readwrite/graphml.html), use:
```
python -m code.src ./examples/examplegraph
```

## Testing

6. Testing is as simple as running the following command in the root directory of this repository in Anaconda prompt:
```
python -m pytest
```
from the root directory of this project.

## Documentation
The docstring documentation (template) was generated using `pyment`.

<!-- Un-wrapped URL's below (Mostly for Badges) -->
[black_badge]: https://img.shields.io/badge/code%20style-black-000000.svg
[python_badge]: https://img.shields.io/badge/python-3.6-blue.svg
