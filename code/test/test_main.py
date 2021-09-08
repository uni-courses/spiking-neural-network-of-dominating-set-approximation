import unittest

from code.src.main import dominating_set_snn, dominating_set_neumann
from code.test.helper import create_graph_abcd


class TestMain(unittest.TestCase):
    """ """

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_main_functions(self):
        # run algorithm 1
        graph = create_graph_abcd()
        dominating_set_snn(graph)
        dominating_set_neumann(graph)


if __name__ == "__main__":
    unittest.main()
