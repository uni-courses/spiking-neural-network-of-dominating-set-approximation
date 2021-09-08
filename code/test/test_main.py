import unittest
import os

from code.src.main import compile_latex_report, dominating_set_snn
from code.test.helper import delete_file_if_exists
from code.test.helper import file_exists
from code.test.helper import create_graph_abcd


class TestMain(unittest.TestCase):
    """ """

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.script_dir = self.get_script_dir()

    # returns the directory of this script regardles of from which level the code is executed
    def get_script_dir(self):
        """ """
        return os.path.dirname(__file__)

    def test_latex_compiles(self):
        latex_pdf_path = "latex/report/main.pdf"
        delete_file_if_exists(latex_pdf_path)
        self.assertFalse(file_exists(latex_pdf_path))

        # run algorithm 1
        graph = create_graph_abcd()
        dominating_set_snn(graph)

        # compile the latex report
        compile_latex_report()
        self.assertTrue(file_exists(latex_pdf_path))


if __name__ == "__main__":
    unittest.main()
