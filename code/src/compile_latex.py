import os
import shutil


class Compile_latex:
    """Runs jupyter notebooks, converts them to pdf,
    exports the notebook pdfs to latex and compiles the
    latex report of the incoming project nr"""

    def __init__(self, latex_filename):
        """Constructs attributes used throughout latex compilation

        :param project_nr: the numberr identifying which project is being  ran and compiled
        :param latex_filename: name of the main latex .tex file that manages the latex document
        """

        self.script_dir = self.get_script_dir()
        relative_dir = f"latex/report/"
        self.compile_latex(relative_dir, latex_filename)
        self.clean_up_after_compilation(latex_filename)
        self.move_pdf_into_latex_dir(relative_dir, latex_filename)

    def compile_latex(self, relative_dir, latex_filename):
        """Executes a commandline line to compile the latex report

        :param relative_dir: the relative dir towards the latex main .tex file
        :param latex_filename: name of the main latex .tex file that manages the latex document

        """
        os.system(f"pdflatex {relative_dir}{latex_filename}")

    def clean_up_after_compilation(self, latex_filename):
        """Removes the unneeded files that were generated during latex to pdf compilation.

        :param latex_filename: name of the main latex .tex file that manages the latex document

        """
        latex_filename_without_extention = latex_filename[:-4]
        self.delete_file_if_exists(f"{latex_filename_without_extention}.aux")
        self.delete_file_if_exists(f"{latex_filename_without_extention}.log")
        self.delete_file_if_exists(f"{latex_filename_without_extention}.toc")
        self.delete_file_if_exists(f"texput.log")

    def move_pdf_into_latex_dir(self, relative_dir, latex_filename):
        """Moves the compiled/generated pdf file from the root of this repository to the
        relative latex directory of this project.

        :param relative_dir: param latex_filename:
        :param latex_filename: name of the main latex .tex file that manages the latex document

        """
        pdf_filename = f"{latex_filename[:-4]}.pdf"
        destination = f"{self.get_script_dir()}/../../{relative_dir}{pdf_filename}"

        try:
            shutil.move(pdf_filename, destination)
        except:
            print("Error while moving file ", pdf_filename)

    def delete_file_if_exists(self, filename):
        """Deletes files if they exist

        :param filename: name of file that will be deleted if it exists in the root of this repository

        """
        try:
            os.remove(filename)
        except:
            print(
                f"Error while deleting file: {filename} but that is not too bad because the intention is for that byproduct to not be there."
            )

    def get_script_dir(self):
        """returns the directory of this script regardles of from which level the code is executed"""
        return os.path.dirname(__file__)
