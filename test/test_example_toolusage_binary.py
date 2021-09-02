'''
Method adapted from:
http://www.christianmoscardi.com/blog/2016/01/20/jupyter-testing.html
'''

from .notebook_tester import validate_notebook, check_results
import os


def test_example_toolusage():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    nb_name = "Example-ToolUsage_BinaryClassification.ipynb"
    nb_path = os.path.join(this_dir, "..", "examples_and_tutorials", nb_name)
    nb, err = validate_notebook(nb_path)

    check_results(nb, err)
