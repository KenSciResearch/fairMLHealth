'''
Method adapted from:
http://www.christianmoscardi.com/blog/2016/01/20/jupyter-testing.html
'''

from .notebook_tester import validate_notebook, list_warnings
import os
import warnings


def test_example_toolusage():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    nb_name = "Example-ToolUsage_Regression.ipynb"
    nb_path = os.path.join(this_dir, "..", "tutorials_and_examples", nb_name)
    nb, err = validate_notebook(nb_path)

    if any(err):
        for e in err:
            for t in e['traceback']:
                print(t)
        raise AssertionError("Notebook Broken")

    warns = list_warnings(nb)
    if any(warns):
        for t in warns:
            if isinstance(t['text'], list):
                wrn = t['text'][0]
            else:
                wrn = t['text']
            warnings.warn(wrn)
