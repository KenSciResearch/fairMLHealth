'''
Method adapted from:
http://www.christianmoscardi.com/blog/2016/01/20/jupyter-testing.html
'''

from notebook_tester import validate_notebook
import os


def test_example_toolusage():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    nb = "Tutorial-EvaluatingFairnessInBinaryClassification.ipynb"
    nb_path = os.path.join(this_dir, "..", "tutorials_and_examples", nb)
    _, err = validate_notebook(nb_path, timeout=1200)

    if any(err):
        for e in err:
            for t in e['traceback']:
                print(t)
        raise AssertionError("Notebook Broken")
    else:
        pass

