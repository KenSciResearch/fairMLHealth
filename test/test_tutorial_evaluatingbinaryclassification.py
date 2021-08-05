'''
Method Adapted from:
http://www.christianmoscardi.com/blog/2016/01/20/jupyter-testing.html

Upon recommendation from the AIF360 development team
(http://aif360.mybluemix.net/)
'''

from .notebook_tester import validate_notebook, check_results
from .__testing_utilities import is_test_environment
import os


def test_tutorial_evaluatingbinaryclassification():
    # Because this notebook relies on MIMIC3 data that should not be stored in
    # the cloud, skip this test if running in the azure CI/CD pipeline
    if is_test_environment():
        return None
    elif not os.path.exists(os.path.abspath(os.path.expanduser("~/data/MIMIC"))):
        # ToDo: use readily downloadable data for notebooks that require
        # integrated testing. Less reliance on MIMIC in general is preferred.
        return None
    else:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        nb_name = "Tutorial-EvaluatingFairnessInBinaryClassification.ipynb"
        nb_path = os.path.join(this_dir, "..",
                               "examples_and_tutorials", nb_name)
        nb, err = validate_notebook(nb_path, timeout=1800)

        check_results(nb, err)

