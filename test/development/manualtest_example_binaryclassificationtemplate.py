'''
Method adapted from:
http://www.christianmoscardi.com/blog/2016/01/20/jupyter-testing.html

Note: because the models for the notebook being tested may take hours to run,
this test must be run manually. However, since other notebooks rely on the same
functions it's very unlikely
'''

from .. import notebook_tester as nbtest, __utils
import os
import warnings





def test_example_binaryclassificationtemplate():
    # Because this notebook relies on MIMIC3 data that should not be stored in
    # the cloud, skip this test if running in the azure CI/CD pipeline
    if __utils.is_test_environment():
        return None
    else:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        nb_name = "Example-BinaryClassificationTemplate.ipynb"
        nb_path = os.path.join(this_dir, "..", "..",
                               "tutorials_and_examples", nb_name)

        nb, err = nbtest.validate_notebook(nb_path, timeout=1800)

        if any(err):
            for e in err:
                for t in e['traceback']:
                    print(t)
            raise AssertionError("Notebook Broken")
        else:
            pass

        warns = nbtest.list_warnings(nb)
        if any(warns):
            for t in warns:
                if isinstance(t['text'], list):
                    wrn = t['text'][0]
                else:
                    wrn = t['text']
                warnings.warn(wrn)
