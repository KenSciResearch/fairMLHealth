'''
Adapted from:
http://www.christianmoscardi.com/blog/2016/01/20/jupyter-testing.html

Upon recommendation from the AIF360 development team
(http://aif360.mybluemix.net/)
'''


import nbformat
import os
import subprocess
import tempfile


def validate_notebook(path, timeout=60, ):
    """ Executes the notebook via nbconvert and collects the output

    Args:
        path (string): path to the notebook of interest
        timeout (int): max allowed time (in seconds)

    Returns:
        (parsed nbformat.NotebookNode object, list of execution errors)
    """
    dirname, __ = os.path.split(path)
    os.chdir(dirname)

    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
        f"--ExecutePreprocessor.timeout={timeout}",
        "--ExecutePreprocessor.allow_errors=True",
        "--output", fout.name, path]

        subprocess.check_call(args)

        fout.seek(0)
        nb = nbformat.read(fout, nbformat.current_nbformat)

    errors = list_errors(nb)

    return nb, errors


def list_errors(nb):
    """ Generates a list of errors displayed in the notebook

    Args:
        nb (parsed nbformat.NotebookNode)
    """
    errors = [output for cell in nb.cells if "outputs" in cell
                for output in cell["outputs"]
                if output.output_type == "error"]
    return errors


def list_warnings(nb):
    """ Returns any errors displayed in the notebook

    Args:
        nb (parsed nbformat.NotebookNode)
    """
    warns = []
    for cell in nb.cells:
        if "outputs" in cell:
            for output in cell["outputs"]:
                if (output.output_type == "stream"
                    and output.name == "stderr"
                    and "warning" in output.text.lower()):
                    warns.append(output)

    return warns

