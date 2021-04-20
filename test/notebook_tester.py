'''
Method Adapted from:
http://www.christianmoscardi.com/blog/2016/01/20/jupyter-testing.html

Upon recommendation from the AIF360 development team
(http://aif360.mybluemix.net/)
'''

import json
import nbformat
import os
import subprocess
import tempfile



def find_kernel(nb_file=None):
    """ Determines which, if any, jupyter kernels are available.

    Args:
        nb_file (str filepath, optional): If present, find_kernel will parse for
        the notebook's default kernel name and check to see if it's among the
        available kernels. Defaults to None.

    Returns:
        str: name of a viable kernel
    """
    if nb_file is not None:
        with open(nb_file) as json_file:
            contents = json.load(json_file)
            if 'kernelspec' in contents['metadata'].keys():
                kname = contents['metadata']['kernelspec']['name']
            else:
                kname = None

    args = ["jupyter", "kernelspec", "list"]
    try:
        kernels = subprocess.check_output(args).decode("utf-8")
        # If any kernels are available, the first item in post-split list will
        # be "Available kernels" (hence truncate)
        kernels = kernels.split("\n")[1:]
        available_kernels = [k.split()[0] for k in kernels if k.split()]
        if kname is None or kname not in available_kernels:
            kname = available_kernels[0]
        return kname
    except OSError:
        return None
    except BaseException as e:
        raise Exception(e)


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


def validate_notebook(nb_path, timeout=60):
    """ Executes the notebook via nbconvert and collects the output

    Args:
        nb_path (string): path to the notebook of interest
        timeout (int): max allowed time (in seconds)

    Returns:
        (parsed nbformat.NotebookNode object, list of execution errors)
    """
    dirname, __ = os.path.split(nb_path)
    os.chdir(dirname)

    kname = find_kernel(nb_path)
    if kname is None:
        raise OSError("No kernel found")

    with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False) as tf:
        args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
        f"--ExecutePreprocessor.timeout={timeout}",
        f"--ExecutePreprocessor.kernel_name={kname}",
        "--ExecutePreprocessor.allow_errors=True",
        "--output", tf.name, nb_path]

        subprocess.check_call(args)

        tf.seek(0)
        nb = nbformat.read(tf, nbformat.current_nbformat)

    errors = list_errors(nb)

    return nb, errors
