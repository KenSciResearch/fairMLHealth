'''
Method Adapted from:
http://www.christianmoscardi.com/blog/2016/01/20/jupyter-testing.html

Upon recommendation from the AIF360 development team
(http://aif360.mybluemix.net/)
'''

from . import __utils as utils
import json
import nbformat
import os
import subprocess
import tempfile



def find_broken_urls(nb):
    ''' Validates most urls with some exceptions (see documentation for
    __utils.is_url_valid)
    '''
    url_list = list_urls(nb)
    broken_urls = []
    for url in url_list:
        is_valid = utils.is_url_valid(url)
        if type(is_valid)==bool and not is_valid:
            code = utils.get_url_status(url, tryonce=True)
            err = f"{repr(url)} ({code} Error)"
            broken_urls.append(err)
    return broken_urls


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
    errs = [output for cell in nb.cells if "outputs" in cell
                for output in cell["outputs"]
                if output.output_type == "error"]
    return errs


def list_urls(nb):
    urls = []
    for cell in nb.cells:
        if "http" in cell['source']:
            search_text = cell['source']
            urls += utils.get_urls(search_text)
    return urls


def list_warnings(nb):
    """ Returns any errors displayed in the notebook

    Args:
        nb (parsed nbformat.NotebookNode)
    """
    wrns = []
    for cell in nb.cells:
        if "outputs" in cell:
            for output in cell["outputs"]:
                if (output.output_type == "stream"
                    and output.name == "stderr"
                    and "warning" in output.text.lower()):
                    wrns.append(output)
    return wrns


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

    # Set delete=False as workaround for Windows OS
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

    # broken urls are currently counted as errors; consider including as
    #   warnings
    broken_urls = find_broken_urls(nb)
    if any(broken_urls):
        broken_urls = ["broken url: " + u for u in broken_urls]
        errors += broken_urls

    return nb, errors
