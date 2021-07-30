'''
Test script to flag obvious errors in markdown documentation files. Example
errors included broken http(s) links. At least verifies the presence of
documentation that is expected.
'''
import os
from pathlib import Path
import pytest
from .__test_utils import (get_urls, get_url_status,  is_test_environment,
                            is_url_valid, URLError)



def base_dir():
    return Path(__file__).parent.parent.resolve()


def validate_filepath(filepath):
    if not os.path.exists(filepath):
        raise OSError(f"File does not exist. Check the path: {filepath}")


def validate_urls(filepath):
    ''' Validates most urls with some exceptions (see documentation for
    __utils.is_url_valid)
    '''
    with open(filepath, "r", encoding="utf-8") as md_file:
        text = md_file.read()
    urls = get_urls(text)
    while any(urls):
        test_url = urls.pop()
        is_valid = is_url_valid(test_url)
        if type(is_valid)==bool and not is_valid:
            err_code = get_url_status(test_url, tryonce=True)
            raise URLError(f"Invalid URL detected in {filepath}:"
                           + f" {repr(test_url)}, {err_code} Error")


def validate_markdown(md_path):
    validate_filepath(md_path)
    # Validating URLs takes time, so only validate if running on test env.
    #   environment (on GitHub)
    if is_test_environment():
        validate_urls(md_path)


''' Testers '''


def test_docsREADME():
    repo_main = base_dir()
    file = os.path.join(repo_main, "docs", "README.md")
    validate_markdown(file)


def test_evaluatingFairness():
    ''' Validates document elaborating on fairness evaluation process '''
    repo_main = base_dir()
    file = os.path.join(repo_main, "docs", "resources",
                        "Evaluating_Fairness.md")
    validate_markdown(file)


def test_main():
    ''' Validates the repository's main README '''
    repo_main = base_dir()
    file = os.path.join(repo_main, "README.md")
    validate_markdown(file)


def test_measuresQuickRef():
    ''' Validates document containing charts of measure definitions '''
    repo_main = base_dir()
    file = os.path.join(repo_main, "docs", "resources",
                        "Measures_QuickReference.md")
    validate_markdown(file)


def test_publicationsREADME():
    repo_main = base_dir()
    file = os.path.join(repo_main, "docs", "publications", "README.md")
    validate_markdown(file)


def test_refsAndResources():
    ''' Validates document containing a list of references and outside resources
    '''
    repo_main = base_dir()
    file = os.path.join(repo_main, "docs", "resources",
                        "References_and_Resources.md")
    validate_markdown(file)


def test_templatesREADME():
    repo_main = base_dir()
    file = os.path.join(repo_main, "templates", "README.md")
    validate_markdown(file)


def test_tutorialsREADME():
    repo_main = base_dir()
    file = os.path.join(repo_main, "examples_and_tutorials", "README.md")
    validate_markdown(file)
