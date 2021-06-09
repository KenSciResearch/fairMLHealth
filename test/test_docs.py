''' 
Test script to flag obvious errors in markdown documentation files. For
example, fails if an http(s) link is broken. 
'''
import os
from pathlib import Path
import pytest
from .__utils import get_urls, is_url_valid, URLError



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
            raise URLError(f"Invalid URL detected in {filepath}: {test_url}")


def validate_markdown(md_path):
    validate_filepath(md_path)
    validate_urls(md_path)


''' Testers '''


def test_docsREADME():
    repo_main = base_dir()
    file = os.path.join(repo_main, "docs", "README.md")
    validate_markdown(file)


def test_evaluatingFairness():
    repo_main = base_dir()
    file = os.path.join(repo_main, "docs", "resources",
                        "Evaluating_Fairness.md")
    validate_markdown(file)


def test_main():
    repo_main = base_dir()
    file = os.path.join(repo_main, "README.md")
    validate_markdown(file)


def test_measuresQuickRef():
    repo_main = base_dir()
    file = os.path.join(repo_main, "docs", "resources",
                        "Measures_QuickReference.md")
    validate_markdown(file)


def test_publicationsREADME():
    repo_main = base_dir()
    file = os.path.join(repo_main, "docs", "publications", "README.md")
    validate_markdown(file)


def test_refsAndResources():
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
    file = os.path.join(repo_main, "tutorials_and_examples", "README.md")
    validate_markdown(file)
