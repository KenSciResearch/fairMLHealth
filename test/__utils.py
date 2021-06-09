from contextlib import suppress
from urllib3.exceptions import InsecureRequestWarning
import warnings
import os
import regex as re
import requests
from time import sleep


def get_urls(text_string):
    """ Scrapes for urls beginning with http or https. Links starting with sftp
        or other protocols should not be tested with this function.

        Note: the regex.findall function may errantly truncate urls rendering
        LaTex (e.g. via https://render.githubusercontent.com/render/) if those
        urls include spaces. For better performance, replace any spaces with
        "%20" in relevant urls.

    Args:
        text_string (str): text to scrape

    Returns:
        list: list of urls detected
    """
    # Regex is proved to be the most robust option assuming "http" or "https"
    url_pattern = r'(https?://[^\s]+)'
    raw_urls = re.findall(url_pattern, text_string)
    output = []
    while any(raw_urls):
        url = raw_urls.pop()
        # Markdown of format [text](link) displaying a url as text
        #     (i.e. "[url](url)") should be split and added back to the list
        if "](http" in url:
            raw_urls += url.split("](http")
            raw_urls[1] = "http" + raw_urls[1]
            continue
        else:
            # Remove errant leading & trailing symbols will be recognized as
            #   part of the url
            url = url.rstrip()
            while __invalid_url_delimiter(url[-1]):
                url = url[:-1]
            output.append(url)
    return output


def get_url_status(url, tryonce=False):
    """ Gets URL response code. Re-tests in case of server error. Does not
        verify secure connection. Does nothing in event of exception.

    Args:
        url (string): url to be validated
        tryonce (bool): when False, will try again in case of certain erros
            (such as server error). Defaults to False.

    Returns:
        bool: None if test unsuccessful; otherwise int of response status.
    """
    status = None
    # ToDo: add validation in case request error; for now, just suppress
    with suppress(requests.exceptions.RequestException):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=InsecureRequestWarning)
            # Use stream=True to download only until reaching Response.content.
            #   Could use requests.head, but some sites don't support it.
            status = requests.get(url, stream=True, verify=False,
                                  timeout=5).status_code
    # Repeat attempt in case of server error or timeout
    if not tryonce and status in range(500, 505):
        sleep(5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=InsecureRequestWarning)
            status = requests.get(url, stream=True, verify=False).status_code
    return status


def is_test_environment():
    """ Determines if the test is being run in its test (Docker)
        environment. Notebook relying on MIMIC3 data, which should not
        be stored in the cloud, should not be tested if in this environment

    Returns:
        bool: True if running in the test environment
    """
    return os.environ.get('IS_CICD', False)


def is_url_valid(url):
    """ Tests url for non-error response code. Does not verify secure
        connection. Does nothing in event of exception.

    Args:
        url (string): url to be validated

    Returns:
        bool: False if response status is 400 or above; None if test
        unsuccessful; otherwise True.
    """
    status = get_url_status(url)
    # Response codes below 400 indicate that the address exists
    if status is not None:
        is_valid = True if status < 400 else False
    else:
        is_valid = None
    return is_valid


class URLError(requests.exceptions.BaseHTTPError):
    pass


def __invalid_url_delimiter(char):
    ''' Some characters will be incorrectly evaluated by regex.findall as part
    of a url when they are actually part of the document text.
    '''
    invalid_chars = ["(", ")", ".", ",", "[", "]"]
    is_invalid = True
    if char not in invalid_chars:
        is_invalid = False
    return is_invalid
