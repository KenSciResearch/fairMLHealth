from contextlib import suppress
import os
import regex as re
import requests
from time import sleep




def __invalid_url_ending(char):
    ''' Some characters will be incorrectly evaluated by regex.findall as part
    of a url when they are actually part of the document text.
    '''
    invalid_chars = [")", ".", ",", "]"]
    is_valid = False
    if char not in invalid_chars:
        is_valid = True
    return is_valid


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
        #     (i.e. "[url](url)") should be split
        if "](" in url and url.endswith(")"):
            raw_urls += url.split("](")
            continue
        # Trailing symbols will be recognized as part of the url
        else:
            while not __invalid_url_ending(url[-1]):
                url = url[:-1]
            output.append(url)
    return output


def is_test_environment():
    """ Determines if the test is being run in its test (Docker)
        environment. Notebook relying on MIMIC3 data, which should not
        be stored in the cloud, should not be tested if in this environment

    Returns:
        bool: True if running in the test environment
    """
    return os.environ.get('IS_CICD', False)


def is_url_valid(url):
    """ Tests url for error response code. Skips validation (returns True) for
        urls for which SSL certificate cannot be validated (this behavior to be
        improved). Currently set to return None unless running in the
        test environment (via GitHub)

    Args:
        url (string): url to be validated

    Returns:
        bool: False if response status is 400 or above; None if test
        unsuccessful; otherwise True.
    """
    status = None
    # ToDo: add validation in case request error; for now, just suppress
    with suppress(requests.exceptions.RequestException):
        # Use stream=True to download only until reaching Response.content.
        #   Could use requests.head, but apparently some sites don't support it.
        status = requests.get(url, stream=True).status_code
    # Repeat attempt in case of server error or timeout
    if status in range(500, 505):
        sleep(5)
        status = requests.get(url).status_code
    # Response codes below 400 indicate that the address exists
    if status is not None and status < 400:
        status = True
    return status


class URLError(requests.exceptions.BaseHTTPError):
    pass