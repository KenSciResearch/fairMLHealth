from contextlib import suppress
import numpy as np
import os
import pandas as pd
import pytest
import regex as re
import requests
from time import sleep
from urllib3.exceptions import InsecureRequestWarning
import warnings


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


def synth_dataset(N:int=16):
    """ Synthesizes a set of random data with multiple "predictions"

    Args:
        N (int, optional): Desired size of the dataset. Defaults to 16.

    Returns:
        pandas DataFrame

    """
    np.random.seed(506)
    # If N is not divisible by 8 this will break
    N = int(N)
    if not N%8 == 0:
        N += 8 - N%8
    # Generate dataframe with ranodm information
    df = pd.DataFrame({'A': np.random.randint(1, 4, N),
                        'B': np.random.randint(1, int(N/2), N),
                        'C': np.random.randint(1, N, N),
                        'D': np.random.randint(1, int(2*N), N),
                        'E': np.random.uniform(-10, 10, N),
                        'prtc_attr': [0, 1]*int(N/2),
                        'prtc_attr2': [1, 1, 1, 1, 0, 0, 0, 0]*int(N/8),
                        'other': [1, 0, 0, 1]*int(N/4),
                        'continuous_target': np.random.uniform(0, 8, N),
                        'binary_target': np.random.randint(0, 2, N),
                        })

    # add predictions that are fair half of the time
    half_correct = pd.Series([1, 1, 0, 0]*int(N/4))
    df['avg_binary_pred'] = df['binary_target']
    df.loc[half_correct.eq(0), 'avg_binary_pred'] = \
        df['binary_target'].apply(lambda x: int(not x))
    df['avg_cont_pred'] = df['continuous_target']
    df.loc[half_correct.eq(0), 'avg_cont_pred'] = \
        df['continuous_target'].apply(lambda x: x + np.random.uniform(-6, 6))

    # add predictions that are biased in one direction or the other
    against_0 = pd.Series([1, 1, 0, 1]*int(N/4))
    df['binary_bias_against0'] = df['binary_target']
    df.loc[against_0.eq(0), 'binary_bias_against0'] = \
        df['binary_target'].apply(lambda x: int(not x))

    toward_0 = pd.Series([1, 1, 1, 0]*int(N/4))
    df['binary_bias_toward0'] = df['binary_target']
    df.loc[toward_0.eq(0), 'binary_bias_toward0'] = \
        df['binary_target'].apply(lambda x: int(not x))

    return df

class URLError(requests.exceptions.BaseHTTPError):
    pass


def __invalid_url_delimiter(char):
    ''' Some characters will be incorrectly evaluated by get_urls as part of
    the address when they are actually part of the document text. Its possible
    that some addresses may end with some of these characters, but more likely
    that they're get_urls errors
    '''
    invalid_chars = ["(", ")", ".", ",", "[", "]", "}"]
    is_invalid = True
    if char not in invalid_chars:
        is_invalid = False
    return is_invalid
