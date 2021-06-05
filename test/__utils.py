import os

def is_test_environment():
    """ Determines if the test is being run in its Azure CI/CD Docker
        environment. Notebook relying on MIMIC3 data, which should not
        be stored in the cloud, should not be tested if in this environment

    Returns:
        bool: True if running in the test environment
    """
    is_azpipeline = os.environ.get('IS_CICD', False)
    return is_azpipeline


def is_url_valid(url):
    """ Tests url for error response code

    Args:
        url (string): url to be validated

    Returns:
        bool: False if response status is 400 or above; otherwise True
    """
    status = requests.get(url).status_code
    if status >= 400:
        return False
    else:
        return True