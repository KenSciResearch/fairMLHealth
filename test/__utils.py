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