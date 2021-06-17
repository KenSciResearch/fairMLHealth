#!/bin/bash
# About: Sets-up test environment & runs tests; for use in development of
#   this library

# Set up test environment
python3 -m pip install --upgrade wheel setuptools pip
python3 setup.py install
python3 -m pip install -U pytest

# Add flag for test environment
export IS_CICD=true

# Force install of test dependencies before running pytest (should happen
#   automatically but sometimes fails)
python3 -m pip install .[test]
python3 -m pytest
