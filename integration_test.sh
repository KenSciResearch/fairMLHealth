#!/bin/bash
apt-get update && apt-get install -y libgomp1
apt-get install -y libgomp1 gcc g++

# ARG PIP_EXTRA_INDEX_URL
python3 -m pip install --upgrade wheel setuptools pip
python3 setup.py install
python3 -m pip install -U pytest

# Add flag for test environment
export IS_CICD=true

# Force install of test dependencies before running pytest (should happen
#   automatically but sometimes fails)
python3 -m pip install .[test]
python3 -m pytest