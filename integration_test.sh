#!/bin/bash

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


# create artifacts folder for built package.
if [ -z ${BDIST_WHEEL+x} ]; then export BDIST_WHEEL=false; fi
if $BDIST_WHEEL
then
    mkdir /artifacts
    python3 setup.py bdist_wheel --dist-dir /artifacts
fi
