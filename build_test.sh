#!/bin/bash

# Set up test environment
python3 -m pip install --upgrade wheel setuptools pip
python3 setup.py install
python3 -m pip install -U pytest

# Add flag for test environment
export IS_CICD=true

# Force install of test dependencies before running pytest (should happen
#   automatically but sometimes fails)
if [["$OSTYPE" == "win32"]]; then python3 -m pip install pywin32 --upgrade; fi
python3 -m pip install .[test]
python3 -m pytest
