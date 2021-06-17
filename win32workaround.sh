#!/bin/bash
# Workaround for pywin32 install on certain Windows machines.
#   Enables/ensures install of win32 on azure CI (req. for notebook_tester)
python3 -m pip install pywin32 --upgrade
PYLOC=$(where python | head -n 1)
PYLOC=$(dirname $PYLOC)
python3 ${PYLOC}\\Scripts\\pywin32_postinstall.py -install
