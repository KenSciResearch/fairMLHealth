#!/bin/bash

# Workaround for pywin32 install on certain Windows machines (req. for
#   notebook_tester). Enables/ensures install of win32 on azure CI. Only
#   necessary if you intend to run pytest for the development of this library.
python3 -m pip install pywin32 --upgrade
PYLOC=$(where python | head -n 1)
PYLOC=$(dirname $PYLOC)
python3 ${PYLOC}\\Scripts\\pywin32_postinstall.py -install
