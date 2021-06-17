#!/bin/bash
# Workaround to enable use of win32 on azure CI pipeline vm
python3 -m pip install pywin32 --upgrade
PYLOC=$(where python | head -n 1)
PYLOC=$(dirname $PYLOC)
ls $PYLOC
python3 ${PYLOC}\Lib\site-packages\win32\pywin32_postinstall.py -install
