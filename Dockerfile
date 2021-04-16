FROM kentoso.azurecr.io/kensciml/python37:latest AS setup

RUN apt-get update && apt-get install -y libgomp1
RUN apt-get install -y libgomp1 gcc g++

WORKDIR /kensci
COPY . .
# ARG PIP_EXTRA_INDEX_URL
RUN python3 -m pip install --upgrade wheel setuptools pip
RUN python3 setup.py install
RUN python3 -m pip install -U pytest
RUN python3 -m pip install -U nbformat nbconvert ipython

# ensure that python3 kernel is avialble
RUN python3 -m ipython kernel install
ENV IS_CICD=true
RUN python3 -m pytest

# create artifacts folder for built package.
RUN mkdir /artifacts
RUN python3 setup.py bdist_wheel --dist-dir /artifacts
