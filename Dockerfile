FROM kentoso.azurecr.io/kensciml/python36:latest AS setup
WORKDIR /kensci
COPY . .
# ARG PIP_EXTRA_INDEX_URL

RUN python3 -m pip install -U .[test]
RUN python3 -m pytest

# create artifacts folder for built package.
RUN mkdir /artifacts
RUN python3 -m pip install --upgrade wheel setuptools pip
RUN python3 setup.py bdist_wheel --dist-dir /artifacts
