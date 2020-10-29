FROM kentoso.azurecr.io/kensciml/python36:latest AS setup
WORKDIR /kensci
COPY . .
# ARG PIP_EXTRA_INDEX_URL

# TODO: Add unit-test and update below to run unit-test
# # installs python build and test dependencies.
# # ex: python3 -m pip install .[test] --extra-index-url ${PIP_EXTRA_INDEX_URL}
# RUN <setup>
# # unit-test. replace <unit-test> with testing command
# RUN <unit-test>

# create artifacts folder for built package.
RUN mkdir /artifacts
RUN python3 -m pip install --upgrade wheel setuptools pip
RUN python3 setup.py bdist_wheel --dist-dir /artifacts
