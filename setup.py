"""A setuptools based setup module for fairMLHealthHealth

"""
from setuptools import setup, find_packages


def _get_version():
    import json
    import os

    version_file = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        'version.json'
    )
    return json.load(open(version_file))['version']



long_description = ("A library facilitating fairness measurement" +
                    " and deployment of fairness-aware ML algorithms")


# OS-specific dependencies required to set up the test environment
os_deps = ['pypiwin32; platform_system == "Windows"',
           'pywin32; platform_system == "Windows"'
            ]


# Requirements for the test environment
test_deps = ["pytest==5.4.2", "ipython", "ipyparallel", "nbformat", "nbconvert",
             "regex"] + os_deps


# Requirements for running tutorial notebooks
tutorial_deps = ['fairlearn>=0.4.6', 'lightgbm', 'matplotlib', 'seaborn',
                 'xgboost'
                 ]


setup(
    name='fairmlhealth',
    version=_get_version(),
    description='Health-centered fairness measurement and management',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/KenSciResearch/fairMLHealth',
    author='christine allen',
    author_email='ca.magallen@gmail.com',
    tests_require=test_deps,
    extras_require={
                    "test": test_deps,
                    "tutorial": tutorial_deps
                    },
    python_requires='>=3.6,<4',
    install_requires=[
                      'aif360>=0.3.0',
                      'ipython',
                      'jupyter',
                      'numpy>=1.16',
                      'pandas>=1.0.3',
                      'requests',
                      'scipy>=1.4.1,<1.6.0',
                      'scikit-learn>=0.21'
                    ] + tutorial_deps,
    project_urls={'KenSci': 'https://www.kensci.com'},
    keywords='healthcare, machine learning, fairness, fair ML',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
    packages=find_packages(include=['fairmlhealth', 'fairmlhealth.*'])
)
