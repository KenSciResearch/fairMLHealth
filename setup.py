"""A setuptools based setup module for fairMLHealth

"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name='fairMLHealth',
    version='1.0.0',
    description='Health-centered fairness measurement and management',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/KenSciResearch/fairMLHealth',
    author='KenSci',
    author_email='christine.allen@kensci.com',
    python_requires='>=3.5, <4',
    install_requires=['aif360',
                      'fairlearn',
                      'requests',
                      'pandas>=0.21.0'
                      'scipy',
                      'scikit-learn>=0.2.0',
                      'xgboost'],
    project_urls={'KenSci': 'https://www.kensci.com'},
    keywords='healthcare, machine learning, fairness',

    # For valid classifiers see https://pypi.org/classifiers/
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ]

    # When your source code is in a subdirectory under the project root, e.g.
    # `src/`, it is necessary to specify the `package_dir` argument.
    #package_dir={'': 'src'},  # Optional

    # You can just specify package directories manually here if your project is
    # simple. Or you can use find_packages().
    #
    # Alternatively, if you just want to distribute a single Python file, use
    # the `py_modules` argument instead as follows, which will expect a file
    # called `my_module.py` to exist:
    #
    #   py_modules=["my_module"],
    #
    #packages=find_packages(where='src'),  # Required

)
