"""A setuptools based setup module for fairMLHealth

"""
from setuptools import setup, find_packages
import pathlib

# Get the long description from the README file
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='fairMLHealth',
    version='0.0.1',
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
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ]
)
