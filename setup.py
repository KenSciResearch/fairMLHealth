"""A setuptools based setup module for fairMLHealthHealth

"""
from setuptools import setup, find_packages
import pathlib

version = '0.1.0'

# Get the long description from the README file
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='fairMLHealth',
    version=version,
    description='Health-centered fairness measurement and management',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/KenSciResearch/fairMLHealth',
    author='KenSci',
    author_email='christine.allen@kensci.com',
    python_requires='>=3.5, <4',
    install_requires=['aif360>=0.3.0',
                      'fairlearn>=0.4.6',
                      'lightgbm',
                      'matplotlib',
                      'numpy>=1.17.2',
                      'pandas>=0.25.1',
                      'requests',
                      'scipy>=1.3.1',
                      'scikit-learn>=0.22.1',
                      'seaborn',
                      'tensorflow',
                      'xgboost'
                    ],
    project_urls={'KenSci': 'https://www.kensci.com'},
    keywords='healthcare, machine learning, fairness',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ]
    packages=find_packages(include=['fairMLHealth', 'fairMLHealth.*'])
)

