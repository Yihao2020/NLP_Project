import os
from setuptools import setup, find_packages
import subprocess
from setuptools.command.install import install as _install

PACKAGE_NAME = 'NLP_Project'

setup(
    name=PACKAGE_NAME,
    version='0.0.1',
    description='Toxic Content Classification',
    author='Data Lab ',
    url = "https://github.com/deveshiitkgp2013/NLP_Project",
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=[
    'numpy',
    'pandas',
    'nltk'
    ],

)

