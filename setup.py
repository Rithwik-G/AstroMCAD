from setuptools import setup, find_packages
import codecs
import os

with open("READMEPyPI.md", "r") as fh:
    long_description = fh.read();

VERSION = '0.0.10'
DESCRIPTION = 'Anomaly Detection for Astronomical Transients'
LONG_DESCRIPTION = 'Train custom classifiers to be used as anomaly detectors for astronmical transients'

# Setting up
setup(
    name="astromcad",
    version=VERSION,
    author="Rithwik Gupta",
    author_email="<rithwikca2020@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['numpy', 'keras', 'tensorflow', 'matplotlib', 'scikit-learn'],
    keywords=[],
    
)