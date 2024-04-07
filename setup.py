from setuptools import setup, find_packages
import codecs
import os

with open("READMEPyPI.md", "r") as fh:
    long_description = fh.read();

VERSION = '1.0.0'
DESCRIPTION = 'Classifier-Based Anomaly Detection for Astronomical Transients'
LONG_DESCRIPTION = long_description

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
    install_requires=['numpy', 'keras==2.15.0', 'tensorflow==2.15.0', 'matplotlib', 'scikit-learn==1.2.2'],
    keywords=[],
    package_data={'astromcad': ['*.keras', '*.pickle']},
    include_package_data=True,
    license='MIT',

    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # 'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
    ],
    
)