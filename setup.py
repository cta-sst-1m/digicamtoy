import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="digicamtoy",
    version="1.1.0",
    author="Cyril Alispach",
    author_email="cyril.alispach@gmail.com",
    description="A Monte Carlo package of the SST-1m Camera",
    license="GNU GPLv3",
    keywords="Monte-Carlo SiPM SST-1m",
    url="https://github.com/calispac/digicamtoy",
    packages=['digicamtoy', 'digicamtoy.io', 'digicamtoy.core', 'digicamtoy.test', 'digicamtoy.utils',
              'digicamtoy.container', 'digicamtoy.production', 'digicamtoy.gui'],
    long_description=read('README.md'),
    install_requires=['numpy', 'matplotlib', 'scipy', 'h5py', 'pyyaml', 'logging', 'tqdm', 'datetime']

)
