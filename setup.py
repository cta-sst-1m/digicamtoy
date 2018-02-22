import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="digicamtoy",
    version="1.1.1",
    author="Cyril Alispach",
    author_email="cyril.alispach@gmail.com",
    description="A Monte Carlo package of the SST-1m Camera",
    license="GNU GPLv3",
    keywords="Monte-Carlo SiPM SST-1m",
    url="https://github.com/cta-sst-1m/digicamtoy",
    packages=find_packages(),
    package_data={'': ['utils/pulse_SST-1M_pixel_0.dat',
                       'utils/pulse_SST-1M_AfterPreampLowGain.dat']},
    include_package_data=True,
    long_description=read('README.md'),
    install_requires=['numpy', 'matplotlib', 'scipy', 'h5py', 'pyyaml',
                      'logging', 'tqdm', 'datetime', 'cython', 'pandas'],
)
