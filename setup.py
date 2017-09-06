from distutils.core import setup

setup(
    name='digicamtoy',
    version='1.0.0',
    packages=['digicamtoy', 'digicamtoy.io', 'digicamtoy.core', 'digicamtoy.test', 'digicamtoy.utils',
              'digicamtoy.container', 'digicamtoy.production', 'digicamtoy.gui'],
    url='https://github.com/calispac/digicamtoy',
    license='GNU GPLv3',
    author='Cyril Alispach',
    author_email='cyril.alispach@gmail.com',
    description='A Monte Carlo package of the SST-1m Camera',
    package_data={'digicamtoy.utils': ['pulse_SST-1M_AfterPreampLowGain.dat']}
)
