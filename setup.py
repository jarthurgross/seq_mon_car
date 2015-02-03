from setuptools import setup

requires = [
        'numpy',
        'scipy',
        'qinfer',
        'bloch_distribution',
        'qubit_dst',
        ]

setup(name='seq_mon_car',
      version='0.0',
      py_modules=['model', 'dists', 'simulation'],
      install_requires=requires
     )
