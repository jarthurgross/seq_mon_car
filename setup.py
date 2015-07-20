from setuptools import setup

# Workaround from
# https://github.com/numpy/numpy/issues/2434#issuecomment-65252402
setup_requires=['numpy']

requires = [
        'numpy',
        'scipy',
        ]

dependency_links = [
        'https://github.com/csferrie/python-qinfer.git',
        'https://github.com/jarthurgross/bloch_distribution.git',
        'https://github.com/jarthurgross/qubit_dst.git',
        ]

setup(name='seq_mon_car',
      version='0.0',
      py_modules=['model', 'dists', 'simulation'],
      setup_requires=setup_requires,
      install_requires=requires,
      dependency_links=dependency_links,
      packages=['seq_mon_car'],
      package_dir={'seq_mon_car': 'src/seq_mon_car'},
     )
