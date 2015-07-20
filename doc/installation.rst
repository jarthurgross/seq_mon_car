.. Installation instructions

Installation
============

Linux
-----

Anaconda
++++++++

Installation is easiest if you already have a working `Anaconda`_ installation.
In this case, simply navigate to the root directory of the project where the
setup script ``setup.py`` is and execute ``python setup.py install``.

Non-Anaconda
++++++++++++

If you don't have Anaconda, there may be some additional complications. This
package depends upon scipy, matplotlib, and h5py, which have some external
C-library dependencies. If you don't have those python packages pre-installed,
you may need the following libraries in order to build them when invoking the
setup script:

* scipy

  * LAPACK (``liblapack-dev``)
  * BLAS (``libopenblas-dev``)
  * Fortran compiler (``gfortran``)

* h5py

  * HDF5 shared library with development headers (1.8.4 or newer, packaged as
    ``libhdf5-dev`` or similar).

* matplotlib

  * freetype (``libfreetype6-dev``)

Once you have that installed you should be able to successfully run
``python setup.py install``, although I can't guarantee it. The code
successfully builds on Travis-CI, and I only had to manually instruct it to
install LAPACK, BLAS, a Fortran compiler, and HDF5.

Issues
------

Building seems to work fine with Python 2.7 when starting with a base Anaconda
installation. When building with a minimal Python installation (i.e. in an
environment created by ``conda create --name new_env python=2.7`` or ``conda
create --name new_env python=3``, there seem to be problems building matplotlib.
In Python 3 I get ``TypeError: unorderable types: str() < int()`` at the line
reading ``if self.version < other.version:``, and in Python 2 I get::

  /usr/bin/ld: cannot find -lnpymath

.. _Anaconda: http://continuum.io/downloads
