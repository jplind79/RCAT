Installation
============

RCAT requires Python>=3.10. The major dependencies are `dask_`, `xarray_` and
`ESMPy_` (via the regridding tool `xESMF_`), and the best way to install them
is using Conda_ or Mamba_.  If you don't have conda installed follow the
installation guide from start.  Otherwise you can follow from `RCAT
environment`_ .

Miniconda
---------

Install `Miniconda <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_

**Add conda-forge channel**

* create $HOME/.condarc file

::

    channels:
        - conda-forge
        - defaults

**Update conda**

.. code-block:: bash

    $ conda update -n base conda

RCAT environment
----------------

When having installed Conda (or Mamba), the next step is to create and install
the RCAT environment. We recommend to create a new, clean Conda environment. 

**Create environment**

* environment.yml
The quickest way is to create and install the environment is to use the
environment.yml file

.. code-block:: bash

    $ conda env create --file environment.yml
    $ conda activate rcatenv

* Install dependencies separately
An alternative is to install RCAT and the dependencies separately:

.. code-block:: bash

    $ conda create -n rcatenv
    $ conda activate rcatenv
    $ conda install xarray, esmpy, xesmf>=0.8, dask, dask-jobqueue, matplotlib, cartopy, netcdf4, h5netcdf

RCAT package can then be installed from PyPI using pip:

.. code-block:: bash

    $ conda install pip
    $ pip install rcatool

Note on netCDF4 and h5netcdf
----------------------------

Xarray uses `netCDF4 <https://unidata.github.io/netcdf4-python/>`_ as the
default backend (depending on available dependencies) to read netcdf files.
However, issues have occurred recently when opening multiple files in parallel
(with ``dask.delayed``) with ``netCDF4>=1.6.1``, see e.g. this
`github issue <https://github.com/pydata/xarray/issues/7079>`_.
The suggested alternative here is to use the `h5netcdf <https://h5netcdf.org/>`_ instead,
which does not depend on the ``netCDF-C`` library, but it may have some impact
on the performance.


.. _xarray: http://xarray.pydata.org
.. _dask: https://docs.dask.org/en/stable/
.. _ESMPy: http://earthsystemmodeling.org/esmpy/
.. _xESMF: https://xesmf.readthedocs.io/en/latest/
.. _Conda: https://docs.conda.io/
.. _Miniconda: https://docs.anaconda.com/miniconda/
.. _Miniforge: https://github.com/conda-forge/miniforge
.. _Mamba: https://mamba.readthedocs.io/en/latest/index.html
.. _PyPI: https://pypi.python.org/pypi
