Installation
============

RCAT is a python3 based tool and the easiest way to get started is by using the
`Conda <https://conda.io/projects/conda/en/latest/index.html>`_ framework.
If you don't have conda installed follow the installation guide from start.
Otherwise you can follow from `RCAT environment`_ .

N.B. conda-forge channel need to be added in $HOME/.condarc

Miniconda
---------

Install `Miniconda <https://conda.io/projects/conda/en/latest/user-guide/install/linux.html>`_

**Add conda-forge channel**

* create $HOME/.condarc file

::

    channels:
        - conda-forge
        - defaults

**Update conda**

* conda update conda

RCAT environment
----------------

**Create environment**

* conda create -n rcat

**Activate environment**

* conda activate rcat

**Install dependencies**

It's important that you follow the order of the installation list below due to
a bug in the esmpy module.

* conda install esmpy
* conda install xesmf dask
* conda install netcdf4 dask-jobqueue matplotlib basemap
