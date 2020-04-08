# Regional Climate Analysis Tool (RCAT) #

RCAT is an analysis tool primarily developed for analysis of regional climate
models, but may also be used for global models as well. It's purely written in
Python where we aim for a modular code in a functional style. The purpose is
to get an efficient and structured way of collaboration within our model
developers but also for non pythonists that want to use the tool for standard
climate data analysis.  

## Documentation ##

To get started, the most relevant reads (under *docs*) are:

* [README_configuration](docs/README_configuration.md)
* [README_statistics](docs/README_statistics.md)

For more information and tutorials see the documentation at [Read the
Docs](https://regional-climate-analysis-tool.readthedocs.io "readthedocs.io")

### Dependencies ###

Many modules are included in the repo but some standard python modules are a prerequisite:

* python3 (python2 is not supported)
* esmpy
* xesmf
* dask (for parallelization purposes)
* dask_jobqueue
* matplotlib
* basemap
* netcdf4
* palettable (for color options in visualization)

## Contribution guidelines ##

* Writing tests
* Code review
* Other guidelines

## Issues and bug reports ##

Bug reports, ideas, wishes are very welcome. Please report any issues using the links for Wiki and Issues.

## Who do I talk to? ##

See [AUTHORS](AUTHORS.md)
