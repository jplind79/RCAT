# Regional Climate Analysis Tool (RCAT) #

The Regional Climate Analysis Tool (RCAT) is an analysis tool originally
developed for evaluation of regional climate models, but may be used to analyze
output from most types of weather and climate models. It is purely written in
Python where the aim is for a modular code in a functional style.

The purpose is to have an efficient and user-friendly tool to both facilitate
quick standard evaluations in model development processes, but also to perform
more in-depth climate data analysis.

The tool is adapted to new demands in regional climate modeling, where the
produced output data volumes become very large and analysis is typically
carried out on HPC systems.

## Documentation ##

To get started, the most relevant reads (under *docs*) are:

* [README_configuration](docs/README_configuration.md)
* [README_statistics](docs/README_statistics.md)

For more information and tutorials see the documentation at [Read the
Docs](https://regional-climate-analysis-tool.readthedocs.io "readthedocs.io")

### Dependencies ###

Many modules are included in the repo but some standard python modules are a prerequisite:

* python3
* esmpy
* xesmf
* dask (for parallelization purposes)
* dask_jobqueue
* matplotlib
* cartopy
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
