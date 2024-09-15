.. _configuration:

==================
RCAT Configuration
==================

The main setup is done in the **config_main.ini** file (located under
``src/rcatool/config/``). Here you will set up paths to model data, variables
to analyze and how (choosing statistics), observations to compare with etc. In
other words, this is your starting point in applications of RCAT.

Set up folder structure
=======================
If you don't want to pollute your cloned git repository we suggest to
create a new folder for your analysis and copy the main RCAT configuration 
file to the new folder.

.. code-block:: bash

    $ mkdir -p $HOME/rcat_analysis/test
    $ cd $HOME/rcat_analysis/test
    $ cp <path-to-RCAT>/src/rcatool/config/config_main.ini .

Running RCAT 
============
When you have done your configuration and saved config_main.ini you can
start the analysis step. The main program is located in the *rcat* directory
and called RCAT_main.py. See point 1: :ref:`Setup folder structure
<configuration>` and run main RCAT_main.py from your analysis folder.


.. code-block:: bash

   $ python <path-to-RCAT>/src/rcatool/runtime/RCAT_main.py -c config_main.ini

.. note::
   The configuration file does not need to be named *config_main.ini*. Any file
   name can be used, and a suggestion is to choose names appropriate for the
   specific application.

Settings in config_main.ini
===========================
A configuration ``.ini`` file has a specific structure based
on sections, properties and values. The RCAT config_main.ini file consists of a handful
of these sections, for example **MODELS**, under which you specify certain
properties or values. The latter may in some cases be common structures
used in python like lists or dictionaries. Below follows a description of
each of the sections needed to setup the analysis.

MODELS
------
Here you specify the path to model data. At the moment a specific
folder structure is anticipated, with sub-folders under ``fpath``
according to output frequency; ``fpath/day``, ``fpath/6hr``, ``fpath/15min``, etc.
Names of these sub-folders are inherited from the ``freq`` property set
under variables in the **SETTINGS** section.

::

    model_name = {
    	'fpath': '/path/to/netcdf/files',
    	'grid type': 'reg', 'grid name': 'label_grid_model_1',
    	'start year': 1998, 'end year': 2000, 'months': [1,2,3,4,5,6,7,8,9,10,11,12],
    	'date interval start': '<yyyy>-<mm>', 'date interval end': '<yyyy>-<mm>',
        'chunks_time': {'time': -1}, 'chunks_x': {'x': -1}, 'chunks_y': {'y': -1},
    	}

Selection of time periods can be done in two ways; i) specifying start and end
years plus months to consider, and ii) specification of a date interval, i.e. a
start and end date defined as strings on the format 'yyyy-mm'. If both are set option ii) takes
precedence over i). Setting date values to ``None`` instead, option i) will be
used.

There is also the possibility to specify how you want to chunk (split) the data
(for parallel computations using ``dask.delayed``) with the *chunks_time*,
*chunks_x* and *chunks_y* keys. The value set here will be the size of the
individual chunks in each of the dimensions. If the value is set to -1, default
chunking is used, normally chunked according to the time dimension of each
opened input file. Read more about chunking `here <https://docs.dask.org/en/stable/array-chunks.html>`_.

Here you also set a couple of grid specifications - namely *grid type*
and *grid name*. Grid type defines the type of grid the model data is currently on;
it can be set to either *rot* or *reg*. The former means that
model data is on a rotated grid and the latter that it is on a non-rotated
grid (i.e. regular, rectilinear, curvilinear).

If data is on rotated grid RCAT will "de-rotate" the grid. However, it requires
that model files include coordinate variables in accordance with CF conventions;
*rlon*/*rlat* for longitudes and latitudes as well as the variable
*rotated_pole* with attributes *grid_north_pole_longitude* and
*grid_north_pole_latitude*. Grid name is a user defined label for the grid. If
data is to be remapped to this model grid, the output filenames from RCAT
analysis will include this specified grid name.

Here is another example with two simulations; one for the historic period and
one future scenario.

::

    model_his = {
        'fpath': '/path/to/model_1/data',
        'grid type': 'reg', 'grid name': 'FPS-ALPS3',
        'start year': 1985, 'end year': 2005, 'months': [6,7,8]
    	'date interval start': '<yyyy>-<mm>', 'date interval end': '<yyyy>-<mm>',
        'chunks_time': {'time': -1}, 'chunks_x': {'x': -1}, 'chunks_y': {'y': -1},
        }
    model_scn = {
        'fpath': '/path/to/model_2/data',
        'grid type': 'reg', 'grid name': 'FPS-ALPS3',
        'start year': 2080, 'end year': 2100, 'months': [6,7,8]
    	'date interval start': '<yyyy>-<mm>', 'date interval end': '<yyyy>-<mm>',
        'chunks_time': {'time': -1}, 'chunks_x': {'x': -1}, 'chunks_y': {'y': -1},
        }

More models can be added to the section, but note that the first defined model
(e.g.  *model_his* in the above example) will be the reference model. That is,
if validation plot is True, and no observations are specified (see below), the
anomaly/difference plots will use the first specified model in this section as
the reference data.

.. note:: 
    If you want to see how RCAT uses defined file paths and other
    information to retrieve lists of model data files, see the
    *get_mod_data* function in *src/rcatool/runtime/RCAT_main.py*. 

OBS
---
If observations should be included in the analysis, you will need to
specify a meta data file by setting the full path to
*observations_metadata_NN.py* (located under *src/rcatool/config*).
*NN* is any label that signifies the observation meta data for a
specific location or system (for example a HPC system). If no such
meta data file exists yet, it should be created
(SAMPLE_observations_metadata.py in the same folder can be used as a template) and
modified. **N.B.** Changes should only be done in the *obs_data* function, where
reference data sets are specified.

In addition, in the **OBS** section the time period and months for obs data shall
be defined. As for the models there is also the option to set a date interval
instead, which, if set, takes precedence over years/months settings.
The same time period will be applied to all observations included in the analysis. Which
specific observations to include is not defined here, but in the
**SETTINGS** section, in the variables properties.

SETTINGS
--------

output dir
**********
The path for the output (statistics files, plots). If
you re-run the analysis with the same output directory, you will
prompted to say whether to overwrite existing output. "overwrite" does
not mean that existing folder will be completely overwritten (deleted
and created again). The existing folder structure will be kept intact
together with output files. However, potentially some output
(statistics/figure files) with same names will be overwritten.

variables
*********
This is a key settings in the configuration file. The
value of this property is represented by a dictionary; the keys are
strings of variable names ('pr', 'tas', ...) and the value of each key
(variable) is another dictionary consisting of a number of specific
settings (the same for all variables):

::

 variables = {
    'tas': {
       'var names': {'model_1': {'prfx': 'tas', 'vname': 'var167'}},
       'freq': 'day',
       'units': 'K',
       'scale factor': None,
       'offset factor': -273.15,
       'accumulated': False,
       'obs': 'EOBS',
       'obs scale factor': None,
       'obs freq': 'day',
       'regrid to': 'model_2',
       'regrid method': 'bilinear'},
    'psl': {
       'var names': None,
       'freq': '3hr',
       'units': 'hPa',
       'scale factor': 0.01,
       'offset factor': None,
       'accumulated': False,
       'obs': None,
       'obs scale factor': None,
       'obs freq': 'day',
       'regrid to': None,
       'regrid method': 'bilinear'},
    'pr': {
       'var names': None,
       'freq': '1hr',
       'units': 'mm',
       'scale factor': 3600,
       'offset factor': None,
       'accumulated': False,
       'obs': 'ERA5',
       'obs scale factor': 86400,
       'obs freq': 'day',
       'regrid to': 'ERA5',
       'regrid method': 'conservative'},
 }

- **var names**:

  Variable names specified in the top key of *variables*
  usually refers to common names defined in CF conventions. However,
  there might be cases where either the variable name specified in the
  file name or of the variable in the file differ from these
  conventions. Var names provides an option to account for this; it is
  specified as a dictionary with keys *prfx* and *vname* for the file
  name prefix and variable name respectively. If file formats follows
  the conventions, and thus have same prefix and name as the top key
  variable name, *var names* should be set to *None*. See code snippet
  above for examples of both types of settings.

- **freq**: 

  A string of the time resolution of input model data. The
  string should match any of the sub-folders under the path to model
  data, e.g. 'day', '1hr', '3hr'. In effect, you may choose different
  time resolutions for different variables in the analysis.

- **units**: 

  The units of the variable data (which will appear in
  figures created in RCAT, and thus should reflect the units after
  data have been manipulated through the analysis).

- **scale factor**: 

  A numeric factor (integer/float) that model data is
  multiplied with, to convert to desired units (e.g. from J/m2 to
  W/m2) and to ensure that all data (model and observations) have the
  same units. If no scaling is to be done, set value to None. An
  arithmetic expression is not allowed; for example if data is to be
  divided by 10 you cannot define factor as 1/10, it must then be 0.1.

  A list of scale factors can be defined if different scale factors should be
  used for the different models specified under **MODELS** section. Thus, for 
  a list of scale factors [f1, f2, f3, ...], these will applied as f1*model_1,
  f2*model_2, etc. Note that the list of scale factors need then to be of the
  same length as number of specified models. 

- **offset factor**: 

  The same as for *scale factor*, although the *offset factor* is added to the
  model data. A negative value will then subtract the factor from model data.

- **accumulated**: 

  Boolean switch identifying variable data as
  accumulated fields or not. If the former (True), then data will be
  de-accumulated "on the fly" when opening files of data.

- **obs**: 

  String or list of strings with acronyms of observations to be
  included in the analysis (for the variable of choice, and therefore
  different observations can be chosen for different variables).
  Available observations, and their acronyms, are specified in the
  *src/rcatool/config/observations_metadata_NN.py* file. In this
  file you can also add new observational data sets. 

- **obs scale factor**: 

  The same as *scale factor* above but applied to observations. If
  multiple observations are defined, some of which would need
  different scale factors, a list of factors can be provided. However,
  if the same factor should be used for all observations, it is enough
  to just specify a single factor.

- **obs freq**: 

  A string of the time resolution of observation data. 

- **regrid to**:

  If data is to be remapped to a common grid, you specify
  either the name (model name or observation acronym) of a model
  defined under **MODELS** section or an observation defined under
  *obs* key. Or, if an external grid should be used, it can be set to a
  dictionary with the *name* and *file* keys. *name* has the same
  purpose as *grid name* in the **MODELS** section above.
  The value of
  *file* must be the full path to a netcdf file that at least contains
  *lon* and *lat* variables defining the target grid. If no remapping
  is to be done, set *regrid to* to None.

- **regrid method**: 

  String defining the interpolation method: 'conservative' or 'bilinear'.

regions
*******
A list of strings with region names, defining
geographical areas data will be extracted from. If set, 2D statistical
fields calculated by RCAT will be cropped over these regions (polygons), and in
line plots produced in RCAT the statistical values will be averaged over
and plotted for each of the regions. Read more about
how to handle regions and polygons in RCAT :ref:`here <polygons_howto>`.
Set ``regions=`` or ``regions=[]`` if not using any regions.

STATISTICS
----------
This is another important section of the analysis configuration. Therefore, the
description of this segment is given separately, see :doc:`RCAT
Statistics </statistics>`

PLOTTING
--------
This section is intended for the case you want to perform a general
evaluation/validation of the model. This means that (for the moment) a
set of standards plots (maps and line plots) can be done by RCAT for a
set of standard statistical output: annual, seasonal and diurnal
cycles, pdf's, percentiles and ASoP analysis. If plotting procedures
for other statistics is wished for, they need to be implemented in the
RCAT :doc:`plotting module <plots>`.


validation plot
***************
If validation plot is set to True, standard plots
will be produced for the defined statistics. Otherwise, plotting can
be done elsewhere using the statistical output files (netcdf format)
created by RCAT.

moments plot config
*******************
Here one configures the plots for *moments* statistics (see :doc:`RCAT Statistics </statistics>`).
At the moment, only timeseries or map plots can be produced (``plot type: 'timeseries'`` and ``plot type: 'map'`` respectively).
For timeseries, one can also choose to add either running mean values and/or
linear trends (using numpy's ``polyfit`` and ``poly1d`` functions). To use
running means, set the window size (in terms of time units from the
calculated moment statistics), otherwise set to False. The switch for
trendlines is True/False.  

map projection
**************
Define here the projection to use in the map plots. See Cartopy's
documentation for available `projections <https://scitools.org.uk/cartopy/docs/v0.15/crs/projections.html>`_. 
For example, to use Lambert Conformal conic projection, set ``map
projection = 'LambertConformal'``. 

map configuration
*****************
Settings for the specified map projection. 
See Cartopy documentation for available options of the projections.
The options shall be set in dictionary format, e.g.

::

    map configuration = {
    'central_longitude': 10,
    'central_latitude': 60.6,
    'standard_parallels': (60.6, 60.6),
 }

map extent
**********
Define the geographical extent of the maps. The value of *map extent* is a
list of values corrsponding to [longitude_start, longitude_end,
latitude_start, latitude_end]

map gridlines
*************
Whether to plot labeled latitude/longitude lines in map plots (True/False)

map grid config
***************
Settings for the map plot panel configuration, for
example whether to use a single colorbar or not (cbar_mode), colorbar location,
the padding between panels, etc.

::

 map grid config = {
    'axes_pad': 0.3,
    'cbar_mode': 'each',
    'cbar_location': 'right',
    'cbar_size': '5%%',
    'cbar_pad': 0.05
    }

The *map grid cofig* dictionary constitute the keyword arguments (kwargs)
for the *map_setup* function in the :doc:`plots module <plots>`. See this
function for more information on other available options.

map plot kwargs
***************
Additional keyword arguments to be added in the
matplotlib contour plot call, see the make_map_plot function in
the :doc:`plotting module <plots>`.

map model domain
****************
For the map plots, all data is masked by the spatial extent of the reference model 
(first model listed under **MODELS**), mainly to achieve appropriate levels for
the data values. The *map model domain* can be set to choose another model from
**MODELS** for the data masking (other than the first one).

line plot settings
******************
Likewise, settings for line plots can be made,
e.g. line widths and styles as well as axes configurations. There are
a number of functions in the :doc:`plotting module <plots>` that
handles line/scatter/box plots, see for example the fig_grid_setup and
make_line_plot functions.

::

   line grid setup = {'axes_pad': (11., 6.)}
   line kwargs = {'lw': 2.5}

CLUSTER
-------
The last section control the cluster type. You can choose between local
machine and SLURM at the moment.

cluster type
************
Choose ``local`` for running on your local machine and ``slurm`` if you want to
run RCAT on a HPC with a SLURM job scheduler.  For local machine no other
settings need to be made in this section, for SLURM see information below.

*SLURM*
    RCAT uses `Dask <https://docs.dask.org/>`_ to perform file managing
    and statistical analysis in an efficient way through parallelization.
    When using Dask on queuing systems like PBS or SLURM,
    `Dask-Jobqueue <https://dask-jobqueue.readthedocs.io>`_ provides an
    excellent interface for deploying and managing such a work flow. To properly use Dask and Dask-Jobqueue on an HPC system you need
    to provide some information about the system and how you plan to use
    it. By default, when Dask-Jobqueue is first imported a configuration
    file is placed in ~/.config/dask/jobqueue.yaml. Below is an example of the
    content in this file:
    
    .. code-block:: yaml

       jobqueue:
           slurm:
           name: dask-worker

           # Dask worker options
           cores: 16
           memory: "64 GB"
           processes: 1

           interface: ib0
           death-timeout: 60
           local-directory: $TMP

           # SLURM resource manager options
           queue: null
           project: null
           walltime: '01:00:00'
           job-extra-directives: ['--exclusive']

    When default settings have been set up, the main properties that you
    usually want to change in the **CLUSTER** section are the number of nodes
    to use, ``nodes`` and walltime:

    ::

       nodes = 15
       slurm kwargs = {'walltime': '02:00:00', 'memory': '256GB'}
