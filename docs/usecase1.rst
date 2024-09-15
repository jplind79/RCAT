.. _use_case_1:

Use Case 1: Annual & and Seasonal Cycles
========================================

In this example annual and seasonal statistics will be calculated and in most
steps, changes are made in different sections of the configuration file,
*src/rcatool/config/config_main.ini*. It is recommended to copy this file to your
experiment folder. Detailed information of what can be configured and how can be
found in :ref:`configuration`.

How to calculate monthly and seasonal mean statistics?
******************************************************

Daily data from two models is used as input for calculation of monthly and
seasonal means at each point of their respective (native) grids. Statistical
results are written to disk in netcdf files; a separate file for each model and
statistic (and sub-region if specified). No plotting is done here.

STEP 1: Data input
..................

Under section **MODELS** specify the paths to model data. Configure for two or
more models, and select start and end years. Since the annual and seasonal
cycles will be calculated, select all 12 months.

::

   model_1 = {
        'fpath': '<path-to-folder-1>',
        'grid type': 'reg', 'grid name': '<grid-name>',
        'start year': <start year>, 'end year': <end year>, 'months': [1,2,3,4,5,6,7,8,9,10,11,12]
	'date interval start': '<yyyy>-<mm>', 'date interval end': '<yyyy>-<mm>',
     	'chunks_time': {'time': -1}, 'chunks_x': {'x': -1}, 'chunks_y': {'y': -1},
        }
   model_2 = {
        'fpath': '<path-to-folder-2>',
        'grid type': 'reg', 'grid name': '<grid-name>',
        'start year': <start year>, 'end year': <end year>, 'months': [1,2,3,4,5,6,7,8,9,10,11,12]
	'date interval start': '<yyyy>-<mm>', 'date interval end': '<yyyy>-<mm>',
     	'chunks_time': {'time': -1}, 'chunks_x': {'x': -1}, 'chunks_y': {'y': -1},
        }

In this example we will not use any observations so no modifications are needed
in the **OBS** section.

STEP 2: Variables
.................

Under **SETTINGS** set the full path to an output directory. If the
folder doesn't exist already it will be created by RCAT.

::

    output dir = <path-to-output-directory>

The key *variables* defines which variables to analyze along with some options
regarding that particular variable. Since only models will be analyzed here,
*'obs'* is set to None. Further, models will be kept at their respective grids,
thus *'regrid to'* is also set to None. Statistics is configured for T2m (*tas*)
and precipitation (*pr*) with daily data as input (*'freq'* set to *'day'*).

::

    variables = {
        'pr': {
            'freq': 'day',
            'units': 'mm', 
            'scale factor': 86400, 
            'offset factor': None,
            'accumulated': True, 
            'obs': None, 
            'obs scale factor': None,
            'obs freq': 'day',
            'var names': None,
            'regrid to': None
            },
        'tas': {
            'freq': 'day', 
            'units': 'K', 
            'scale factor': None, 
            'offset factor': None,
            'accumulated': False, 
            'obs': None, 
            'obs scale factor': None,
            'obs freq': 'day',
            'var names': None,
            'regrid to': None
            },
        }


::

    regions = ['Norway', 'British Isles', 'Spain']

*regions* is a list of pre-defined regions -- see available regions in *<path-to-RCAT>/utils/polygon_files* folder (see also :ref:`Polygons` module).
Statistics will be selected for the specified sub-regions.


STEP 3: Select statistics
.........................

Under **STATISTICS** seasonal and annual cycles are chosen.

::

    stats = {
    	'annual cycle': 'default',
    	'seasonal cycle': {'thr': {'pr': 1.0}},
        }

The *'default'* property means that default options for the particular statistic are used.
All default options can be seen in the *default_stats_config* function in
:ref:`RCAT Statistics <stats_control_functions>`. For seasonal cycle, we choose to
use a threshold for precipitation of *1.0* and so calculation is only based on wet days.


STEP 4: Plotting
................

Set validation plot to True if you want plots to be produced. 

::

    validation plot = True/False

If plotting, you need to set some map and line plot configurations, the code below is an example.
Leave *map model domain* empty.

::

    map projection = 'LambertConformal'
    map configuration = {
        'central_longitude': 10,
        'central_latitude': 60.6,
        'standard_parallels': (60.6, 60.6),
     }
    map extent = [4, 29, 52, 72]  # Extent of the map; [lon_start, lon_end, lat_start, lat_end]
    map gridlines = False
    map grid config = {'axes_pad': 0.3, 'cbar_mode': 'each', 'cbar_location': 'right',
                  	  'cbar_size': '5%%', 'cbar_pad': 0.05}
    map plot kwargs = {'filled': True, 'mesh': True}
    map model domain =
    
    # Line plot settings
    line grid setup = {'axes_pad': (2., 2.)}
    line kwargs = {'lw': 2}


STEP 5: Configure cluster
.........................

Under the **CLUSTER** section one should specify which type of cluster to use.
Here, it is configured for a SLURM cluster. *nodes* specify the number of nodes
to be used. In *cluster kwargs* a number of different options can be set (here
specific for SLURM), for example walltime which is set to 2 hours.

::

    cluster type = slurm
    nodes = 10
    cluster kwargs = {'walltime': '02:00:00', 'cores': 24, 'memory': '128GB'}


STEP 6: Run RCAT
................

To run the analysis run from terminal (see *Run RCAT* in :ref:`configuration`):

     .. code-block:: bash

        python <path-to-RCAT>/src/rcatool/runtime/RCAT_main.py -c config_main.ini


If successfully completed, output statistics netcdf files will be located in
the sub-folder *stats* under the user-defined output directory. An *img* folder
is also created, and produced figures are saved there if *validation plot* is
set to True.


Adding comparison to observations and visualize results
*******************************************************

In order to include observations and visualize the end results, follow the
procedure as in the previous example with the following changes introduced:

#. Under **OBS** section, choose same years and months as models

    ::
    
        start year = <start year>
        end year = <end year>
        months = [1,2,3,4,5,6,7,8,9,10,11,12]
        date interval start = None
        date interval end = None

#. The *variables* property in **SETTINGS** section shall be modified:

    - Include observations; *'obs': ['EOBS', 'ERA5']*. Also, scale
      factors are now included for observations as well.

    - Since models and observations will be compared, taking differences, the data
      must be on the same grid. Therefore, set *'regrid to': 'ERA5'*. This means that
      all data will be interpolated to the *ERA5* grid. Further, the *'regrid method'*
      needs to be set -- *bilinear* for T2m and *conservative* for pr.

    ::
    
        variables = {
            'pr': {
                'freq': 'day', 
                'units': 'mm', 
                'scale factor': 86400, 
                'offset factor': None,
                'accumulated': True, 
                'obs': ['EOBS20', 'ERA5'], 
                'obs scale factor': [86400, 86400], 
                'obs freq': 'day',
                'var names': None,
                'regrid to': 'ERA5', 
                'regrid method': 'conservative'
                },
            'tas': {
                'freq': 'day', 
                'units': 'K', 
                'scale factor': None, 
                'offset factor': None,
                'accumulated': False, 
                'obs': ['EOBS20', 'ERA5'], 
                'obs scale factor': None, 
                'obs freq': 'day',
                'var names': None,
                'regrid to': 'ERA5', 
                'regrid method': 'bilinear'
                },
            }

#. Run RCAT again (STEP 6 above).
