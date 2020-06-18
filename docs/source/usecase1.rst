.. _use_case_1:

Use Case 1: Annual & and Seasonal Cycles
========================================

In this example annual and seasonal statistics will be calculated and in most
steps, changes are made in different sections of the configuration file,
*<path-to-RCAT>/config/config_main.ini*. It is recommended to copy this file to your
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

Under section **MODELS** you'll specify the path to model data. Configure for
two models -- *arome* and *aladin* -- using the same time period and months.
Since the annual and seasonal cycles will be calculated, all months are chosen.

::

   arome = {
        'fpath': '/nobackup/rossby21/rossby/joint_exp/norcp/NorCP_AROME_ERAI_ALADIN_1997_2017/netcdf',
        'start year': 1998, 'end year': 2002, 'months': [1,2,3,4,5,6,7,8,9,10,11,12]
        }
   aladin = {
        'fpath': '/nobackup/rossby21/rossby/joint_exp/norcp/NorCP_ALADIN_ERAI_1997_2017/netcdf',
        'start year': 1998, 'end year': 2002, 'months': [1,2,3,4,5,6,7,8,9,10,11,12]
        }

In this example we will not use any observations so no modifications are needed
in the **OBS** section.

STEP 2: Variables
.................

Under **SETTINGS** the full path to output directory should be defined. If
folder doesn't exist already it will be created by RCAT.

::

    output dir = /nobackup/rossby22/sm_petli/analysis/test_analysis

The key *variables* defines which variables to analyze along with some options
regarding that particular variable. Since only models will be analyzed here,
*'obs'* is set to None. Further, models will be kept at their respective grids,
thus *'regrid to'* is also set to None. Statistics is configured for T2m (*tas*)
and precipitation (*pr*) with daily data as input (*'freq'* set to *'day'*).

::

    variables = {
        'pr': {'freq': 'day', 'units': 'mm', 'scale factor': None, 'accumulated': True, 'obs': None, 'regrid to': None},
        'tas': {'freq': 'day', 'units': 'K', 'scale factor': None, 'accumulated': False, 'obs': None, 'regrid to': None},
        }


::

    regions = ['Fenno-Scandinavia']

*regions* is a list of pre-defined regions -- see available regions in *<path-to-RCAT>/rcat/utils/polygon_files* folder (see also :ref:`Polygons` module).
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


STEP 4: No plotting
...................

Set validation plot to false -- no plotting done in this example.

::

    validation plot = False


STEP 5: Configure cluster
.........................

Under the **CLUSTER** section one should specify which type of cluster to use.
Here, it is configured for a SLURM cluster. *nodes* specify the number of nodes
to be used. In *cluster kwargs* a number of different options can be set (here
specific for SLURM), for example walltime which is set to 2 hours.

::

    cluster type = slurm
    nodes = 10
    cluster kwargs = {'walltime': '02:00:00'}


STEP 6: Run RCAT
................

To run the analysis run from terminal (see *Run RCAT* in :ref:`configuration`):

     .. code-block:: bash

        python <path-to-RCAT>/rcat/RCAT_main.py -c config_main.ini


If successfully completed, output statistics netcdf files will be located in the
sub-folder *stats* under the user-defined output directory. An *img* folder
is also created, however, it will be empty as no plotting have been done.


Adding comparison to observations and visualize results
*******************************************************

In order to include observations and visualize the end results, follow the
procedure as in the previous example with the following changes introduced:

#. Under **OBS** section, choose same years and months as models

    ::
    
        start year = 1998
        end year = 2002
        months = [1,2,3,4,5,6,7,8,9,10,11,12]

#. The *variables* property in **SETTINGS** section shall be modified:

    - Include observations; *'obs': ['EOBS20', 'ERA5']*. Also, scale
      factors are now included for observations as well.

    - Since models and observations will be compared, taking differences, the data
      must be on the same grid. Therefore, set *'regrid to': 'ERA5'*. This means that
      all data will be interpolated to the *ERA5* grid. Further, the *'regrid method'*
      needs to be set -- *bilinear* for T2m and *conservative* for pr.

    ::
    
        variables = {
            'pr': {'freq': 'day', 'units': 'mm', 'scale factor': None, 'accumulated': True, 'obs': ['EOBS20', 'ERA5'], 'obs scale factor': [86400, 86400], 'regrid to': 'ERA5', 'regrid method': 'conservative'},
            'tas': {'freq': 'day', 'units': 'K', 'scale factor': None, 'accumulated': False, 'obs': ['EOBS20', 'ERA5'], 'obs scale factor': None, 'regrid to': 'ERA5', 'regrid method': 'bilinear'},
            }

#. Under **PLOTTING**, *validation plot* should be set to *True* to enable plotting.
   It is possible to configure the visualization in different ways, for
   example various map configurations in map plots or the looks of line plots.
   However, for simplicity here, the default configurations will be used, which means
   setting all properties to an empty dictionary ({}).

    ::
    
        validation plot = True
    
        map configure = {}
        map grid setup = {}
        map kwargs = {}
        
        line grid setup = {}
        line kwargs = {}

With these modifications in place, run RCAT again (STEP 6 above).
