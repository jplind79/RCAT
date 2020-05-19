Use Case 3: Diurnal Cycles
==========================

One main advantage of RCAT is that it can run analyses of large data sets
through the use of parallelization (using the *dask* module). In principle it
involves splitting the data into several "chunks" (in the space and/or time
dimensions) and then run a set of operations on each of the chunk in parallel.
Read more about it on `dask <https://dask.org/>`_ homepage.  

Depending on the analysis you want to run on your data, you might consider
chunking your data differently. If, for example, you would like to calculate a
quantile value for the data over all time steps then you should do the chunking
in space only so that each chunk has all time steps available. Here, RCAT will
be applied to calculate diurnal cycles of some model output using different
statistical measures and how the splitting/chunking of data matters.

Similar to :ref:`Use Case 1 <use_case_1>` most changes will be done in the
configuration file, *src/config/config_main.ini*.


Calculate diurnal cycles of mean CAPE and plot the results
**********************************************************

STEP 1: Data input
..................

Under section **MODELS** specify the path to model data and set start and end
years as well as months to analyze.

::

   arome = {
        'fpath': '/nobackup/rossby21/rossby/joint_exp/norcp/NorCP_AROME_ERAI_ALADIN_1997_2017/netcdf',
        'start year': 1998, 'end year': 2002, 'months': [5,6,7,8,9]
        }
   aladin = {
        'fpath': '/nobackup/rossby21/rossby/joint_exp/norcp/NorCP_ALADIN_ERAI_1997_2017/netcdf',
        'start year': 1998, 'end year': 2002, 'months': [5,6,7,8,9]
        }

If you would like to include observations as well, set accordingly in the **OBS** section.


STEP 2: Variables
.................

Set output directory under the **SETTINGS** section.

In the key *variables* we specify in this example *pcape* (a specific model
version of CAPE) available on 3 hourly time resolution.  If only models will be
analyzed set *'obs'* to None.  *'regrid to'* is set to the coarser grid of the
two models and data is interpolated using the *bilinear* method.

Specify region(s) in the *regions* key for which statistics will be selected,
and finally plotted, for.

::

    output dir = /nobackup/rossby22/sm_petli/analysis/test_dcycle_analysis

    variables = {
        'pcape': {'freq': '3H', 'units': 'J/kg', 'scale factor': None, 'accumulated': False, 'obs': None, 'regrid to': 'aladin', 'regrid method': 'bilinear'},
        }

    regions = ['Sweden', 'Denmark', 'Norway', 'Finland']


STEP 3: Select statistics
.........................

The statistics, *diurnal cycle*, is specified under the *stats* key in the
**STATISTICS** section. Default options for diurnal cycle is found in the
*default_stats_config* function in the :doc:`statistics <rcat_statistics>`
module. In default settings, *hours* is set to all 24 hours in a day. Since the
data here is on 3 hourly resolution we specify these hours. The *stat method*
(the statistical measure) for each hour is *mean* in default and it is kept
here, and the data is chunked in the time dimension (also default so not
specified here).

::

    stats = {
       'diurnal cycle': {'hours': [0, 3, 6, 9, 12, 15, 18, 21]} 
        }


STEP 4: Plotting
................

Under **PLOTTING**, *validation plot* should be set to *True* to enable
plotting.  Plotting of diurnal cycles will be both maps (for each hour) and
line plots for specified regions.

::

    validation plot = True

    map configure = {}
    map grid setup = {}
    map kwargs = {}
    
    line grid setup = {}
    line kwargs = {'lw': 2.5}


STEP 5: Configure cluster
.........................

The number of nodes to be used in the selected SLURM cluster is set to 10
(increase if needed) and a walltime of 2 hours.

::

    cluster type = slurm
    nodes = 10
    cluster kwargs = {'walltime': '02:00:00'}


STEP 6: Run RCAT
................

To run the analysis run from terminal (see *Run RCAT* in :ref:`configuration`):

     .. code-block:: bash

        python <path-to-RCAT>/src/RCAT_main.py -c config_main.ini


Output statistics and image files will be located under the user-defined output
directory in the *stats* and *imgs* sub-folders respectively


Calculate diurnal cycles of 99th percentile CAPE values
*******************************************************

Instead of the mean value for each hour in the diurnal cycle (at any grid point
in the domain) it could be meaningful to use another statistical measure, for
example the 99th percentile. To do this, in addition to changing the *stat
method* value, one will need to have all time steps available for the
calculation and thus the *chunk dimension* should be changed from *'time'*
(default) to *'space'*:

::

    stats = {
       'diurnal cycle': {'hours': [0, 3, 6, 9, 12, 15, 18, 21], 'stat method': 'percentile 99', 'chunk dimension': 'space'} 
        }

When set, run RCAT again.
