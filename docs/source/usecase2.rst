.. _use_case_2:

Use Case 2: Probability distributions
=====================================

In the following RCAT is applied to calculate standard empirical probability
distribution functions. Similar to :ref:`Use Case 1 <use_case_1>` most changes
will be done in the configuration file, *src/config/config_main.ini*.


Create hourly PDF statistics and visualize the results
******************************************************

Outline
-------
PDF's based on hourly data for historical and scenario simulations will be
calculated for precipitation and T2m. Output statistics are then compared in
line plots for specified regions.


STEP 1: Data input
..................

Under **MODELS** section configure for two *arome* simulations -- historic
(*arome_his*) and future scenario (*arome_scn*). Thus, different years are
specified, however, in the example here months 6,7,8 are specified so that
only data for June, July and August is extracted. 

::

   arome_his = {
        'fpath': '/nobackup/rossby21/rossby/joint_exp/norcp/NorCP_AROME_ECE_ALADIN_1985_2005/netcdf',
        'start year': 1990, 'end year': 1994, 'months': [6,7,8]
        }
   arome_scn = {
        'fpath': '/nobackup/rossby21/rossby/joint_exp/norcp/NorCP_AROME_ECE_ALADIN_RCP85_2080_2100/netcdf',
        'start year': 2090, 'end year': 2094, 'months': [6,7,8]
        }

We're looking at climate change signal in the model, so the **OBS** section can be left as is.


STEP 2: Variables
.................

Set output directory under the **SETTINGS** section.

The key *variables* defines which variables to analyze along with some options
regarding that particular variable. Since only models will be analyzed here,
*'obs'* is set to None. Further, models will be kept at their respective grids,
thus *'regrid to'* is also set to None. Statistics is configured for T2m (*tas*)
and precipitation (*pr*) with hourly data as input (*'freq'* set to *'1H'*).

Specify regions the *regions* key for which statistics will be selected for.

::

    output dir = /nobackup/rossby22/sm_petli/analysis/test_pdf_analysis

    variables = {
        'pr': {'freq': '1H', 'units': 'mm', 'scale factor': None, 'accumulated': True, 'obs': None, 'regrid to': None},
        'tas': {'freq': '1H', 'units': 'K', 'scale factor': None, 'accumulated': False, 'obs': None, 'regrid to': None},
        }

    regions = ['Scandinavia']


STEP 3: Select statistics
.........................

Under **STATISTICS** *pdf* will be specified. A list of bins will be used in the pdf.
If not given here these bins will be defined automatically in RCAT by taking minimum
and maximum of input data. This can be quite crude and not so representative, so
it is suggested to define them here under the *pdf* key. They are specified in a
dictionary where the keys are input variables and the values are the respective bin
definitions. The bin definition is a list/tuple with start, stop and step values.
For example, for precpitation a list of bins starting with 0 and ending with 50
using a step of 1 is defined here.

::

    stats = {
       'pdf': {'bins': {'pr': (0, 50, 1), 'tas': (264, 312, 1)}} 
        }

Default options for pdf can be seen in the *default_stats_config* function in the
:doc:`statistics <rcat_statistics>` module.


STEP 4: Plotting
................

* Under **PLOTTING**, *validation plot* should be set to *True* to enable plotting.
  Plotting of pdf's will be line plots only (regions should therefore be
  specified). We only specify linewidths to be 2.5.

::

    validation plot = True

    map configure = {}
    map grid setup = {}
    map kwargs = {}
    
    line grid setup = {}
    line kwargs = {'lw': 2.5}


STEP 5: Configure SLURM
.......................

We will use 20 nodes (increase if needed) and a walltime of 2 hours.

::

    nodes = 20
    slurm kwargs = {'walltime': '02:00:00'}


STEP 6: Run RCAT
................

To run the analysis run from terminal (see *Run RCAT* in :ref:`configuration`):

     .. code-block:: bash

        python $HOME/git/rcat/src/main.py -c config_main.ini


Output statistics files will be located in the sub-folder *stats* under the
user-defined output directory.


Calculate PDF's for daily maximum values instead
************************************************

Outline
-------
Imagine one would like to do the same kind of statistical analysis as above,
however, with a different temporal resolution and/or time statistic on the input
data. For example, let's assume that pdf's should be calculated for daily
maxmimum data instead. How can this be achieved?

This can be done during RCAT runtime, using an option in the *stats* property
(under **SETTINGS**) called *resample resolution*. It is specified by a
list/tuple with two locations; the first index represents the time resolution
sought after and the second location the statistic used for each sample in the
resampling. In the example here data is resampled into daily maximum values:

::

    stats = {
       'pdf': {'bins': {'pr': (0, 50, 1), 'tas': (264, 312, 1)}, 'resample resolution': ['D', 'max']} 
        }

When set, run RCAT again.
