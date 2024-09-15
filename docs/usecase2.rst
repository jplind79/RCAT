.. _use_case_2:

Use Case 2: Probability distributions
=====================================

In the following RCAT is applied to calculate standard empirical probability
distribution functions, PDF's. 

Create hourly PDF statistics and visualize the results
******************************************************

The target of this example is to calculate PDF's based on hourly data
of precipitation and T2m for the summer season. 


STEP 1: Data input
..................

Under section **MODELS** specify the paths to model data. Configure for two or
more models, and select start and end years. Select months 6,7,8 in order to extract 
data for June, July and August.

::

   model_1 = {
        'fpath': '<path-to-folder-1>',
        'grid type': 'reg', 'grid name': '<grid-name>',
        'start year': <start year>, 'end year': <end year>, 'months': [6,7,8]
	'date interval start': '<yyyy>-<mm>', 'date interval end': '<yyyy>-<mm>',
     	'chunks_time': {'time': -1}, 'chunks_x': {'x': -1}, 'chunks_y': {'y': -1},
        }
   model_2 = {
        'fpath': '<path-to-folder-2>',
        'grid type': 'reg', 'grid name': '<grid-name>',
        'start year': <start year>, 'end year': <end year>, 'months': [6,7,8]
	'date interval start': '<yyyy>-<mm>', 'date interval end': '<yyyy>-<mm>',
     	'chunks_time': {'time': -1}, 'chunks_x': {'x': -1}, 'chunks_y': {'y': -1},
        }


We can skip the observations here and ignore **OBS** settings.


STEP 2: Variables
.................

Set output directory under the **SETTINGS** section.

The key *variables* defines which variables to analyze along with some options
regarding that particular variable. Since only models will be analyzed here,
*obs* is set to None. Further, models will be kept at their respective grids,
thus *regrid to* is also set to None. Statistics is configured for T2m (*tas*)
and precipitation (*pr*) with hourly data as input (*freq* set to *1hr*).

Specify regions the *regions* key for which statistics will be selected for.

::

    output dir = <path-to-output-directory>

    variables = {
        'pr': {
            'freq': '1hr',
            'units': 'mm', 
            'scale factor': 3600, 
            'offset factor': None,
            'accumulated': True, 
            'obs': None, 
            'obs scale factor': None,
            'obs freq': 'day',
            'var names': None,
            'regrid to': None
            },
        'tas': {
            'freq': '1hr', 
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

    regions = ['Germany', 'France']


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

See the *default_stats_config* function in :ref:`RCAT Statistics
<stats_control_functions>` module for the default options for pdf.


STEP 4: Plotting
................

* Under **PLOTTING**, set *validation plot* to *True* to enable plotting.
  Plotting of pdf's will be line plots only (regions should therefore be
  specified). We only specify linewidths to be 3.

::

    validation plot = True

    line grid setup = {}
    line kwargs = {'lw': 3}


STEP 5: Configure cluster
.........................

::

    cluster type = slurm
    nodes = 10
    cluster kwargs = {'walltime': '02:00:00', 'cores': 24, 'memory': '128GB'}


STEP 6: Run RCAT
................

Run the analysis run from the command line (see *Run RCAT* in :ref:`configuration`):

     .. code-block:: bash

        python <path-to-RCAT>/src/rcatool/runtime/RCAT_main.py -c config_main.ini


Output statistics files will be located in the sub-folder *stats* under the
user-defined output directory.


Calculate PDF's for daily maximum values instead
************************************************

Say you would like to do the same statistical analysis as above,
however, with a different temporal resolution and/or time statistic on the input
data. For example, let's assume that pdf's should be calculated for daily
maxmimum data instead. How can this be achieved?

This can be done using an option in the *stats* property
(under **SETTINGS**) called *resample resolution*. It is specified by a
list/tuple with two locations; the first index represents the time resolution
sought after and the second location the statistic used for each sample in the
resampling. In the example here data is resampled into daily maximum values:

::

    stats = {
       'pdf': {'bins': {'pr': (0, 50, 1), 'tas': (264, 312, 1)}, 'resample resolution': ['D', 'max']} 
        }

When set, run RCAT again.
