Use Case 4: Rain-on-snow events
===============================

**N.B.**
For this exercise, one might need to install the bottleneck package

::

    $ conda install bottleneck

In this example we will consider more than one variable in the statistical
calculation, by estimating the frequency of rainfall days when there is snow on
the ground -- so called *rain-on-snow* events.

Typical climate model output variables will be used, namely total precipitation
(*pr*), solid precipitation (*prsn*) and surface snow amount (water equivalent;
*snw*). Other variables (and temporal resolutions) could be used, for example
the liquid precipitation. However, we use this set of variables to test certain
processing functions in RCAT.

STEP 1: Data input
..................

Similarly as for the other use cases, specify the model data to use. We will not
use any observations in this example.

::

   model_1 = {
        'fpath': '<path-to-folder-1>',
        'grid type': 'reg', 'grid name': '<grid-name>',
        'start year': <start year>, 'end year': <end year>, 'months': [10,11,12,1,2,3]
	'date interval start': '<yyyy>-<mm>', 'date interval end': '<yyyy>-<mm>',
     	'chunks_time': {'time': -1}, 'chunks_x': {'x': -1}, 'chunks_y': {'y': -1},
        }
   model_2 = {
        'fpath': '<path-to-folder-2>',
        'grid type': 'reg', 'grid name': '<grid-name>',
        'start year': <start year>, 'end year': <end year>, 'months': [10,11,12,1,2,3]
	'date interval start': '<yyyy>-<mm>', 'date interval end': '<yyyy>-<mm>',
     	'chunks_time': {'time': -1}, 'chunks_x': {'x': -1}, 'chunks_y': {'y': -1},
        }



STEP 2: Variables
.................

Set output directory under the **SETTINGS** section.

In the key *variables* we specify the three variables. Precipitation variables
are available with daily resolution while *snw* is available on 3hr resolution.

We skip regional estimates, i.e. set ``regions = ``.

::

    output dir = <path-to-output-directory>

    variables = {
        'pr': {'freq': 'day',
               'units': 'mm',
               'scale factor': 86400,
               'offset factor': None,
               'accumulated': False,
               'obs': None,
               'obs scale factor': None,
               'obs freq': None,
               'var names': None,
               'regrid to': None,
               'regrid method': 'bilinear'},
        'prsn': {'freq': 'day',
                 'units': 'mm',
                 'scale factor': 86400,
                 'offset factor': None,
                 'accumulated': False,
                 'obs': None,
                 'obs scale factor': None,
                 'obs freq': None,
                 'var names': None,
                 'regrid to': None,
                 'regrid method': 'bilinear'},
        'snw': {'freq': '3hr',
                'units': 'mm',
                'scale factor': 86400,
                'offset factor': None,
                'accumulated': False,
                'obs': None,
                'obs scale factor': None,
                'obs freq': None,
                'var names': None,
                'regrid to': None,
                'regrid method': 'bilinear'},
        }

    regions =


STEP 3: Set the *variable modification* option
..............................................
We will test the *variable modification* option here by calculating the rainfall
from total and solid precipitation. It is set as a dictionary with the key as
the name of the new created variable:

::

    variable modification = {
     'prrain': {
     'models': 'all',
     'obs': None,
     'input': {'x': 'pr', 'y': 'prsn'},
     'expression': 'x - y',
     'replace': True},
     }

In the dictionary, you should specify for which 'models' and 'obs' the new
variable will be created. Set specific names, 'all' to do it for all models/obs
or None to not do it for any model/obs.

In the 'input' key you will specify algebraic notations for the variables
included in the creation of the new variable, and in the 'expression' key one
will provide the mathematical expression (using the algebraic notation). Since
we want to get the liquid precipitation we simply take the difference between
total and solid precipitation, and call the new variable *prrain*.
The *replace* switch (boolean) is set to True if the new variable will replace
the variables used to create the new one. If set to False, these will be kept.


STEP 4: Select statistics
.........................

Use the *moment stat* in the **STATISTICS** section. Its default options are found in the
*default_stats_config* function in :ref:`RCAT Statistics
<stats_control_functions>`.

The statistical calculation will be performed for the new *prrain* variable
only using conditional selection based on snow amounts (which will be resampled
to daily mean values). The criteria for rain-on-snow events are here set to
daily rainfall amounts of 5 mm or more occurring over gridpoints with snow on
the ground corresponding to 10 mm of snow water equivalent:

::

 'moment stat': {'vars': ['prrain'], 'moment stat': ['all', 'count'], 'thr': {'prrain': 5},
                'cond analysis': {'prrain': {
                           'cvar': 'snw',
                           'resample resolution': ['D', 'mean'],
                           'type': 'static',
                           'operator': '>=',
                           'value': 10}}
               },

Note that *moment stat* is set to ``['all', 'count']`` which here means
summation of all the true values (where conditions are fulfilled) per gridpoint.

STEP 5: Plotting
................

Under **PLOTTING**, *validation plot* should be set to *True* to enable
plotting.  Plot a map of the results by setting 

::

    validation plot = True

    moments plot config = {'plot type': 'map'}

    map projection = 'LambertConformal'
    map configuration = {
        'central_longitude': 10,
        'central_latitude': 60.6,
        'standard_parallels': (60.6, 60.6),
     }
    map extent = [3, 16, 42, 51]  # Extent of the map; [lon_start, lon_end, lat_start, lat_end]
    map gridlines = False
    map grid config = {'axes_pad': 0.3, 'cbar_mode': 'each', 'cbar_location': 'right',
                  	  'cbar_size': '4%%', 'cbar_pad': 0.05}
    map plot kwargs = {'filled': True, 'mesh': True}
    map model domain =



STEP 6: Configure cluster
.........................

::

    cluster type = slurm
    nodes = 10
    cluster kwargs = {'walltime': '02:00:00'}


STEP 7: Run RCAT
................

To run the analysis run from terminal (see *Run RCAT* in :ref:`configuration`):

     .. code-block:: bash

        python <path-to-RCAT>/src/rcatool/runtime/RCAT_main.py -c config_main.ini


Output statistics and image files will be located under the user-defined output
directory in the *stats* and *imgs* sub-folders respectively
