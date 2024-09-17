.. _statistics:

RCAT Statistics
===============

In the **STATISTICS** section in the file config_main.ini you specify which
statistics to calculate, in a dictionary structure.

::

   stats = {
     'annual cycle': 'default',
     'seasonal cycle': {'stat method': 'mean', 'thr': {'pr': 1.0}},
     'percentile': {'resample resolution': ['day', 'max'], 'pctls': [90, 95, 99, 99.9]},
     'pdf': {'thr': {'pr': 1.0}, 'normalized': True, 'bins': {'pr': (0, 50, 1), 'tas': (265, 290, 5)}}
     'diurnal cycle': {'dcycle stat': 'amount', 'stat method': 'percentile 95'}
     'moments': {'moment stat': {'pr': ['Y', 'max'], 'tas': ['D', 'mean']}},
   }

The keys in stats are the statistical measures and values provides the
settings/configurations to be applied to the specific statistical calculation.
The statistics that are currently available in RCAT and their default settings
are given in the :ref:`RCAT Statistics <stats_control_functions>` module. Some
related information is also found in the API-reference :ref:`Statistics
<statistics_functions>`. In particular, the *default_stats_config* function in
that module specifies the statistics possible to calculate along with their
properties. Many of the properties (or settings) are common for each of the
measures, for example *resample resolution*, *thr* (threshold) or *chunk
dimension*, while others may be specific for the kind of statistics.

If you set *default* as the key value in stats, as is the case for *annual cycle*
in the code snippet above, then (obviously) the default settings will be used.
To modify the settings, create a dictionary containing
the particular properties to be modified as keys and values with the modified
item values.

#. Common settings and properties
    Here we list the most common settings and give some information of them.

    **vars**: String *variable_name* or list of strings *[variable_name_1,
    variable_name_2, ...]* representing which variables (as defined in
    variables under **SETTINGS** in the main configuration file). If set to empty
    list [] the calculation will be performed for all defined variables. Some
    statistics are specifically for a certain variable and then this variable
    is set as default.

    **resample resolution**: If you want to resample input data (model or obs)
    to another time resolution before statistical computation, you can set this
    property to a list with two elements; the first defines the temporal
    resolution (e.g. 3 hours, day, month, etc) and the second is the arithmetic
    method to be used during the resampling. The `xarray
    <http://xarray.pydata.org>`_ resample function is applied here which builds
    on the similar function in the `pandas <https://pandas.pydata.org/>`_
    package. For example, resampling to 6 hourly data, taking the sum over
    intervening time steps would be defined as follows in the configuration
    file:

    ::

        'resample resolution': ['6H', 'sum']

    The documentation of and available options for the resampling function can
    be found `here (xarray)
    <http://xarray.pydata.org/en/stable/time-series.html#resampling-and-grouped-operations>`_
    and `here (pandas)
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html>`_
    (see DateOffset Objects section for frequency string options).

    You may also define the resample option as a dictionary with the keys
    representing the variables to be resampled and the values the corresponding
    resampling (two element list) to use for each of these variables:

    ::

        'resample resolution': {'pr': ['M', 'sum'], 'tas': ['Y', 'max']}

    Resampling will not be performed for variables not listed in the dictionary.

    There are two additional options two *resample resolution*: i) 'select
    hours' and 'select dates'. In option i) you may select specific hours of the
    day, e.g. only 12Z time steps or a list of hours:

    ::

        'resample resolution': ['select hours', 12]
        'resample resolution': ['select hours', [00, 06, 12, 18]]
     
    'select dates' can be a specific date 'YYYY-MM-HH' or a list of dates that
    one can specify and extract from the input data. 


    **cond analysis**: With this option one may perform statistics on a
    sub-selection of the input data, a sub-selection that is achieved through a
    conditional sampling. The latter is defined by choosing i) a metod; either 'static',
    'percentile' or 'file'; ii) an operator: *>*, *<*, *<=*, *>=* and *=*; and
    iii) a value, only used for 'static' and 'percentile', where the value
    corresponds to the threshold to use in the first case and the percentile value in
    the second. For the third option, 'file', a path to the file must be
    provided, along with the variable (its name) in the file to be used. So, the
    field of this file (it must be 2D and of the same spatial resolution as data to
    be sub-selected) will be applied as 2D array of thresholds for the
    conditional selection.

    If this option should be used, the settings should be provided in a nested
    dictionary with the first key being the variable which should be
    sub-selected, and the value another dictionary constisting of the selected
    settings metnioned above, including which variable to use as conditional
    data (unless using the 'file' option). This data can also be resampled by
    including the *resample resolution* option in the dictionary. The example
    below shows the case of sub-selecting precipitation ('pr') data conditioned
    on values of CAPE (resampled to daily maximum) being larger than a threshold
    defined by the 95th percentile:

    ::

       'cond analysis': {'pr': {
            'cvar': 'cape',
            'resample resolution': ['D', 'max'],
            'type': 'percentile',
            'operator': '>',
            'value': 95}
            }


    **chunk dimension**: An important feature behind Dask parallelization is
    the use of *blocked algorithms*; to divide the data into chunks and then
    run computations on each block of data in parallel (see `Dask
    <https://docs.dask.org/>`_ documentation for more information). When
    working with multiple dimension data arrays in Dask you can chunk the data
    along different dimensions, depending on what kind of calculations you may
    want to do. For example, when computing seasonal means of data with
    dimensions (time, lat, lon) it doesn't matter along which dimension
    the data is chunked along. However, if you want to calculate percentiles based on time series
    in each grid point then chunking should be done along the spatial dimensions.
    The chunk dimension property has two options; time/space.
    For example to chunk along time, set

    ::

        'chunk dimension': 'time'

    **pool data**: **Not yet implemented!**
    Boolean switch (True/False). Set to True if statistics should
    be done on pooled data, i.e. assembling all the grid points and time steps
    and then perform calculations. If you want the pooling to be done over
    certain sub-regions, then you need to specify these in the regions property
    in the main configuration file, *<path-to-RCAT>/config/config_main.ini*.


    **thr**: Thresholding of data. The value of this (None is the default)
    should be a dictionary with keys defining variables and values an integer
    of float; e.g.

    ::

        'thr': {'pr': 0.1, 'tas': 273}

#. Specific settings and properties
    Here we list and describe more specific settings. For some statistical
    calculations, more information on the required settings and their
    descriptions can be found in in the API-reference :ref:`Statistics
    <statistics_functions>`

    **stat method**: In many of the available statistical calculations,
    computations can be done using various methods or moments (e.g. mean, sum,
    std, etc). For example, if calculating the diurnal cycle, one could compute
    the mean of all values for each time unit in the cycle or another measure
    such as a percentile value. This can be specified with this property.
    Default value is mean. To use a percentile, set (for 95th percentile);

    ::

       'stat method': 'percentile 95'

    **dcycle stat**: In the computation of the diurnal cycle (including
    harmonic fit) the *dcycle stat* defines whether to compute magnitudes or
    frequency of occurrences. For the former set it to 'amount', for the latter
    to 'frequency'. When calculating frequencies you must also set the 'thr'
    option, so for each unit of time in the cycle the occurrence above this
    threshold is calculated.

    **hours** (in diurnal cycle): The value of this property is a list of hours
    that should be used in the diurnal cycle computation. It might be changed
    if you want to compare data sets with different temporal resolution (this
    can also be achieved with the *resample resolution* option).

    **moment stat**: The moment statistical calculation involve a basic
    calculation on the data, such as means, sums or standard deviations. It is
    basically the same as the resample resolution property and the *moment
    stat* is set the same way. For example, if you want to calculate the annual
    maximum of the input data set.

    ::

       'moment stat': ['Y', 'max']

    If the first element in the list is set to 'all', the statistical
    measure (second element of list) will be computed over all the values in the
    temporal dimension -- essentially producing a 2D output field.

    **normalized** (in pdf): Boolean switch. With normalization, the normalized
    contribution (by the total mean) from each bin interval in the pdf (or
    frequency intensity distribution) is computed.

    **normalized** (in Rxx): In the Rxx function (see
    :ref:`statistics <statistics_functions>` module) the counts above the
    threshold is normalized by the total number of values if this property is
    set to True.

#. How do you add new statistical methods to RCAT?  The code in RCAT is heavily
   based on `xarray <http://xarray.pydata.org/>`_ as well as `dask
   <https://docs.dask.org/>`_. Xarray has been interfaced closely with dask
   applications so many of the operations that can be done in xarray, like many
   (basic) statistical calculations, are already dask compliant and therefore
   relatively easy to implement in RCAT. If you would like to include any new
   such feature, have a look in the :ref:`RCAT Statistics
   <stats_calc_functions>` module, for example how the implementation of
   'seasonal cycle' has been done.

   For more elaborate statistics, using for example functions created by the
   user (using standard numpy/python code), it may be a bit more complex.
   Xarray has a function called `apply_ufunc
   <http://xarray.pydata.org/en/stable/generated/xarray.apply_ufunc.html#xarray.apply_ufunc>`_
   which allows repeatedly applying a user function to xarray objects
   containing Dask arrays in an automatic way. See `here
   <http://xarray.pydata.org/en/stable/computation.html#comput-wrapping-custom>`_
   for_some more information.
