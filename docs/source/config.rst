.. _configuration:

RCAT Configuration
==================

The main set up is done in the **<path-to-RCAT>/config/config_main.ini** configuration file.
In this file you will set up paths to model data, which variables to analyze
and how (define statistics), which observations to compare with etc. In other
words, this is your starting point when applying RCAT.

#. Setup folder structure
     If you don't want to pollute your cloned git repository we suggest you to
     create a new folder for your analysis and copy the main RCAT configuration 
     file to the new folder.

     .. code-block:: bash

         mkdir -p $HOME/rcat_analysis/test
         cd $HOME/rcat_analysis/test
         cp <path-to-RCAT>/config/config_main.ini .

#. Configure settings in config_main.ini
     A configuration .ini file has a specific structure based
     on sections, properties and values. The RCAT config_main.ini file consists of a handful
     of these sections, for example **MODELS**, under which you specify certain
     properties or values. The latter may in some cases be common structures
     used in python like lists or dictionaries. Below follows a description of
     each of the sections needed to setup the analysis.

     -  MODELS
         Here you specify the path to model data. At the moment a specific
         folder structure is anticipated, with sub-folders under fpath
         according to output frequency; fpath/day, fpath/6H, fpath/15Min, etc.
         Names of these sub-folders are inherited from the freq property set
         under variables in the **SETTINGS** section.

         ::

            model = {
               'fpath': '/path/to/model/data',
               'grid type': 'reg', 'grid name': 'FPS-ALPS3',
               'start year': 1998, 'end year': 2000, 'months': [1,2,3,4,5,6,7,8,9,10,11,12]
            }
         
         Here you also set a couple of grid specifications - namely *grid type*
         and *grid name*. Grid type defines the type of grid for the model
         data; it can be set to either *rot* or *reg*. The former means that
         model data is on rotated grid and the latter that it is on non-rotated
         grid (i.e. regular, rectilinear, curvilinear). If data is on rotated
         grid RCAT will de-rotate the grid. However, it requires that model
         files include coordinate variables in accordance with CF conventions -
         *rlon*/*rlat* for longitudes and latitudes as well as the variable
         *rotated_pole* with attributes *grid_north_pole_longitude* and
         *grid_north_pole_latitude*. 

         Here's another example comparing two models:

         ::

            model_his = {
               'fpath': '/path/to/model_1/data',
               'grid type': 'reg', 'grid name': 'FPS-ALPS3',
               'start year': 1985, 'end year': 2005, 'months': [1,2,3,4,5,6,7,8,9,10,11,12]
            }
            model_scn = {
               'fpath': '/path/to/model_2/data',
               'grid type': 'reg', 'grid name': 'FPS-ALPS3',
               'start year': 2080, 'end year': 2100, 'months': [1,2,3,4,5,6,7,8,9,10,11,12]
            }

         Two different periods is set here because a simulation of historic
         period will be compared with a simulation of future climate. More
         models can be added to the section, but note that the first model (e.g.
         model_his in the above example) will be the reference model. That is,
         if validation plot is True, and no obs data is specified, the
         difference plots will use the first specified model in section as reference data.

     -  OBS
         If observation data is to be used in the analysis, you will need to 
         specify a meta data file by setting the full path to
         *observations_metadata_NN.py* (located under <path-to-RCAT>/config).
         *NN* is any label that signifies the observation meta data for a
         specific location or system (for example a HPC system). If such a
         specific meta data file does not exist, it should be created
         (SAMPLE_observations_metadata.py can be used as a template) and
         modified. **N.B.** Change only the *obs_data* function -- where
         observations are specified.

         In addition, in this section one will specify the time period and
         months for obs data. The same time period will be used for all
         observations.  Which specific observations to include in the analysis
         is not defined here, but in the **SETTINGS** section, in the variables
         property.

     - SETTINGS
         **output dir**: The path for the output (statistics files, plots). If
         you re-run the analysis with the same output directory, you will
         prompted to say whether to overwrite existing output. "overwrite" does
         not mean that existing folder will be completely overwritten (deleted
         and created again). The existing folder structure will be kept intact
         together with output files. However, potentially some output
         (statistics/figure files) with same names will be overwritten.

         **variables**: One of the key settings in the configuration file. The
         value of this property is represented by a dictionary; the keys are
         strings of variable names ('pr', 'tas', ...) and the value of each key
         (variable) is another dictionary consisting of a number of specific
         settings:

         ::

            variables = {
                'psl': {'freq': 'day', 'units': 'hPa', 'scale factor': 0.01, 'accumulated': False, 'obs': ['ERA5', 'EOBS'], 'obs scale factor': 0.01, 'regrid to': 'ERA5', 'regrid method': 'bilinear'},
                'pr': {'freq': 'day', 'units': 'mm', 'scale factor': None, 'accumulated': True, 'obs': 'EOBS', 'obs scale factor': 86400, 'regrid to': 'EOBS', 'regrid method': 'conservative'},
                'tas': {'freq': 'day', 'units': 'K', 'scale factor': None, 'accumulated': False, 'obs': ['ERA5', 'EOBS'], 'obs scale factor': None, 'regrid to': 'ERA5', 'regrid method': 'bilinear'},
            }

         * *freq*: A string of the time resolution of input model data. The
           string should match any of the sub-folders under the path to model
           data, e.g. 'day', '1H', '3H'. In effect, you may choose different
           time resolutions for different variables in the analysis.

         * *units*: The units of the variable data (which will appear in
           figures created in RCAT, and thus should reflect the units after
           data have been manipulated through the analysis).

         * *scale factor*: A numeric factor (integer/float) that model data is
           multiplied with, to convert to desired units (e.g. from J/m2 to
           W/m2) and to ensure that all data (model and observations) have the
           same units. If no scaling is to be done, set value to None. An
           arithmetic expression is not allowed; for example if data is to be
           divided by 10 you cannot define factor as 1/10, it must then be 0.1.
           It is assumed that all model data will use the same factor..

         * *accumulated*: Boolean switch identifying variable data as
           accumulated fields or not. If the former (True), then data will be
           de-accumulated "on the fly" when opening files of data.

         * *obs*: String or list of strings with acronyms of observations to be
           included in the analysis (for the variable of choice, and therefore
           different observations can be chosen for different variables).
           Available observations, and their acronyms, are specified in the
           <path-to-RCAT>/config/observations_metadata_NN.py file. In this
           file you can also add new observational data sets. 

         * *obs scale factor*:As scale factor above but for observations. If
           multiple observations are defined, some of which would need
           different scale factors, a list of factors can be provided. However,
           if the same factor should be used for all observations, it is enough
           to just specify a single factor.

         * *regrid to*: If data is to be remapped to a common grid, you specify
           the name (model name or observation acronym) here. If not, set to
           None.

         * *regrid method*: String defining the interpolation method:
           'conservative' or 'bilinear'.

         **regions**: A list of strings with region names, defining
         geographical areas data will be extracted from. If set, 2D statistical
         fields calculated by RCAT will be cropped over these regions, and in
         line plots produced in RCAT mean statistical values will calculated
         and plotted for each of the regions. If the pool data option in
         statistics configuration (see below) is set to True, then data over
         regions will be pooled together before statistical calculations. If no
         cropping of data is wanted, set this property to None. Read more about
         how to handle regions and polygons in RCAT :ref:`here <polygons_howto>`.

     - STATISTICS
         Another main section of the analysis configuration. Therefore, the
         description of this segment is given separately, see :doc:`RCAT
         Statistics </statistics>`

     - PLOTTING
         This section is intended for the case you want to perform a general
         evaluation/validation of the model. This means that (for the moment) a
         set of standards plots (maps and line plots) can be done by RCAT for a
         set of standard statistical output: annual, seasonal and diurnal
         cycles, pdf's, percentiles and ASoP analysis. If plotting procedures
         for other statistics is wished for, they need to be implemented in the
         RCAT :doc:`plotting module <plots>`.

         **validation plot**: If validation plot is set to True, standard plots
         will be produced for the defined statistics. Otherwise, plotting can
         be done elsewhere using the statistical output files (netcdf format)
         created by RCAT.

         **map configure**: In this property you can change/add key value pairs
         that control for example map projection ('proj') and resolution
         ('res') as well as the dimensions of the map; 'zoom' can be set to
         'crnrs' if corners of model grid is to be used, or 'geom' if you want
         to specify width and height (in meters) of the map. In the latter case
         you need to set 'zoom_geom' [width, height]. Note that these settings
         refers to the reference model in the analysis which is the first model
         data set specified in the **MODELS** section.

         ::

            map configure = {'proj': 'stere', 'res': 'l', 'zoom': 'geom', 'zoom_geom': [1700000, 2100000], 'lon_0': 16.5, 'lat_0': 63}

         For more settings, see the map_setup function in the :doc:`plots module <plots>`.

         **map grid setup**: Settings for the map plot configuration, for
         example whether to use a colorbar or not (cbar_mode) and where to put
         it and the padding between panels. For more info, see the
         *image_grid_setup* function in the :doc:`plots module <plots>`.

         ::

            map grid setup = {'axes_pad': 0.5, 'cbar_mode': 'each', 'cbar_location': 'right', 'cbar_size': '5%%', 'cbar_pad': 0.03}

         **map kwargs**: Additional keyword arguments to be added in the
         matplotlib contour plot call, see the make_map_plot function in
         the :doc:`plotting module <plots>`.

         **line plot settings**: Likewise, settings for line plots can be made,
         e.g. line widths and styles as well as axes configurations. There are
         a number of functions in the :doc:`plotting module <plots>` that
         handles line/scatter/box plots, see for example the fig_grid_setup and
         make_line_plot functions.

         ::

            line grid setup = {'axes_pad': (11., 6.)}
            line kwargs = {'lw': 2.5}

     - CLUSTER
        The last section control the cluster type. You can choose between local
        pc and SLURM at the moment.

        **cluster type**: choose "local" for running on you local pc and
        "slurm" if you want to run RCAT on a HPC with a SLURM job scheduler and
        read information below. For local pc no other settings need to be made
        in this section.

        *SLURM*
            RCAT uses `Dask <https://docs.dask.org/>`_ to perform file managing
            and statistical analysis in an efficient way through parallelization.
            When applying Dask on queuing systems like PBS or Slurm,
            `Dask-Jobqueue <https://dask-jobqueue.readthedocs.io>`_ provides an
            excellent interface for handling such work flow. It is used in RCAT
            and to properly use Dask and Dask-Jobqueue on an HPC system you need
            to provide some information about that system and how you plan to use
            it. By default, when Dask-Jobqueue is first imported a configuration
            file is placed in ~/.config/dask/jobqueue.yaml. What is set in this
            file are the default settings being used. On Bi/NSC we have set up a
            default configuration file as below.

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
                   local-directory: $SNIC_TMP

                   # SLURM resource manager options
                   queue: null
                   project: null
                   walltime: '01:00:00'
                   job-extra: ['--exclusive']

            When default settings have been set up, the main properties that you
            usually want to change in the **CLUSTER** section are the number of nodes
            to use and wall time:

            ::

               nodes = 15
               slurm kwargs = {'walltime': '02:00:00', 'memory': '256GB', 'job_extra': ['-C fat']}

            **nodes**: Sometimes you might need more memory on the nodes, and on
            Bi/NSC there are fat nodes available. If you want to use fat nodes,
            you can specify this through

            ::

               slurm kwargs = {'walltime': '02:00:00', 'memory': '256GB', 'job_extra': ['-C fat']}

#. Run RCAT
     When you have done your configuration and saved config_main.ini you can
     start the analysis step. The main program is located in the *rcat* directory
     and called RCAT_main.py. See point 1: :ref:`Setup folder structure
     <configuration>` and run main RCAT_main.py from your analysis folder.


     .. code-block:: bash

        python <path-to-RCAT>/rcat/RCAT_main.py -c config_main.ini

    .. note::

        Don't forget to set $PYTHONPATH to your RCAT
        directory (<path-to-RCAT>).
