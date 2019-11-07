## Statistics in RCAT ##

### How do you specify which statistics to do? ###
In the **STATISTICS** section in
[config_main.ini](../scripts/config/config_main.ini) you specify the statistics
to be done, in a Python dictionary structure.

```
stats = {
	'annual cycle': 'default',
	'seasonal cycle': {'time stat': 'mean', 'thr': {'pr': 1.0}},
	'percentile': {'resample resolution': ['day', 'max'], 'pctls': [90, 95, 99, 99.9]},
	'pdf': {'thr': {'pr': 1.0}, 'normalized': True, 'bins': {'pr': (0, 50, 1), 'tas': (265, 290, 5)}}
	'diurnal cycle': {'dcycle stat': 'amount', 'time stat': 'percentile 95'}
	'moments': {'moment stat': {'pr': ['Y', 'max'], 'tas': ['D', 'mean']}},
    }
```

The keys in *stats* are the statistical measures and values provides the
settings/configurations to be applied to the specific statistical calculation.
The statistics that are currently available in RCAT and their default settings
are given in the [statistics.py](../scripts/statistics.py) module. In
particular, the *default_stats_config* function in that module specifies the
statistics possible to calculate along with their properties. Many of the
properties (or settings) are common for each of the measures, for example
*resample resolution*, *thr* or *chunk dimension*, while others may be specific
for the kind of statistics.

If you set *default* as the key value in *stats*, as is the case for *annual
cycle* in the code snippet above, then (obviously) the default settings will be
used. To modify the settings, the key value should be set to a dictionary
containing the particular properties to be modified as keys and values with the
modified item values.

#### Common setting properties ####
* *vars*
* *resample resolution*
* *chunk dimension*
* *pool data*
* *thr*

### How do you add new statistical methods to RCAT? ###
