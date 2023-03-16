import sys
import numpy as np
import dask.array as da
import xarray as xa
from rcat.stats import ASoP
from rcat.stats import convolve
from rcat.stats import event_duration as eda
from rcat.stats import climateindex as ci
from pandas import to_timedelta
from copy import deepcopy


############################################################
#                                                          #
#     FUNCTIONS CONTROLLING STATISTICAL CALCULATIONS       #
#                                                          #
############################################################

def default_stats_config(stats):
    """
    The function returns a dictionary with default statistics configurations
    for a selection of statistics given by input stats.
    """
    stats_dict = {
        'generic': {
            'vars': [],
            'statistic': {'description': 'Average', 'algorithm': 'mean'},
            'function args': None,
            'apply dim': 'time',
            'new dims': None,
            'group data': None,
            'resample resolution': None,
            'pool data': False,
            'thr': None,
            'cond analysis': None,
            'chunk dimension': 'time'},
        'moments': {
            'vars': [],
            'moment stat': ['D', 'mean'],
            'moment resample kwargs': None,
            'resample resolution': None,
            'pool data': False,
            'thr': None,
            'cond analysis': None,
            'chunk dimension': 'time'},
        'seasonal cycle': {
            'vars': [],
            'resample resolution': None,
            'pool data': False,
            'stat method': 'mean',
            'thr': None,
            'cond analysis': None,
            'chunk dimension': 'time'},
        'annual cycle': {
            'vars': [],
            'resample resolution': None,
            'pool data': False,
            'stat method': 'mean',
            'thr': None,
            'cond analysis': None,
            'chunk dimension': 'time'},
        'diurnal cycle': {
            'vars': [],
            'resample resolution': None,
            'hours': None,
            'dcycle stat': 'amount',
            'stat method': 'mean',
            'method kwargs': None,
            'thr': None,
            'cond analysis': None,
            'pool data': False,
            'chunk dimension': 'space'},
        'dcycle harmonic': {
            'vars': [],
            'resample resolution': None,
            'pool data': False,
            'dcycle stat': 'amount',
            'thr': None,
            'cond analysis': None,
            'chunk dimension': 'space'},
        'asop': {
            'vars': ['pr'],
            'resample resolution': None,
            'pool data': False,
            'nr_bins': 80,
            'bin_type': 'Klingaman',
            'thr': None,
            'cond analysis': None,
            'chunk dimension': 'space'},
        'eda': {
            'vars': ['pr'],
            'resample resolution': None,
            'pool data': False,
            'duration bins': np.arange(1, 51),
            'event statistic': 'amount',
            'statistic bins': [.1, .2, .5, 1, 2, 5, 10, 20, 50, 100, 150, 200],
            'dry events': False,
            'dry bins': None,
            'event thr': 0.1,
            'cond analysis': None,
            'chunk dimension': 'space'},
        'pdf': {
            'vars': [],
            'resample resolution': None,
            'pool data': False,
            'bins': None,
            'normalized': False,
            'thr': None,
            'cond analysis': None,
            'dry event thr': None,
            'chunk dimension': 'space'},
        'percentile': {
            'vars': [],
            'resample resolution': None,
            'pool data': False,
            'pctls': [95, 99],
            'thr': None,
            'cond analysis': None,
            'chunk dimension': 'space'},
        'Rxx': {
            'vars': ['pr'],
            'resample resolution': None,
            'pool data': False,
            'normalize': False,
            'thr': 1.0,
            'cond analysis': None,
            'chunk dimension': 'space'},
        'cdd': {
            'vars': ['pr'],
            'resample resolution': None,
            'pool data': False,
            'thr': 1.0,
            'periods': np.arange(1, 61),
            'maxper': False,
            'cond analysis': None,
            'chunk dimension': 'space'},
        'signal filtering': {
            'vars': [],
            'resample resolution': None,
            'pool data': False,
            'filter': 'lanczos',
            'cutoff type': 'lowpass',
            'window': 61,
            'mode': 'same',
            '1st cutoff': None,
            '2nd cutoff': None,
            'filter dim': 1,
            'thr': None,
            'cond analysis': None,
            'chunk dimension': 'space'},
            }

    return {k: stats_dict[k] for k in stats}


def mod_stats_config(requested_stats):
    """
    Get the configuration for the input statistics 'requested_stats'.
    The returned configuration is a dictionary.
    """
    stats_dd = default_stats_config(list(requested_stats.keys()))

    # Update dictionary based on input
    for k in requested_stats:
        if requested_stats[k] == 'default':
            pass
        else:
            for m in requested_stats[k]:
                msg = "For statistic {}, the configuration key {} is not "\
                        "available. Check possible configurations  in "\
                        "default_stats_config in stats_template "\
                        "module.".format(k, m)
                try:
                    stats_dd[k][m] = requested_stats[k][m]
                except KeyError:
                    print(msg)

    return stats_dd


def _stats(stat):
    """
    Dictionary that relates a statistical measure to a specific function that
    do the calculation.
    """
    p = {
        'generic': generic_function,
        'moments': moments,
        'seasonal cycle': seasonal_cycle,
        'annual cycle': annual_cycle,
        'percentile': percentile,
        'diurnal cycle': diurnal_cycle,
        'dcycle harmonic': dcycle_harmonic_fit,
        'pdf': freq_int_dist,
        'asop': asop,
        'eda': eda_calc,
        'Rxx': Rxx,
        'cdd': cdd,
        'signal filtering': filtering,
    }
    return p[stat]


def calc_statistics(data, var, stat, stat_config):
    """
    Calculate statistics 'stat' according to configuration in 'stat_config'.
    This function calls the respective stat function (defined in _stats).
    """

    stat_data = _stats(stat)(data, var, stat, stat_config)
    return stat_data


def _check_hours(ds):
    """
    Checking time stamps for subdaily data
    Hours in time stamps should be the right edge of interval if accumulated
    or resampled data.
    """
    # Round to nearest minute
    ds['time'] = ds.indexes['time'].round('min')

    # Hour offsets
    offset_dict = {12: 6, 6: 3, 3: 1, 1: 0}
    _delta = ds['time.hour'].diff('time')
    delta = _delta[_delta > 0].values[0]
    hr_check = np.all(np.isin(ds['time.hour'], np.arange(0, 24, delta)))
    if hr_check and not np.any(ds.time.dt.minute > 0):
        pass
    else:
        offset = offset_dict[delta]
        print("\t\t\tShifting time stamps to whole hours!\n")
        hbool = np.all(ds.indexes['time'].minute >= 30)
        ds['time'] = ds.indexes['time'].ceil('H').shift(offset, 'H') if hbool\
            else ds.indexes['time'].floor('H').shift(offset, 'H')
    return ds


def _get_freq(tf):
    from functools import reduce

    d = [j.isdigit() for j in tf]
    if np.any(d):
        freq = int(reduce((lambda x, y: x+y), [x for x, y in zip(tf, d) if y]))
    else:
        freq = 1

    unit = reduce((lambda x, y: x+y), [x for x, y in zip(tf, d) if not y])

    if unit in ('M', 'Y'):
        freq = freq*30 if unit == 'M' else freq*365
        unit = 'D'
    elif unit[0] == 'Q':
        freq = 90
        unit = 'D'

    return freq, unit


############################################################
#                                                          #
#                   STATISTICS FUNCTIONS                   #
#                                                          #
############################################################

def generic_function(data, var, stat, stat_config):
    """
    Calculate statistics applying user defined function. Arguments and returned
    data need to be specified in main RCAT configuration file. Statistical
    calculations may be done on sub-grouped data (using xarray resample),
    as defined by user in configuration file.
    """

    def _calc(da, modpath, module, funcnm, **kwargs):
        sys.path.append(modpath)
        import importlib
        fmod = importlib.import_module(module)
        func = getattr(fmod, funcnm)

        out = func(da.values, **kwargs)
        da_out = xa.DataArray(out, dims=da.dims, coords=da.coords)
        return da_out

    def _dd_grp(g):
        if g in ('Y', 'y', 'year', 'ann', 'annual'):
            lbl = 'Y'
        elif g in ('D', 'd', 'day'):
            lbl = 'D'
        elif g in ('season', 'QS-DEC'):
            lbl = 'QS-DEC'
        elif g in ('hour', 'h', 'H'):
            lbl = 'H'
        else:
            lbl = g

        return lbl

    # Data thresholding
    in_thr = stat_config[stat]['thr']
    if in_thr is not None:
        if var in in_thr:
            thr = in_thr[var]
            data = data.where(data[var] >= thr)
        else:
            thr = None
    else:
        thr = in_thr

    dim = stat_config[stat]['apply dim']
    outdim = stat_config[stat]['new dims']

    # Group/sub-select data?
    grp = stat_config[stat]['group data']
    if grp is not None:
        data = data.chunk({'time': -1})
        # NOTE: Use groupby or resample function?
        # indata = data[var].groupby(f'time.{grp}').apply(lambda x: x)
        gr = _dd_grp(grp)
        indata = data[var].resample(time=f'{gr}').apply(lambda x: x)
    else:
        gr = None
        indata = data[var]

    # Stat function to apply
    fstat = stat_config[stat]['statistic']
    kwargs = stat_config[stat]['function args']
    assert isinstance(fstat, dict), \
        print("stat function in configuration file must be a dictionary"
              " with keys 'description' and 'algorithm'")

    if fstat['algorithm'] in ('sum', 'count', 'mean', 'max', 'min', 'std'):
        st_data = eval(f"indata.{fstat['algorithm']}('time')")
    else:
        if 'lambda' in fstat['algorithm']:
            func = fstat['algorithm']
        elif isinstance(fstat['algorithm'], dict):
            import pathlib
            fpth = fstat['algorithm']['module file'].rsplit('/', 1)
            f = pathlib.Path(fpth[1])
            fmodule = f.with_suffix('').name
            funcname = fstat['algorithm']['function name']

        newdim = [[outdim['dim name']]] if outdim is not None else outdim
        dimsize = outdim['dim size'] if outdim is not None else outdim
        if gr is None:
            if newdim is not None:
                out = xa.apply_ufunc(
                    func, indata, input_core_dims=[[dim]],
                    output_core_dims=newdim, dask='parallelized',
                    output_sizes={newdim: dimsize}, output_dtypes=[float],
                    kwargs=kwargs)
            else:
                out = xa.apply_ufunc(func, indata, input_core_dims=[[dim]],
                                     dask='parallelized',
                                     output_dtypes=[float], kwargs=kwargs)
            dims = list(out.dims)
            st_data = out.to_dataset().transpose(dims[-1], dims[0], dims[1])
        else:
            out = xa.map_blocks(_calc, indata, (fpth[0], fmodule, funcname),
                                kwargs, template=indata)
            st_data = out.to_dataset(name=var)
            # Add back time stamps
            st_data = st_data.assign(
                {'time': data['time'].resample(time=f'{gr}').apply(
                    lambda x: x).values})

    st_data.attrs['Description'] =\
        (f"Statistic: {fstat['description']} | Threshold: {thr}  | "
         f"Grouped data: {gr}")

    return st_data


def moments(data, var, stat, stat_config):
    """
    Calculate standard moment statistics: avg, median, std, max/min
    """

    # Data thresholding
    in_thr = stat_config[stat]['thr']
    if in_thr is not None:
        if var in in_thr:
            thr = in_thr[var]
            data = data.where(data[var] >= thr)
        else:
            thr = None
    else:
        thr = in_thr

    # Moment stats configuration
    _mstat = deepcopy(stat_config[stat]['moment stat'])
    mstat = _mstat[var] if isinstance(_mstat, dict) else _mstat

    if mstat is None:
        st_data = data.copy()
        st_data.attrs['Description'] =\
            f"Moment statistic: No statistics applied | Threshold: {thr}"
    else:
        if mstat[0] == 'all':
            st_data = eval(f"data.{mstat[1]}(dim='time', skipna=True)")
        else:
            # Resample expression
            res_kw = stat_config[stat]['moment resample kwargs']
            if res_kw is None:
                expr = (f"data[var].resample(time='{mstat[0]}')"
                        f".{mstat[1]}('time').dropna('time', 'all')")
            else:
                expr = (f"data[var].resample(time='{mstat[0]}', **res_kw)"
                        f".{mstat[1]}('time').dropna('time', 'all')")
            diff = data.time.values[1] - data.time.values[0]
            nsec = to_timedelta(diff).total_seconds()
            tr, fr = _get_freq(mstat[0])
            sec_resample = to_timedelta(tr, fr).total_seconds()
            if nsec >= sec_resample or mstat is None:
                print("\t\t* Moment statistics:\n\t\tData already at the same "
                      "or coarser time resolution as selected resample "
                      "frequency!\n\t\tKeeping data as is ...\n")
                st_data = data.copy()
            else:
                _st_data = eval(expr)
                st_data = _st_data.to_dataset()

        st_data.attrs['Description'] =\
            "Moment statistic: {} | Threshold: {}".format(
                ' '.join(s.upper() for s in mstat), thr)
    return st_data


def seasonal_cycle(data, var, stat, stat_config):
    """
    Calculate seasonal cycle
    """
    tstat = stat_config[stat]['stat method']
    in_thr = stat_config[stat]['thr']
    if in_thr is not None:
        if var in in_thr:
            thr = in_thr[var]
            data = data.where(data[var] >= thr)
        else:
            thr = None
    else:
        thr = in_thr
    if 'percentile' in tstat:
        q = tstat.partition(' ')[2]
        errmsg = ("Make sure percentile(s) in stat method is given correctly; "
                  "i.e. with a white space e.g. 'percentile 95'")
        if not q:
            raise ValueError(errmsg)
        else:
            q = float(q)
        sc_pctls = xa.apply_ufunc(
            _percentile_func, data[var].groupby('time.season'),
            input_core_dims=[['time']],
            dask='parallelized', output_dtypes=[float],
            kwargs={'q': q, 'axis': -1, 'thr': thr})
        st_data = sc_pctls.to_dataset()
    else:
        st_data = eval("data.groupby('time.season').{}('time')".format(
            tstat))
    st_data = st_data.reindex(season=['DJF', 'MAM', 'JJA', 'SON'])
    st_data.attrs['Description'] =\
        "Seasonal cycle | Season stat: {} | Threshold: {}".format(
                tstat, thr)
    return st_data


def annual_cycle(data, var, stat, stat_config):
    """
    Calculate annual cycle
    """
    tstat = stat_config[stat]['stat method']
    in_thr = stat_config[stat]['thr']
    if in_thr is not None:
        if var in in_thr:
            thr = in_thr[var]
            data = data.where(data[var] >= thr)
        else:
            thr = None
    else:
        thr = in_thr
    if 'percentile' in tstat:
        q = tstat.partition(' ')[2]
        errmsg = ("Make sure percentile(s) in stat method is given correctly; "
                  "i.e. with a white space e.g. 'percentile 95'")
        if not q:
            raise ValueError(errmsg)
        else:
            q = float(q)
        ac_pctls = xa.apply_ufunc(
            _percentile_func, data[var].groupby('time.month'),
            input_core_dims=[['time']],
            dask='parallelized', output_dtypes=[float],
            kwargs={'q': q, 'axis': -1, 'thr': thr})
        st_data = ac_pctls.to_dataset()
    elif isinstance(tstat, dict):
        func = tstat['function']
        st_data = eval("data.groupby('time.month').reduce({}, dim='time')".\
                       format(func))
    else:
        st_data = eval("data.groupby('time.month').{}('time')".format(
            tstat))
    st_data.attrs['Description'] =\
        "Annual cycle | Month stat: {} | Threshold: {}".format(
                tstat, thr)
    st_data = st_data.chunk({'month': -1})

    return st_data


def diurnal_cycle(data, var, stat, stat_config):
    """
    Calculate diurnal cycle
    """
    # Type of diurnal cycle; amount or frequency
    dcycle_stat = stat_config[stat]['dcycle stat']

    # Threshold; must be defined for frequency
    in_thr = stat_config[stat]['thr']
    if in_thr is not None:
        if var in in_thr:
            thr = in_thr[var]
            data = data.where(data[var] >= thr)
        else:
            thr = None
    else:
        thr = in_thr

    # Check time stamps in data
    data = _check_hours(data)

    if dcycle_stat == 'amount':
        tstat = stat_config[stat]['stat method']
        if 'percentile' in tstat:
            q = tstat.partition(' ')[2]
            errmsg = ("Make sure percentile(s) in stat method is given "
                      "correctly; i.e. with a white space e.g. "
                      "'percentile 95'")
            if not q:
                raise ValueError(errmsg)
            else:
                q = float(q)
            dc_pctls = xa.apply_ufunc(
                _percentile_func, data[var].groupby('time.hour'),
                input_core_dims=[['time']],
                dask='parallelized', output_dtypes=[float],
                kwargs={'q': q, 'axis': -1, 'thr': thr})
            dcycle = dc_pctls.to_dataset()
        elif 'pdf' in tstat:
            # Bins
            assert 'bins' in stat_config[stat]['method kwargs'],\
                    "\n\tBins are missing in 'method kwargs'!\n"
            bin_r = stat_config[stat]['method kwargs']['bins']
            bins = np.arange(bin_r[0], bin_r[1], bin_r[2])
            lbins = bins.size - 1
            dc_pdf = xa.apply_ufunc(
                _pdf_calc, data[var].groupby('time.hour'),
                input_core_dims=[['time']], output_core_dims=[['bins']],
                dask='parallelized', output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'bins': lbins+1}},
                kwargs={
                    'keepdims': True, 'bins': bins, 'axis': -1, 'thr': thr})
            dims = list(dc_pdf.dims)
            dcycle = dc_pdf.to_dataset().assign_coords(
                {'bins': bins}).transpose('bins', 'hour', dims[0], dims[1])
        else:
            dcycle = eval("data.groupby('time.hour').{}('time')".format(tstat))

        statnm = "Amount | stat: {} | thr: {}".format(tstat, thr)

    elif dcycle_stat == 'frequency':
        errmsg = "For frequency analysis, a threshold ('thr') must be set!"
        assert thr is not None, errmsg

        dcycle = data.groupby('time.hour').count('time')
        totdays = np.array([(data['time.hour'].values == h).sum()
                            for h in np.arange(24)])
        statnm = "Frequency | stat: counts | thr: {}".format(thr)
    else:
        print("Unknown configured diurnal cycle stat: {}".format(dcycle_stat))
        sys.exit()

    dcycle = dcycle.chunk({'hour': -1})
    _hrs = stat_config[stat]['hours']
    hrs = _hrs if _hrs is not None else dcycle.hour
    st_data = dcycle.sel(hour=hrs)
    if dcycle_stat == 'frequency':
        st_data = st_data.assign({'ndays_per_hour': ('nday', totdays)})
    st_data.attrs['Description'] =\
        "Diurnal cycle | {}".format(statnm)
    return st_data


def dcycle_harmonic_fit(data, var, stat, stat_config):
    """
    Calculate diurnal cycle with Harmonic oscillation fit
    """
    # Type of diurnal cycle; amount or frequency
    dcycle_stat = stat_config[stat]['dcycle stat']

    # Threshold; must be defined for frequency
    in_thr = stat_config[stat]['thr']
    if in_thr is not None:
        if var in in_thr:
            thr = in_thr[var]
            data = data.where(data[var] >= thr)
        else:
            thr = None
    else:
        thr = in_thr

    if dcycle_stat == 'amount':
        data = _check_hours(data)
        dcycle = data.groupby('time.hour').mean('time')
        statnm = "Amount | thr: {}".format(thr)
    elif dcycle_stat == 'frequency':
        ermsg = "For frequency analysis, a threshold must be set"
        assert thr is not None, ermsg

        data_sub = data.where(data[var] >= thr)
        data_sub = _check_hours(data_sub)
        dcycle = data_sub.groupby('time.hour').count('time')
        totdays = np.array([(data_sub['time.hour'].values == h).sum()
                            for h in np.arange(24)])
        statnm = "Frequency | thr: {}".format(thr)
    else:
        print("Unknown configured diurnal cycle stat: {}".format(dcycle_stat))
        sys.exit()
    dcycle = dcycle.chunk({'hour': -1})

    dc_fit = xa.apply_ufunc(
        _harmonic_linefit, dcycle[var], input_core_dims=[['hour']],
        output_core_dims=[['fit']], dask='parallelized',
        output_dtypes=[float], output_sizes={'fit': 204},
        kwargs={'keepdims': True, 'axis': -1, 'var': var})
    dims = list(dc_fit.dims)
    st_data = dc_fit.to_dataset().transpose(dims[-1], dims[0], dims[1])
    if dcycle_stat == 'frequency':
        st_data = st_data.assign({'ndays_per_hour': ('nday', totdays)})
    st_data.attrs['Description'] =\
        "Harmonic fit of diurnal cycle | Statistic: {}".format(statnm)
    st_data.attrs['Data info'] = (
        """First four values in each array with fitted data """
        """are fit parameters; (c1, p1, c2, p2), where 1/c2 """
        """and p1/p2 represents amplitude and phase of 1st/2nd """
        """harmonic of the fit.""")
    return st_data


def percentile(data, var, stat, stat_config):
    """
    Calculate percentiles
    """
    in_thr = stat_config[stat]['thr']
    if in_thr is not None:
        thr = None if var not in in_thr else in_thr[var]
    else:
        thr = in_thr
    pctls = stat_config[stat]['pctls']
    lpctls = [pctls] if not isinstance(pctls, (list, tuple)) else pctls
    pctl_c = xa.apply_ufunc(
        _percentile_func, data[var], input_core_dims=[['time']],
        output_core_dims=[['pctls']], dask='parallelized',
        output_sizes={'pctls': len(lpctls)}, output_dtypes=[float],
        kwargs={'q': lpctls, 'axis': -1, 'thr': thr})
    dims = list(pctl_c.dims)
    pctl_ds = pctl_c.to_dataset().transpose(dims[-1], dims[0], dims[1])
    st_data = pctl_ds.assign({'percentiles': ('pctls', lpctls)})
    st_data.attrs['Description'] =\
        "Percentile | q: {} | threshold: {}".format(lpctls, thr)
    return st_data


def freq_int_dist(data, var, stat, stat_config):
    """
    Calculate frequency intensity distributions
    """
    # Bins
    if var not in stat_config[stat]['bins']:
        dmn = data[var].min(skipna=True)
        dmx = data[var].max(skipna=True)
        bins = np.linspace(dmn, dmx, 20)
    else:
        bin_r = stat_config[stat]['bins'][var]
        bins = np.arange(bin_r[0], bin_r[1], bin_r[2])

    # Data threshold
    in_thr = stat_config[stat]['thr']
    if in_thr is not None:
        thr = None if var not in in_thr else in_thr[var]
    else:
        thr = in_thr

    # Dry event threshold
    in_dry_thr = stat_config[stat]['dry event thr']
    if in_dry_thr is not None:
        dry_thr = None if var not in in_dry_thr else in_dry_thr[var]
    else:
        dry_thr = in_dry_thr

    # Output size
    lbins_out = bins.size - 1 if dry_thr is None else bins.size

    # Normalization
    normalized = stat_config[stat]['normalized']
    if isinstance(normalized, bool):
        norm = normalized
    else:
        norm = False if var not in normalized else normalized[var]

    pdf = xa.apply_ufunc(
        _pdf_calc, data[var], input_core_dims=[['time']],
        output_core_dims=[['bins']], dask='parallelized',
        output_dtypes=[float], output_sizes={'bins': lbins_out},
        kwargs={'keepdims': True, 'bins': bins, 'axis': -1, 'norm': norm,
                'thr': thr, 'dry_event_thr': dry_thr})
    dims = list(pdf.dims)
    pdf_ds = pdf.to_dataset().transpose(dims[-1], dims[0], dims[1])
    st_data = pdf_ds.assign(bin_edges=['dry_events']+list(bins))
    st_data.attrs['Description'] =\
        "PDF | threshold: {} | Normalized bin data: {}".format(thr, norm)
    return st_data


def asop(data, var, stat, stat_config):
    """
    Calculate ASoP components for precipitation
    """
    if stat_config[stat]['nr_bins'] is None:
        nbins = np.arange(50)
    else:
        nbins = np.arange(stat_config[stat]['nr_bins'])
    bintype = stat_config[stat]['bin_type']

    # Define bins
    bins = ASoP.bins_calc(nbins, bintype)
    bins = np.insert(bins, 0, 0.0)
    lbins = bins.size - 1

    in_thr = stat_config[stat]['thr']
    if in_thr is not None:
        thr = None if var not in in_thr else in_thr[var]
    else:
        thr = in_thr
    asop_out = xa.apply_ufunc(
        ASoP.asop, data[var], input_core_dims=[['time']],
        output_core_dims=[['factors', 'bins']], dask='parallelized',
        output_dtypes=[float], output_sizes={'factors': 2, 'bins': lbins},
        kwargs={'keepdims': True, 'axis': -1, 'bins': bins})
    dims = list(asop_out.dims)
    # N.B. This does not work in rcat yet! Variable name need still to be 'pr'
    # C = asop.isel(factors=0)
    # FC = asop.isel(factors=1)
    # dims = list(C.dims)
    # C_ds = C.to_dataset().transpose(dims[-1], dims[0], dims[1])
    # FC_ds = FC.to_dataset().transpose(dims[-1], dims[0], dims[1])
    # asop_ds = = xa.Dataset.merge(C_ds, FC_ds)
    asop_ds = asop_out.to_dataset().transpose(dims[-2], dims[-1],
                                              dims[0], dims[1])
    st_data = asop_ds.assign(bin_edges=bins, factors=['C', 'FC'])
    st_data.attrs['Description'] =\
        f"ASoP analysis | bin type: {bintype} | threshold: {thr}"
    return st_data


def eda_calc(data, var, stat, stat_config):
    """
    Event duration analysis for precipitation
    """
    # Statistic used for events
    event_stat = stat_config[stat]['event statistic']

    # Bins
    dur_bins = stat_config[stat]['duration bins']
    dur_bins = np.array(dur_bins) if dur_bins is not None else dur_bins
    st_bins = stat_config[stat]['statistic bins']
    st_bins = np.array(st_bins) if st_bins is not None else st_bins

    # Dry intervals
    dry = stat_config[stat]['dry events']
    dry_bins = stat_config[stat]['dry bins']
    dry_bins = np.array(dry_bins) if dry_bins is not None else dry_bins

    dur_dim = dur_bins.size+1 if dry else dur_bins.size
    frq_dim = st_bins.size-1

    # Event threshold
    thr = stat_config[stat]['event thr']

    eda_out = xa.apply_ufunc(
        eda.eda, data[var], input_core_dims=[['time']],
        output_core_dims=[['frequency', 'duration']],
        dask='parallelized', output_dtypes=[float],
        dask_gufunc_kwargs={'output_sizes': {
            'frequency': frq_dim, 'duration': dur_dim}},
        exclude_dims={'time'}, kwargs={
            'thr': thr, 'axis': -1,  'duration_bins': dur_bins,
            'event_statistic': event_stat, 'statistic_bins': st_bins,
            'dry_events': dry, 'dry_bins': dry_bins, 'keepdims': True})

    dims = list(eda_out.dims)
    eda_ds = eda_out.to_dataset().transpose(dims[-2], dims[-1],
                                            dims[0], dims[1])
    st_data = eda_ds.assign(duration_bins=dur_bins, statistic_bins=st_bins,
                            dry_bins=dry_bins)
    st_data.attrs['Description'] =\
        "EDA analysis | event statistic: {} | threshold: {}".format(
                event_stat, thr)
    return st_data


def Rxx(data, var, stat, stat_config):
    """
    Count of any time units (days, hours, etc) when
    precipitation ≥ xx mm.
    """
    in_thr = stat_config[stat]['thr']
    if in_thr is not None:
        thr = None if var not in in_thr else in_thr[var]
    else:
        thr = in_thr

    errmsg = "\nA threshold must be set for Rxx calculation"
    assert thr is not None, errmsg

    # Normalized values or not
    norm = stat_config[stat]['normalize']

    frq = xa.apply_ufunc(
        ci.Rxx, data[var], input_core_dims=[['time']], dask='parallelized',
        output_dtypes=[float],
        kwargs={'keepdims': True, 'axis': -1, 'thr': thr, 'normalize': norm})

    st_data = frq.to_dataset()
    st_data.attrs['Description'] =\
        "Rxx; frequency above threshold | threshold: {} | normalized: {}".\
        format(thr, norm)
    return st_data


def cdd(data, var, stat, stat_config):
    """
    Calculate frequencies of consecutive dry days (CDD) periods.
    See cdd function in rcat/stats/climateindex.py for more details and
    options.
    """
    in_thr = stat_config[stat]['thr']
    if in_thr is not None:
        thr = None if var not in in_thr else in_thr[var]
    else:
        thr = in_thr

    errmsg = "\nA threshold must be set for CDD calculation"
    assert thr is not None, errmsg

    # Array of CDD interval lengths to consider
    periods = stat_config[stat]['periods']

    # Calculate longest dry period?
    # N.B. If so, value will be positioned last in returned array.
    maxper = stat_config[stat]['maxper']

    # Output size
    dim_out = len(periods) if maxper else len(periods) - 1

    frq_cdd = xa.apply_ufunc(
        ci.cdd, data[var], input_core_dims=[['time']],
        output_core_dims=[['frequencies']], dask='parallelized',
        output_dtypes=[float], output_sizes={'frequencies': dim_out},
        kwargs={'keepdims': True, 'axis': -1, 'thr': thr, 'periods': periods,
                'maxper': maxper})

    dims = list(frq_cdd.dims)
    frq_cdd_ds = frq_cdd.to_dataset().transpose(dims[-1], dims[0], dims[1])
    st_data = frq_cdd_ds.assign(cdd_intervals=periods)
    st_data.attrs['Description'] =\
        f"CDD frequencies | threshold: {thr} | Longest CDD period: {maxper}"

    return st_data


def filtering(data, var, stat, stat_config):
    """
    Filter the input data
    """
    # The type of frequency cutoff
    ftype = stat_config[stat]['cutoff type']

    # The type of filter
    filt = stat_config[stat]['filter']

    # The length of filter window
    window = stat_config[stat]['window']
    assert window % 2 == 1, "Filter window must be odd"

    # The filter mode
    mode = stat_config[stat]['mode']

    # First cutoff frequency
    cutoff = stat_config[stat]['1st cutoff']
    cutoff2 = stat_config[stat]['2nd cutoff']
    if ftype == 'bandpass':
        errmsg = "'2nd cutoff' must be set for bandpass filtering"
        assert cutoff2 is not None, errmsg

    # The filtering dimensions (1D or 2D filtering)
    filt_dim = stat_config[stat]['filter dim']

    # Thresholding data
    in_thr = stat_config[stat]['thr']
    if in_thr is not None:
        if var in in_thr:
            thr = in_thr[var]
            data = data.where(data[var] >= thr)
        else:
            thr = None
    else:
        thr = in_thr

    if filt == 'lanczos':
        wgts = convolve.lanczos_filter(window, 1/cutoff, 1/cutoff2, ftype)
    else:
        # TODO
        print("No other filter implemented yet. To be done.")
        sys.exit()

    if mode == 'valid' or mode is None:
        out_dim = data.time.size - window + 1
    else:
        out_dim = data.time.size

    if filt_dim == 1:
        filtered = xa.apply_ufunc(
            convolve.filtering, data[var], input_core_dims=[['time']],
            output_core_dims=[['filtered']], dask='parallelized',
            dask_gufunc_kwargs={'output_sizes': {'filtered': out_dim}},
            output_dtypes=[float],
            kwargs={'wgts': wgts, 'dim': filt_dim, 'axis': -1, 'mode': mode})
        dims = list(filtered.dims)
        st_data = filtered.to_dataset().transpose('filtered', dims[0], dims[1])
    elif filt_dim == 2:
        print("\nSorry, 2D filtering not available yet.")
        sys.exit()
    else:
        print("\nOnly 1D and 2D filtering (dim = 1 or 2) is possible ...")
        sys.exit()

    statnm = (
        f"Filter Dimension: {filt_dim} | Filter: {filt} | Cutoff Type: {ftype}"
        f" | 1st Cutoff (time steps): {cutoff} | 2nd Cutoff (time steps): "
        f"{cutoff2} | Filter Window Size: {window}"
    )
    st_data.attrs['Description'] = "Convolved data | {}".format(statnm)

    return st_data


def _percentile_func(arr, axis=0, q=95, thr=None):
    if thr is not None:
        arr[arr < thr] = np.nan
    pctl = np.nanpercentile(arr, axis=axis, q=q)
    if axis == -1 and pctl.ndim > 2:
        pctl = np.moveaxis(pctl, 0, -1)
    return pctl


def _dask_percentile(arr, axis=0, q=95):
    if len(arr.chunks[axis]) > 1:
        msg = ('Input array cannot be chunked along the percentile '
               'dimension.')
        raise ValueError(msg)
    return da.map_blocks(np.nanpercentile, arr, axis=axis, q=q,
                         drop_axis=axis)


def _harmonic_linefit(data, keepdims=False, axis=0, var=None):
    """
    Non-linear regression line fit using first two harmonics (diurnal cycle)
    """
    from scipy import optimize

    def _f1(t, m, c1, p1):
        return m + c1*np.cos(2*np.pi*t/24 - p1)

    def _f2(t, m, c2, p2):
        return m + c2*np.cos(4*np.pi*t/24 - p2)

    def _compute(data1d, v):
        if any(np.isnan(data1d)):
            print("Data missing/masked!")
            dcycle = np.repeat(np.nan, 204)
        else:
            m, c1, p1 = optimize.curve_fit(_f1, np.arange(data1d.size),
                                           data1d)[0]
            m, c2, p2 = optimize.curve_fit(_f2, np.arange(data1d.size),
                                           data1d)[0]
            t = np.linspace(0, 23, 200)
            r = m + c1*np.cos(2*np.pi*t/24 - p1) +\
                c2*np.cos(4*np.pi*t/24 - p2)

            dcycle = np.hstack(((c1, p1, c2, p2), r))

        return dcycle

    if keepdims:
        dcycle_fit = np.apply_along_axis(_compute, axis, data, var)
    else:
        if isinstance(data, np.ma.MaskedArray):
            data1d = data.copy()
        else:
            data1d = np.array(data)
        msg = "If keepdims is False, data must be one dimensional"
        assert data1d.ndim == 1, msg
        dcycle_fit = _compute(data1d, var)

    return dcycle_fit


def _pdf_calc(data, bins=None, norm=False, keepdims=False, axis=0, thr=None,
              dry_event_thr=None):
    """
    Calculate pdf
    """
    def _compute(data1d, bins, lbins, norm, thr, dry_thr):
        if all(np.isnan(data1d)):
            print("All data missing/masked!")

            hdata = np.repeat(np.nan, lbins+1) if dry_thr is not None\
                else np.repeat(np.nan, lbins)
        else:
            if any(np.isnan(data1d)):
                data1d = data1d[~np.isnan(data1d)]

            if thr is not None:
                indata = data1d[data1d >= thr]
            else:
                indata = data1d.copy()

            if norm:
                binned = np.digitize(indata, bins)
                binned_dict = {bint: indata[np.where(binned == bint)]
                               if bint in binned else np.nan
                               for bint in range(1, len(bins))}

                # Mean value for each bin
                means = np.array([np.mean(arr) if not np.all(np.isnan(arr))
                                  else 0.0 for k, arr in binned_dict.items()])

                # Occurrences and frequencies
                ocrns = np.array([arr.size if not np.all(np.isnan(arr))
                                  else 0 for k, arr in binned_dict.items()])
                frequency = ocrns/np.nansum(ocrns)
                C = frequency*means     # Relative contribution per bin
                hdata = C/np.nansum(C)     # Normalized contribution per bin

            else:
                hdata = np.histogram(indata, bins=bins,
                                     density=True)[0]

            # Add dry events
            if dry_thr is not None:
                dry_events = np.sum(data1d < dry_thr)
                hdata = np.hstack((dry_events, hdata))

        return hdata

    # Set number of bins to 10 (np.histogram default) if bins not provided.
    inbins = 10 if bins is None else bins
    lbins = inbins if isinstance(inbins, int) else len(inbins) - 1

    if keepdims:
        hist = np.apply_along_axis(_compute, axis, data, bins=inbins,
                                   lbins=lbins, norm=norm, thr=thr,
                                   dry_thr=dry_event_thr)
    else:
        if isinstance(data, np.ma.MaskedArray):
            data1d = data.compressed()
        else:
            data1d = np.array(data).ravel()
        hist = _compute(data1d, bins=inbins, lbins=lbins, norm=norm, thr=thr,
                        dry_thr=dry_event_thr)

    return hist
