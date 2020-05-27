import sys
import numpy as np
import dask.array as da
import xarray as xa
from rcat.stats import ASoP
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
        'moments': {
            'moment stat': ['D', 'mean'],
            'vars': [],
            'resample resolution': None,
            'pool data': False,
            'thr': None,
            'chunk dimension': 'time'},
        'seasonal cycle': {
            'vars': [],
            'resample resolution': None,
            'pool data': False,
            'stat method': 'mean',
            'thr': None,
            'chunk dimension': 'time'},
        'annual cycle': {
            'vars': [],
            'resample resolution': None,
            'pool data': False,
            'stat method': 'mean',
            'thr': None,
            'chunk dimension': 'time'},
        'diurnal cycle': {
            'vars': [],
            'resample resolution': None,
            'hours': None,
            'dcycle stat': 'amount',
            'stat method': 'mean',
            'thr': None,
            'pool data': False,
            'chunk dimension': 'time'},
        'dcycle harmonic': {
            'vars': [],
            'resample resolution': None,
            'pool data': False,
            'dcycle stat': 'amount',
            'thr': None,
            'chunk dimension': 'time'},
        'asop': {
            'vars': ['pr'],
            'resample resolution': None,
            'pool data': False,
            'nr_bins': 50,
            'thr': None,
            'chunk dimension': 'space'},
        'pdf': {
            'vars': [],
            'resample resolution': None,
            'pool data': False,
            'bins': None,
            'normalized': False,
            'thr': None,
            'chunk dimension': 'space'},
        'percentile': {
            'vars': [],
            'resample resolution': None,
            'pool data': False,
            'pctls': [95, 99],
            'thr': None,
            'chunk dimension': 'space'},
        'Rxx': {
            'vars': ['pr'],
            'resample resolution': None,
            'pool data': False,
            'normalize': False,
            'thr': 1.0,
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
        'moments': moments,
        'seasonal cycle': seasonal_cycle,
        'annual cycle': annual_cycle,
        'percentile': percentile,
        'diurnal cycle': diurnal_cycle,
        'dcycle harmonic': dcycle_harmonic_fit,
        'pdf': freq_int_dist,
        'asop': asop,
        'Rxx': Rxx,
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
    if np.any(ds.time.dt.minute > 0):
        print("Shifting time stamps to whole hours!")
        ds.time.values = ds.time.dt.ceil('H').values
    else:
        pass
    return ds


def _get_freq(tf):
    from functools import reduce

    d = [j.isdigit() for j in tf]
    freq = int(reduce((lambda x, y: x+y), [x for x, y in zip(tf, d) if y]))
    unit = reduce((lambda x, y: x+y), [x for x, y in zip(tf, d) if not y])

    if unit in ('M', 'Y'):
        freq = freq*30 if unit == 'M' else freq*365
        unit = 'D'

    return freq, unit


############################################################
#                                                          #
#                   STATISTICS FUNCTIONS                   #
#                                                          #
############################################################

def moments(data, var, stat, stat_config):
    """
    Calculate standard moment statistics: avg, median, std, max/min
    """
    _mstat = deepcopy(stat_config[stat]['moment stat'])
    mstat = _mstat[var] if isinstance(_mstat, dict) else _mstat
    if not isinstance(mstat[0], np.int):
        mstat[0] = str(1) + mstat[0]
    in_thr = stat_config[stat]['thr']
    if in_thr is not None:
        if var in in_thr:
            thr = in_thr[var]
            data = data.where(data[var] >= thr)
        else:
            thr = None
    else:
        thr = in_thr

    diff = data.time.values[1] - data.time.values[0]
    nsec = diff.astype('timedelta64[s]')/np.timedelta64(1, 's')
    tr, fr = _get_freq(mstat[0])
    sec_resample = to_timedelta(tr, fr).total_seconds()
    expr = "data[var].resample(time='{}').{}('time').dropna('time', 'all')"\
        .format(mstat[0], mstat[1])

    if mstat[0] == 'all':
        st_data = eval("data.{}(dim='time', skipna=True)".format(mstat))
    else:
        if nsec >= sec_resample:
            print("* Data already at the same or coarser time resolution "
                  "as statistic!\n* Keeping data as is ...\n")
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
        q = float(tstat.split(' ')[1])
        st_data = data[var].groupby('time.season').reduce(
            _dask_percentile, dim='time', q=q, allow_lazy=True)
        st_data = st_data.to_dataset()
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
        q = float(tstat.split(' ')[1])
        st_data = data[var].groupby('time.month').reduce(
            _dask_percentile, dim='time', q=q, allow_lazy=True)
        st_data = st_data.to_dataset()
    else:
        st_data = eval("data.groupby('time.month').{}('time')".format(
            tstat))
    st_data.attrs['Description'] =\
        "Annual cycle | Month stat: {} | Threshold: {}".format(
                tstat, thr)
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

    data = _check_hours(data)

    if dcycle_stat == 'amount':
        tstat = stat_config[stat]['stat method']
        if 'percentile' in tstat:
            q = float(tstat.split(' ')[1])
            dcycle = data[var].groupby('time.hour').reduce(
                _dask_percentile, dim='time', q=q, allow_lazy=True)
            dcycle = dcycle.to_dataset()
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
    if var not in stat_config[stat]['bins']:
        dmn = data[var].min(skipna=True)
        dmx = data[var].max(skipna=True)
        bins = np.linspace(dmn, dmx, 20)
    else:
        bin_r = stat_config[stat]['bins'][var]
        bins = np.arange(bin_r[0], bin_r[1], bin_r[2])
    lbins = bins.size - 1
    in_thr = stat_config[stat]['thr']
    if in_thr is not None:
        thr = None if var not in in_thr else in_thr[var]
    else:
        thr = in_thr
    normalized = stat_config[stat]['normalized']
    if isinstance(normalized, bool):
        norm = normalized
    else:
        norm = False if var not in normalized else normalized[var]

    if var == 'pr':
        mask = ((np.isnan(data[var])) | (data[var] >= 0.0))
        data_tmp = xa.where(~mask, 0.0, data[var])
        data = data_tmp.to_dataset()

    pdf = xa.apply_ufunc(
        _pdf_calc, data[var], input_core_dims=[['time']],
        output_core_dims=[['bins']], dask='parallelized',
        output_dtypes=[float], output_sizes={'bins': lbins+1},
        kwargs={'keepdims': True, 'bins': bins, 'axis': -1, 'norm': norm,
                'thr': thr, 'var': var})
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
        nr_bins = np.arange(50)
    else:
        nr_bins = np.arange(stat_config[stat]['nr_bins'])
    bins = [ASoP.bins_calc(n) for n in nr_bins]
    bins = np.insert(bins, 0, 0.0)
    lbins = bins.size - 1
    in_thr = stat_config[stat]['thr']
    if in_thr is not None:
        thr = None if var not in in_thr else in_thr[var]
    else:
        thr = in_thr
    asop = xa.apply_ufunc(
        ASoP.asop, data[var], input_core_dims=[['time']],
        output_core_dims=[['factors', 'bins']], dask='parallelized',
        output_dtypes=[float], output_sizes={'factors': 2, 'bins': lbins},
        kwargs={'keepdims': True, 'axis': -1, 'bins': bins})
    dims = list(asop.dims)

    # N.B. This does not work in rcat yet! Variable name need still to be 'pr'
    # C = asop.isel(factors=0)
    # FC = asop.isel(factors=1)
    # dims = list(C.dims)
    # C_ds = C.to_dataset().transpose(dims[-1], dims[0], dims[1])
    # FC_ds = FC.to_dataset().transpose(dims[-1], dims[0], dims[1])
    # asop_ds = = xa.Dataset.merge(C_ds, FC_ds)
    asop_ds = asop.to_dataset().transpose(dims[-2], dims[-1], dims[0], dims[1])
    st_data = asop_ds.assign(bin_edges=bins, factors=['C', 'FC'])
    st_data.attrs['Description'] =\
        "ASoP analysis | threshold: {}".format(thr)
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
              var=None):
    """
    Calculate pdf
    """
    def _compute(data1d, bins, lbins, norm=norm, thr=thr, v=var):
        if all(np.isnan(data1d)):
            print("All data missing/masked!")
            # hdata = np.repeat(np.nan, data1d.size)
            hdata = np.repeat(np.nan, lbins+1)
        else:
            if any(np.isnan(data1d)):
                data1d = data1d[~np.isnan(data1d)]

            if v == 'pr':
                data1d[data1d < 0.0] = 0.0
                # dry_events = np.sum(data1d < 0.001)
                dry_events = np.sum(data1d < 0.1)
            else:
                dry_events = None

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
                hdata = np.hstack((dry_events, hdata))
            else:
                hdata = np.histogram(indata, bins=bins,
                                     density=True)[0]
                hdata = np.hstack((dry_events, hdata))
        return hdata

    # Set number of bins to 10 (np.histogram default) if bins not provided.
    inbins = 10 if bins is None else bins
    lbins = inbins if isinstance(inbins, int) else len(inbins) - 1

    if keepdims:
        hist = np.apply_along_axis(_compute, axis, data, bins=inbins,
                                   lbins=lbins, norm=norm, thr=thr, v=var)
    else:
        if isinstance(data, np.ma.MaskedArray):
            data1d = data.compressed()
        else:
            data1d = np.array(data).ravel()
        hist = _compute(data1d, bins=inbins, norm=norm, thr=thr, v=var,
                        lbins=lbins)

    return hist
