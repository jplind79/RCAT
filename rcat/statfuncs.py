"""
Statistics module
---------------
Functions for different statistical calculations.

Created: Autumn 2016
Authors: Petter Lind & David Lindstedt
"""
import numpy as np
import multiprocessing as mp
from functools import reduce
# from itertools import product


def run_mean(x, N):
    """ Calculating running mean. x is the data vector, N the window. """

    return np.convolve(x, np.ones((N,))/N)[(N-1):]


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def rolling_sum(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:]


def hotdays_calc(data, thrs_75):
    """
    Calculates number of hotdays: days with mean temperature
    above the 75th percentile of climatology.
    Parameters
    ----------
    data: array
        1D-array of temperature input timeseries
    thrs_75: float
        75th percentile daily mean value from climatology
    """
    hotdays = ((data > thrs_75)).sum()
    return hotdays


def extr_hotdays_calc(data, thrs_95):
    """
    Calculates number of extreme hotdays: days with mean temperature
    above the 95th percentile of climatology.
    Parameters
    ----------
    data: array
        1D-array of temperature input timeseries
    thrs_95: float
        95th percentile daily mean value from climatology
    """
    hotdays = ((data > thrs_95)).sum()
    return hotdays


def tropnights_calc(data):
    """
    Calculates number of tropical nights: days with min temperature
    not below 20 degrees C.
    Parameters
    ----------
    data: array
        1D-array of minimum temperatures timeseries in degrees Kelvin
    """
    tropnts = ((data > 293)).sum()
    return tropnts


def ehi(data, thr_95, axis=0, keepdims=False):
    """
    Calculates Excessive Heat Index (EHI).
    Parameters
    ----------
    data: list/array
        1D/2D array of daily temperature timeseries
    thr_95: float
        95th percentile daily mean value from climatology
    axis: int
        The axis along which the calculation is applied (default 0).
    keepdims: boolean
        If data is 2d (time in third dimesion) and keepdims is set to True,
        calculation is applied to the zeroth axis (time) and returns a 2d array
        of freq-int dists. If set to False (default) all values are
        collectively assembled before calculation.
    Returns
    -------
    EHI: float
        Excessive heat index
    """
    def ehi_calc(pdata, thr_95):
        if all(np.isnan(pdata)):
            print("All data missing/masked!")
            ehi = np.nan
        else:
            run_mean = moving_average(pdata, 3)
            ehi = ((run_mean > thr_95)).sum()

        return ehi

    if keepdims:
        EHI = np.apply_along_axis(ehi_calc, axis, data, thr_95)
    else:
        EHI = ehi_calc(data, thr_95)

    return EHI


def cdd_calc(data, thr=1.0):
    """
    Calculates the Consecutive Dry Days index (CDD).
    Parameters
    ----------
    data: array
        1D-array of precipitation timeseries in mm
    thr: float
        Value of threshold to define dry day. Default 1 mm.
    Returns
    -------
    dlen: list
        list with lengths of each dry day event in timeseries
    """
    import itertools
    cdd = [list(x[1]) for x in itertools.groupby(data, lambda x: x > thr)
           if not x[0]]
    d_len = [len(f) for f in cdd]
    return d_len


def Rxx(data, thr, axis=0, normalize=False, keepdims=False):
    """
    Rxx mm, count of any time units (days, hours, etc) when precipitation ≥
    xx mm: Let RRij be the precipitation amount on time unit i in period j.
    Count the number of days where RRij ≥ xx mm.

    Parameters
    ----------
    data: array
        1D/2D data array, with time steps on the zeroth axis (axis=0).
    thr: float/int
        Threshold to be used; eg 10 for R10, 20 R20 etc.
    normalize: boolean
        If True (default False) the counts are normalized by total number of
        time units in each array/grid point. Returned values will then be
        fractions.
    keepdims: boolean
        If False (default) calculation is performed on all data collectively,
        otherwise for each timeseries on each point in 2d space. 'Axis' then
        defines along which axis the timeseries are located.

    Returns
    -------
    Rxx: list/array
        1D/2D array with calculated Rxx indices.
    """
    def Rxx_calc(pdata, thr):
        # Flatten data to one dimension
        if isinstance(pdata, np.ma.MaskedArray):
            data1d = pdata.compressed()
        elif isinstance(pdata, (list, tuple)):
            data1d = np.array(pdata)
        else:
            data1d = pdata.ravel()

        if all(np.isnan(data1d)):
            print("All data missing/masked!")
            Rxx = np.nan
        else:
            if any(np.isnan(data1d)):
                data1d = data1d[~np.isnan(data1d)]
            Rxx = (data1d > thr).sum()
            if normalize:
                Rxx /= data1d.size

        return Rxx

    if keepdims:
        RXX = np.apply_along_axis(Rxx_calc, axis, data, thr)
    else:
        RXX = Rxx_calc(data, thr)

    return RXX


def RRpX(data, percentile, thr=None, axis=0, keepdims=False):
    """
    RRpX mm, total amount of precipitation above the percentile threshold pXX;
    RR ≥ pXX mm: Let RRij be the daily precipitation amount on day i in period
    j. Sum the precipitation for all days where RRij ≥ pXX mm.

    Parameters
    ----------
    data: array
        1D/2D data array, with time steps on the zeroth axis (axis=0).
    percentile: int
        Percentile that defines the threshold.
    thr: float/int
        Pre-thresholding of data to do calculation for wet days/hours only.
    keepdims: boolean
        If False (default) calculation is performed on all data collectively,
        otherwise for each timeseries on each point in 2d space. 'Axis' then
        defines along which axis the timeseries are located.

    Returns
    -------
    RRpx: list/array
        1D/2D array with calculated RRpXX indices.
    """
    def Rpxx_calc(pdata, pctl, thr):
        # Flatten data to one dimension
        if isinstance(pdata, np.ma.MaskedArray):
            data1d = pdata.compressed()
        elif isinstance(pdata, (list, tuple)):
            data1d = np.array(pdata)
        else:
            data1d = pdata.ravel()

        if all(np.isnan(data1d)):
            print("All data missing/masked!")
            Rpxx = np.nan
        else:
            if thr is not None:
                data1d[data1d < thr] = np.nan
            if all(np.isnan(data1d)):
                print("All data missing/masked!")
                Rpxx = np.nan
            else:
                if any(np.isnan(data1d)):
                    data1d = data1d[~np.isnan(data1d)]
                pctl_val = np.percentile(data1d, pctl)
                Rpxx = data1d[data1d >= pctl_val].sum()

        return Rpxx

    if keepdims:
        RRpx = np.apply_along_axis(Rpxx_calc, axis, data, percentile, thr)
    else:
        RRpx = Rpxx_calc(data, pctl=percentile, thr=thr)

    return RRpx


def RRtX(data, thr, axis=0, keepdims=False):
    """
    RRtX mm, total amount of precipitation above the threshold 'thr'.

    Parameters
    ----------
    data: array
        1D/2D data array, with time steps on the zeroth axis (axis=0).
    thr: int
        Threshold that defines the threshold above which data is summed.
    keepdims: boolean
        If False (default) calculation is performed on all data collectively,
        otherwise for each timeseries on each point in 2d space. 'Axis' then
        defines along which axis the timeseries are located.

    Returns
    -------
    RRtx: list/array
        1D/2D array with calculated RRtXX indices.
    """
    def Rtxx_calc(pdata, thr):
        # Flatten data to one dimension
        if isinstance(pdata, np.ma.MaskedArray):
            data1d = pdata.compressed()
        elif isinstance(pdata, (list, tuple)):
            data1d = np.array(pdata)
        else:
            data1d = pdata.ravel()

        if all(np.isnan(data1d)):
            print("All data missing/masked!")
            Rtxx = np.nan
        else:
            if any(np.isnan(data1d)):
                data1d = data1d[~np.isnan(data1d)]
            Rtxx = data1d[data1d >= thr].sum()

        return Rtxx

    if keepdims:
        RRtx = np.apply_along_axis(Rtxx_calc, axis, data, thr)
    else:
        RRtx = Rtxx_calc(data, thr=thr)

    return RRtx


def SDII(data, thr=1.0, axis=0, keepdims=False):
    """
    SDII, Simple pricipitation intensity index: Let RRwj be the daily
    precipitation amount on wet days, w (RR ≥ 1mm) in period j.

    Parameters
    ----------
    data: list/array
        2D array.
    thr: float/int
        threshold for wet events (wet days/hours etc)
    axis: int
        The axis along which the calculation is applied (default 0).
    keepdims: boolean
        If data is 2d (time in third dimesion) and keepdims is set to True,
        calculation is applied to the zeroth axis (time) and returns a 2d array
        of freq-int dists. If set to False (default) all values are
        collectively assembled before calculation.
    """

    def sdii_calc(pdata, thr):
        # Flatten data to one dimension
        if isinstance(pdata, np.ma.MaskedArray):
            data1d = pdata.compressed()
        elif isinstance(pdata, (list, tuple)):
            data1d = np.array(pdata)
        else:
            data1d = pdata.ravel()

        if all(np.isnan(data1d)):
            print("All data missing/masked!")
            sdii = np.nan
        else:
            if any(np.isnan(data1d)):
                data1d = data1d[~np.isnan(data1d)]
            sdii = data1d[data1d >= thr].sum()/data1d[data1d >= thr].size
        return sdii

    if keepdims:
        SDII = np.apply_along_axis(sdii_calc, axis, data, thr)
    else:
        SDII = sdii_calc(data, thr)

    return SDII


def block_bootstr(data, block=5, nrep=500, nproc=1):
    """
    This is a block boostrap function, converted from R into python, based on:
    http://stat.wharton.upenn.edu/~buja/STAT-541/time-series-bootstrap.R

    Parameters
    ----------
    data: list/array
        1D data array on which to perform the block bootstrap.
    block: int
        the block length to be used. Default is 5.
    nrep: int
        the number of resamples produced in the bootstrap. Default is 500.
    nproc: int
        Number of processors, default 1. If larger than 1, multiple processors
        are used in parallell using the multiprocessing module.

    Returns
    -------
    arrBt: Array
        2D array with bootstrap samples; rows are the samples, columns the
        values.
    """

    # Make sure the data is a numpy array
    data = np.array(data)

    error_message = "*** ERROR ***\n Data array should be 1D"
    assert np.ndim(data) == 1, error_message

    if nproc > 1:
        # Number of cores to be used.
        # Available cores on system is a constraint
        nr_procs_set = np.minimum(nproc, mp.cpu_count())

        pool = mp.Pool(processes=nr_procs_set)
        computations = [pool.apply_async(_get_bootsample,
                        args=(data, block)) for j in range(nrep)]

        arrBt = [k.get() for k in computations]
    else:
        arrBt = [_get_bootsample(data, block) for irep in range(nrep)]

    return np.array(arrBt)


def _get_bootsample(data, block):
    """
    Function to sample 1D input data by filling a vector
    with random blocks extracted from data.
    """

    N = data.size                   # size of data series
    k = block                       # size of moving blocks
    nk = int(np.ceil(N/k))          # number of blocks

    dataBt = np.repeat(np.nan, N)   # local vector for a bootstrap replication

    # fill the vector with random blocks by
    # randomly sampling endpoints and copying blocks
    for i in range(nk):
        endpoint = np.random.randint(k, N+1, size=1)
        dataBt[(i-1)*k + np.array(range(k))] = \
            data[endpoint-np.array(range(k))[::-1]-1]

    return dataBt[0:N]


def _mproc_get_bootsamples(data, nx, j, block):
    # print("Getting bootsamples for ny pos: {}".format(j))
    # print()
    bs = np.array([_get_bootsample(data[:, i], block=block)
                   for i in range(nx)])
    return bs


def freq_int_dist(data, keepdims=False, axis=0, bins=10, thr=None,
                  density=True, ci=False, bootstrap=False, nmc=500,
                  block=1, ci_level=95, nproc=1):
    """
    Calculates frequency (probability) - instensity distriutions.

    Parameters
    ----------
    data: array
        2D or 1D array of data. All data points are collectively used in the
        freq-instensity calculation unless 'keepdims' is True. Then calculation
        is performed along the dimension defined by axis argument (default 0).
    keepdims: boolean
        If data is 2d (time in third dimesion) and keepdims is set to True,
        calculation is applied to the dimension defined by axis argument
        (default 0) and returns a 2d array of freq-int dists. If set to False
        (default) all values are collectively assembled before calculation.
    axis: int
        The axis over which to apply the calculation if keepdims is set to
        True. Default is 0.
    bins: int/list/array
        If an int, it defines the number of equal-width bins in the given
        range (10, by default). If a sequence, it defines the bin edges,
        including the rightmost edge, allowing for non-uniform bin widths.
        If bins is set to 'None' they will be automatically calculated.
    thr: float
        Value of threshold if thresholding data. Default None.
    density: boolean
        If True (default) then the value of the probability density function at
        each bin is returned, otherwise the number of samples per bin.
    bootstrap: boolean
        If to use block bootstrap to calculate confidence interval.
    nmc: int/float
        Number of bootstrap samples to use.
    block: int/float
        Size of block to use in block bootstrap
    ci_level: int/float
        The confidence interval level to use (eg 95, 99 representing 95%, 99%
        levels)
    nproc: int
        Number of processes to use in bootstrap calculation. Default 1.

    Returns
    -------
    pdf: array
        data array with size len(bins)-1 with counts/probabilities
    ci: dict
        data dictionary with confidence level for each bin; keys
        'min_levels'/'max_levels' with corresponding values. If bootstrap is
        False, then None values are returned.
    """

    def pdf_calc(pdata, bins):
        lbins = bins if isinstance(bins, int) else len(bins) - 1
        # Flatten data to one dimension
        if isinstance(pdata, np.ma.MaskedArray):
            data1d = pdata.compressed()
        elif isinstance(pdata, (list, tuple)):
            data1d = np.array(pdata)
        else:
            data1d = pdata.ravel()

        if all(np.isnan(data1d)):
            print("All data missing/masked!")
            # hdata = np.repeat(np.nan, data1d.size)
            hdata = np.repeat(np.nan, lbins)
            min_level = np.nan
            max_level = np.nan
        else:
            if any(np.isnan(data1d)):
                data1d = data1d[~np.isnan(data1d)]

            if thr is not None:
                indata = data1d[data1d >= thr]
            else:
                indata = data1d

            hdata = np.histogram(indata, bins=bins,
                                 density=density)[0]

            if bootstrap:
                if isinstance(pdata, (np.ma.MaskedArray, np.ndarray)):
                    if pdata.ndim > 2 and not keepdims:
                        nx = pdata.shape[1]
                        ny = pdata.shape[2]
                        iterlist = [(pdata[:, :, ix], nx, ix, block)
                                    for ix in range(ny)]
                        bs_hdata = []
                        for bs in range(nmc):
                            print("Generating pdf for bootsample\n{}".format(
                                bs))
                            pool = mp.Pool(processes=nproc)
                            comps = pool.starmap_async(_mproc_get_bootsamples,
                                                       iterlist)
                            pool.close()
                            pool.join()
                            arr = np.array(comps.get())
                            arr1d = arr.ravel() if isinstance(arr, np.ndarray)\
                                else arr.compressed()
                            if thr is not None:
                                indata = arr1d[arr1d >= thr]
                            else:
                                indata = arr1d
                            bs_hdata.append(np.histogram(indata, bins=bins,
                                                         density=density)[0])
                        bs_hdata = np.array(bs_hdata)
                    else:
                        btsamples = block_bootstr(indata, block=block,
                                                  nrep=nmc, nproc=nproc)
                        bs_hdata = np.array([np.histogram(btsamples[i, :],
                                                          bins=bins,
                                                          density=density)[0]
                                            for i in range(nmc)])
                else:
                    btsamples = block_bootstr(indata, block=block,
                                              nrep=nmc, nproc=nproc)
                    bs_hdata = np.array([np.histogram(btsamples[i, :],
                                                      bins=bins,
                                                      density=density)[0]
                                        for i in range(nmc)])
                alpha = 100 - ci_level
                min_level = [np.nanpercentile(bs_hdata[:, x], alpha/2.)
                             for x in range(bs_hdata.shape[1])]
                max_level = [np.nanpercentile(bs_hdata[:, x], 100-alpha/2.)
                             for x in range(bs_hdata.shape[1])]
            else:
                min_level = None
                max_level = None

        ci = {'min_levels': min_level, 'max_levels': max_level}

        return (hdata, ci)

    if keepdims:
        hist_o, ci_data = np.apply_along_axis(pdf_calc, axis=axis, arr=data,
                                              bins=bins)
        hist = reduce((lambda x, y: np.c_[x, y]),
                      hist_o.ravel()).reshape(hist_o[0, 0].size,
                                              hist_o.shape[0], hist_o.shape[1])
    else:
        hist, ci_data = pdf_calc(data, bins=bins)

    outdata = (hist, ci_data) if ci else hist

    return outdata


def asop(data, keepdims=False, axis=0, bins=None, thr=None, return_bins=False):
    """
    Analyzing Scales of Precipitation (ASoP)
    Source: Klingaman et al. (2017)
    https://www.geosci-model-dev.net/10/57/2017/

    Parameters
    ----------
    data: array
        2D or 1D array of data. All data points are collectively used in the
        asop calculation unless 'keepdims' is True. Then calculation
        is performed along zeroth axis (expected time dimension).
    keepdims: boolean
        If data is 2d (time in third dimesion) and keepdims is set to True,
        calculation is applied to the dimension defined by axis argument
        (default 0) and returns a 2d array of asop components. If set to False
        (default) all values are collectively assembled before calculation.
    axis: int
        The axis over which to apply the calculation if keepdims is set to
        True. Default is 0.
    bins: list/array
        Defines the bin edges, including the rightmost edge, allowing for
        non-uniform bin widths. If bins is set to 'None' they will be
        automatically calculated using Klingaman bins; function bins_calc in
        this module.
    thr: float
        Value of threshold if thresholding data. Default None.
    return_bins: boolean
        If set to True (default False), bins that have been used in the
        calculation are returned.

    Returns
    -------
    Cfactor: array
        data array with relative contribution per bin to the total mean.
    FCfactor: array
        data array with relative contribution per bin independent
        of the total mean.
    bins_ret: array
        If return_bins is True, the array of bin edges is returned.
    """

    def asop_calc(pdata, bins):
        # Flatten data to one dimension
        if isinstance(pdata, np.ma.MaskedArray):
            data1d = pdata.compressed()
        elif isinstance(pdata, (list, tuple)):
            data1d = np.array(pdata)
        else:
            data1d = pdata.ravel()

        if all(np.isnan(data1d)):
            print("All data missing/masked!")
            C = np.repeat(np.nan, bins.size-1)
            FC = np.repeat(np.nan, bins.size-1)
        else:
            if any(np.isnan(data1d)):
                data1d = data1d[~np.isnan(data1d)]

            if thr is not None:
                indata = data1d[data1d >= thr]
            else:
                indata = data1d

            binned = np.digitize(indata, bins)

            # Put into dictionary; keys are bin number,
            # values data in respective bin
            binned_dict = {bint: indata[np.where(binned == bint)]
                           if bint in binned else np.nan
                           for bint in range(1, len(bins))}

            # --- Calculate statistics of bins --- #
            # Mean value for each bin
            means = np.array([np.mean(arr) if not np.all(np.isnan(arr))
                              else 0.0 for k, arr in binned_dict.items()])

            # Occurrences and frequencies
            ocurrence = np.array([arr.size if not np.all(np.isnan(arr))
                                  else 0 for k, arr in binned_dict.items()])
            frequency = ocurrence/np.nansum(ocurrence)

            # Relative contribution per bin to the total mean
            C = frequency*means

            # Contribution per bin independent of the total mean
            FC = C/np.nansum(C)

        output = np.stack((C, FC), axis=0)

        return output

    if bins is None:
        bin_data = np.arange(np.floor(np.nanmin(data)),
                             np.ceil(np.nanmax(data)))
        bins = [bins_calc(n) for n in bin_data]
        print("Bins are not part of arguments in function call!")
        print("Calculated internally to: {}".format(bins))
    if keepdims:
        asop_comp = np.apply_along_axis(asop_calc, axis=axis, arr=data,
                                        bins=bins)
    else:
        asop_comp = asop_calc(data, bins=bins)

    if return_bins:
        results = (asop_comp, bins)
    else:
        results = asop_comp

    return results


def bins_calc(n):
    """
    Calculates bins with edges according to Eq. 1 in Klingaman et al. (2017);
    https://www.geosci-model-dev.net/10/57/2017/

    Parameter
    ---------
    n: array/list
        1D array or list with bin numbers

    Returns
    -------
    bn: array
        1D array of bin edges
    """
    bn = np.e**(np.log(0.005)+(n*(np.log(120)-np.log(0.005))**2/59)**(1/2))

    return bn


def prob_of_exceed(data, keepdims=False, axis=0, thr=None, pctls_levels=None):
    """
    Calculates probability of exceedance distriutions.

    Parameters
    ----------
    data: array
        2D or 1D array of data. All data points are collectively used in the
        freq-instensity calculation unless 'keepdims' is True. Then calculation
        is performed along zeroth axis.
    pctls_levels: 'default', None or array/list
        If set to 'default', probability levels of exceedance are defined by a
        set of percentiles ranging from 0-100 and calculated from input data.
        If an array or list, these levels (between 0 and 100) will be used
        instead.  Default is None in which case input data is merely ranked
        from 0 to 1.
    keepdims: boolean
        If data is 2d (time in third dimesion) and keepdims is set to True,
        calculation is applied to the zeroth axis (time) and returns a 2d array
        of freq-int dists. If set to False (default) all values are
        collectively assembled before calculation.
    axis: int
        The axis over which to apply the calculation if keepdims is set to
        True. Default is 0.
    thr: float
        Value of threshold if thresholding data. Default None.

    Returns
    -------
    eop: array
        exceedance of probability array.
    """
    import pandas as pd

    def eop_calc(pdata, thr):
        # Flatten data to one dimension
        if isinstance(pdata, np.ma.MaskedArray):
            data1d = pdata.compressed()
        elif isinstance(pdata, (list, tuple)):
            data1d = np.array(pdata)
        else:
            data1d = pdata.ravel()

        # Check for missing values
        if all(np.isnan(data1d)):
            print("All data missing/masked!")
            odata = np.repeat(np.nan, data1d.size)
        else:
            if any(np.isnan(data1d)):
                data1d = data1d[~np.isnan(data1d)]

            # Thresholding of data
            if thr is not None:
                indata = data1d[data1d >= thr]
            else:
                indata = data1d

            # Probability of exceedance calculation
            if not pctls_levels:
                cum_dist = np.linspace(0., 1., indata.size)
                indata.sort()
                ser_cdf = pd.Series(cum_dist, index=indata)
                odata = 1. - ser_cdf
            else:
                if pctls_levels == 'default':
                    pctls = np.hstack((np.linspace(0, 99, 100),
                                       np.linspace(99, 99.9, 10),
                                       np.linspace(99.9, 99.99, 10)))
                else:
                    pctls = pctls_levels
                lvls = np.percentile(indata, pctls)
                ser_cdf = pd.Series(pctls/100, index=lvls)
                odata = 1. - ser_cdf

        return odata

    if keepdims:
        eop = np.apply_along_axis(eop_calc, axis, data, thr)
    else:
        eop = eop_calc(data, thr=thr)

    return eop


def perkins_skill_score(p_mod, p_obs, axis=0):
    """
    Calculate the Perkins Skill Score (PSS).

    Parameters
    ----------
    p_mod, p_obs: list/array
        1d or 2d arrays with frequency of values (probability) in a given bin
        from the model and observations respectively.
        Make sure that the sum of probabilities over all the bins should be
        equal to one. This depends on how the pdf was calculated. Bins with
        unity width gives total prob of one.
    axis: int
        If data is 2d, the PSS will be calculated along this axis. Default
        axis is zero.

    Returns
    -------
    pss: float/array
        Returns Perkins Skill Score, single float or array with floats.
    """
    def pss_calc(x, y):
        return np.sum(np.minimum(x, y))

    if isinstance(p_mod, (list, tuple)):
        p_mod = np.array(p_mod)
        p_obs = np.array(p_obs)
    if np.ndim(p_mod) > 2:
        pss = np.apply_along_axis(pss_calc, axis, p_mod, p_obs)
    else:
        pss = pss_calc(p_mod, p_obs)

    return pss
