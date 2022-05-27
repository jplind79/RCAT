"""
Climate indices
---------------
Functions for various climate index calculations.

Authors: Petter Lind & David Lindstedt
Created: Autumn 2016
Updates:
        May 2020
"""

import numpy as np
from .arithmetics import run_mean


def hotdays_calc(data, thr_p75):
    """
    Calculate number of hotdays.

    Return days with mean temperature above the 75th percentile
    of climatology.

    Parameters
    ----------
    data: array
        1D-array of temperature input timeseries
    thr_p75: float
        75th percentile daily mean value from climatology
    """
    hotdays = ((data > thr_p75)).sum()
    return hotdays


def extr_hotdays_calc(data, thr_p95):
    """
    Calculate number of extreme hotdays.

    Return days with mean temperature above the 95th percentile
    of climatology.

    Parameters
    ----------
    data: array
        1D-array of temperature input timeseries
    thr_p95: float
        95th percentile daily mean value from climatology
    """
    xtr_hotdays = ((data > thr_p95)).sum()
    return xtr_hotdays


def tropnights_calc(data):
    """
    Calculate number of tropical nights.

    Return days with minimum temperature not below 20 degrees C.

    Parameters
    ----------
    data: array
        1D-array of minimum temperatures timeseries in degrees Kelvin
    """
    tropnts = ((data > 293)).sum()
    return tropnts


def ehi(data, thr_95, axis=0, keepdims=False):
    """
    Calculate Excessive Heat Index (EHI).

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
        of calculated statistics. If set to False (default) all values are
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
            # run_mean = moving_average(pdata, 3)
            rmean = run_mean(pdata, 3)
            ehi = ((rmean > thr_95)).sum()

        return ehi

    if keepdims:
        EHI = np.apply_along_axis(ehi_calc, axis, data, thr_95)
    else:
        EHI = ehi_calc(data, thr_95)

    return EHI


def cdd(data, thr=1.0, periods=np.arange(1, 61), maxper=False,
        axis=0, keepdims=False):
    """
    Calculate the Consecutive Dry Days index (CDD).

    Parameters
    ----------
    data: array
        1D/2D daily precipitation data array in mm.
    thr: float
        Value of threshold to define dry day. Default 1 mm.
    periods: list/array
        Array of lenghts of dry periods to consider; e.g.
        [1, 3, 10, 14, 21, 30] computes frequency of dry periods with lengths
        1-3 days, 3-10 days, etc. Leftmost interval edge is included, not the
        right. Default periods is set to 60 days with 1 day increment.
    maxper: boolean
        If set to True the longest CDD period and positioned at last position
        in returned array. Default False.
    axis: int
        Along which axis to calculate cdd. Defaults to 0
    keepdims: boolean
        If False (default) calculation is performed on all data collectively,
        otherwise for each timeseries on each point in 2d space. 'Axis' then
        defines along which axis the timeseries are located.

    Returns
    -------
    dlen: list
        list with lengths of each dry day event in timeseries
    cdd: list/array
        1D/2D array with frequencies of cdd intervals. For intervals where non
        exists, positions are set to NaN. Length of returned array (along
        computed 'axis') is equal to length of 'periods' list/array minus 1.
    """
    import itertools

    def cdd_calc(data1d, thr, lbins):
        dim_out = len(periods) if maxper else len(periods) - 1
        if all(np.isnan(data1d)):
            print("All data missing/masked!")
            cdd_out = np.repeat(np.nan, dim_out)
        else:
            cdd = [list(x[1]) for x in itertools.groupby(
                data1d, lambda x: x > thr) if not x[0]]
            cdd_len = [len(f) for f in cdd]
            cdd_max = np.max(cdd_len)

            # Sort lengths into bins and return counts per bin
            cdd_out = np.histogram(cdd_len, lbins)[0]
            # binned = np.histogram(cdd_len, lbins)[0]
            # cdd_out = np.array([v if v != 0 else np.nan for v in binned])
            if maxper:
                cdd_out = np.hstack((cdd_out, cdd_max))

        return cdd_out

    if keepdims:
        CDD = np.apply_along_axis(cdd_calc, axis, data, thr, lbins=periods)
    else:
        CDD = cdd_calc(data, thr, lbins=periods)

    return CDD


def Rxx(data, thr=1.0, axis=0, normalize=False, keepdims=False):
    """
    Rxx mm, count of any time units (days, hours, etc) when precipitation ≥
    xx mm: Let RRij be the precipitation amount on time unit i in period j.
    Count the number of days where RRij ≥ xx mm.

    Parameters
    ----------
    data: array
        1D/2D data array, with time steps on the zeroth axis (axis=0).
    thr: float/int
        Threshold to be used; eg 10 for R10, 20 R20 etc. Default 1.0.
    axis: int
        Along which axis to calculate Rxx. Defaults to 0
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
    def rxx_calc(pdata, thr):
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
        RXX = np.apply_along_axis(rxx_calc, axis, data, thr=thr)
    else:
        RXX = rxx_calc(data, thr=thr)

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
    def rpxx_calc(pdata, pctl, thr):
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
        RRpx = np.apply_along_axis(rpxx_calc, axis, data, percentile, thr)
    else:
        RRpx = rpxx_calc(data, pctl=percentile, thr=thr)

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
    def rtxx_calc(pdata, thr):
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
        RRtx = np.apply_along_axis(rtxx_calc, axis, data, thr)
    else:
        RRtx = rtxx_calc(data, thr=thr)

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
