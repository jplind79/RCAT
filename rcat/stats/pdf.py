"""
Probability distributions
-------------------------

Authors: Petter Lind
Created: Spring 2015
Updates:
        May 2020
"""
import numpy as np
from functools import reduce
import multiprocessing as mp
from .bootstrap import block_bootstr, _mproc_get_bootsamples


def freq_int_dist(data, keepdims=False, axis=0, bins=10, thr=None,
                  density=True, ci=False, bootstrap=False, nmc=500,
                  block=1, ci_level=95, nproc=1):
    """
    Calculate frequency - instensity distriutions.

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
