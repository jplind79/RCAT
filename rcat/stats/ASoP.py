"""
ASoP - Analyzing Scales of Precipitation
----------------------------------------

Reference: Klingaman et al (2017)
https://www.geosci-model-dev.net/10/57/2017/

Authors: Petter Lind
Created: Spring 2019
Updates:
        May 2020
"""

import numpy as np


def asop(data, keepdims=False, axis=0, bins=None, thr=None, return_bins=False):
    """
    Calculate ASoP parameters.

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


def bins_calc(n, bintype='Klingaman'):
    """
    Calculates bins with edges according to Eq. 1 in Klingaman et al. (2017);
    https://www.geosci-model-dev.net/10/57/2017/

    Parameters
    ----------
    n: array/list
        1D array or list with bin numbers
    bintype: str
        The type of bins to be calculated; 'Klingaman' (see reference) or
        'exp' for exponential bins.

    Returns
    -------
    bn: array
        1D array of bin edges
    """
    if bintype == 'Klingaman':
        # bn = np.e**(np.log(0.005)+(n*(np.log(120)-np.log(0.005))**2/59)**(1/2))
        bn = 0.005*np.exp(np.sqrt(1.724*n))
    elif bintype == 'exp':
        bn = 0.02*np.exp(0.12*n)

    return bn
