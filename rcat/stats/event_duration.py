"""
Event Duration Analysis (EDA) of Precipitation
----------------------------------------------

Author: Petter Lind
Created: Fall 2020
Updates:

"""

import numpy as np


def eda(data, keepdims=False, axis=0, thr=0.1, duration_bins=None,
        event_statistic='amount', statistic_bins=None, dry_events=False,
        dry_bins=None):
    """
    Calculate event duration statistics

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
    event_statistic: str
        The statistic to calculate for each event; choices are 'amount',
        'mean int' or 'max int'.
    duration_bins: list/array
        Defines the bin edges for event durations, including the rightmost
        edge, allowing for non-uniform bin widths.
    statistic_bins: list/array
        Defines the bin edges for event statistic (amount/mean/max),
        including the rightmost edge, allowing for non-uniform bin widths.
    thr: float
        Value of threshold to identify start/end of events. Default 0.1.
    dry_events: bool
        If set to True, duration of dry intervals will be calculated.
        'dry_bins' must then be provided.

    Returns
    -------
    eda_arr: array
        data array with frequency of event statistic (amount, mean, max) per
        duration bin.
    """

    def eda_calc(pdata, inthr, dur_bins, stat, stat_bins, calc_dry, dbins):
        # Flatten data to one dimension
        if isinstance(pdata, np.ma.MaskedArray):
            data1d = pdata.compressed()
        elif isinstance(pdata, (list, tuple)):
            data1d = np.array(pdata)
        else:
            data1d = pdata.ravel()

        if all(np.isnan(data1d)):
            print("All data missing/masked!")
            eda_arr = np.zeros((dur_bins.size-1, stat_bins.size-1))
            eda_arr[:] = np.nan
        else:
            # if any(np.isnan(data1d)):
            #     data1d = data1d[~np.isnan(data1d)]

            # When is data above threshold
            indata = data1d >= inthr

            # Make sure all events are well-bounded
            bounded = np.hstack(([0], indata, [0]))

            # Identify start and end of events
            diffs = np.diff(bounded)
            run_starts, = np.where(diffs > 0)
            run_ends, = np.where(diffs < 0)

            # Calculate durations
            durations = run_ends - run_starts

            if stat == 'amount':
                stat_data = np.array([np.sum(data1d[s:e])
                                      for s, e in zip(run_starts, run_ends)])
            elif stat == 'mean int':
                stat_data = np.array([np.mean(data1d[s:e])
                                      for s, e in zip(run_starts, run_ends)])
            elif stat == 'max int':
                stat_data = np.array([np.max(data1d[s:e])
                                      for s, e in zip(run_starts, run_ends)])
            stat_dict = {bint: stat_data[durations == bint]
                         if bint in durations else np.nan
                         for bint in dur_bins}
            eda_arr = np.array(
                [np.histogram(arr, bins=stat_bins)[0]
                 for d, arr in stat_dict.items()]).swapaxes(0, 1)

            if calc_dry:
                dry = ~indata
                bounded_dry = np.hstack(([0], dry, [0]))
                diffs_dry = np.diff(bounded_dry)
                run_starts_dry, = np.where(diffs_dry > 0)
                run_ends_dry, = np.where(diffs_dry < 0)
                dry_durations = run_ends_dry - run_starts_dry
                dry_data = np.histogram(dry_durations, bins=dbins)[0]

                eda_arr = np.c_[dry_data, eda_arr]

        return eda_arr

    # Bins
    msg = "\t\n'N.B. 'duration_bins' must be provided!"
    assert duration_bins is not None, msg
    msg = "\t\n'N.B. 'statistic_bins' must be provided!"
    assert statistic_bins is not None, msg
    if dry_events:
        msg = "\t\n'dry_bins' must be provided if 'dry_events' is True!"
        assert dry_bins is not None, msg

    if keepdims:
        eda_comp = np.apply_along_axis(
            eda_calc, axis=axis, arr=data, inthr=thr, dur_bins=duration_bins,
            stat=event_statistic, stat_bins=statistic_bins,
            calc_dry=dry_events, dbins=dry_bins)
    else:
        eda_comp = eda_calc(data, inthr=thr, dur_bins=duration_bins,
                            stat=event_statistic, stat_bins=statistic_bins,
                            calc_dry=dry_events, dbins=dry_bins)

    return eda_comp
