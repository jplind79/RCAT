"""
Bootstrapping
-------------
Functions for bootstrap calculations.

Authors: Petter Lind
Created: Autumn 2016
Updates:
        May 2020
"""

def block_bootstr(data, block=5, nrep=500, nproc=1):
    """
    Calculate block bootstrap samples.

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
    Sample one-dimensional data with replacement.

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
    """Return samples from bootstrapping using multi-processing module"""
    bs = np.array([_get_bootsample(data[:, i], block=block)
                   for i in range(nx)])
    return bs
