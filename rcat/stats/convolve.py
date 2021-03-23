#!/usr/bin/env python
#
#
# Functions for convolution of arrays ##
#
# By: Petter Lind
#     2014-04-11
#
#
#

# Import necessary modules
import numpy as np
from scipy import signal

"""
This module includes functions to perform convolution, for example
image smoothing, partly using scipy's convolution routines.
"""


def kernel_gen(n, ktype='square', kfun='mean'):
    """
    Function to create a kernel, i.e. a moving window (box or disk) with
    side/radius equal to 'n'.

    Parameters
    ----------
    n: int
        Side/radius of square/disk of smoothening window.
    ktype: string
        The type of box; 'square' (default) or 'disk'.
    kfun: string
        The function 'kfun' applied to each sub-sample within the moving
        window. Either 'mean' (default) or 'sum'.
    """

    if (ktype == 'square'):
        if (kfun == 'mean'):
            kernel = np.ones(shape=(n, n)) * (1./n**2)
        if (kfun == 'sum'):
            kernel = np.ones(shape=(n, n))
    elif (ktype == 'disk'):
        if (kfun == 'mean'):
            y, x = np.ogrid[-n: n+1, -n: n+1]
            disk = x**2+y**2 <= n**2
            disk = np.where(disk, 1.0, 0.0)
            kernel = disk/n**2

    return kernel


def filtering(data, wgts, mode='valid', dim=1, axis=None, fft=False,
              fftn=np.fft.fftn, ifftn=np.fft.ifftn):
    """
    1D and 2D filtering procedures.

    Filters input data, both 1D and 2D, with user defined weights. Set fft to
    True for fast fourier transform to speed things up when data is large.

    Parameters
    ----------
    data: array
        Data to be filtered.
    wgts: array/list
        The weights (kernel) to be used in the filtering.
    mode: str
        String indicating the size of output (see
        https://docs.scipy.org/doc/scipy/reference/signal.html)
    dim: int
        If 1 one-dimensional filtering is performed and if 'axis' is also set,
        1D-filtering is applied along this axis. If dim=2 two-dimensional
        filtering is applied.
    fft: boolean
        Set True to use fast fourier transform in the 2D filtering.

    Returns
    -------
    data_conv: array
        Convoluted data
    """
    if isinstance(data, np.ma.MaskedArray):
        mask = data.mask    # Set masked values to zero
        indata = np.ma.filled(data, 0)
    else:
        indata = np.copy(data)

    if dim == 1:
        # One-dimensional filtering
        if axis is None:
            data_conv = signal.convolve(indata, wgts, mode=mode)
        else:
            data_conv = np.apply_along_axis(signal.convolve, axis, indata,
                                            wgts, mode=mode)
    elif dim == 2:
        # Two-dimensional filtering
        if fft:
            data_conv = convolve_fft(indata, wgts, fftn=fftn, ifftn=ifftn)
        else:
            data_conv = signal.convolve2d(indata, wgts, mode=mode,
                                          boundary='symm')

    # Set values in mask to NaN (if masked array)
    if isinstance(data, np.ma.MaskedArray):
        data_conv[mask] = np.nan

    return data_conv


def lanczos_filter(window, cutoff, cutoff_2=None, ftype='lowpass'):
    """
    Calculate weights for a low pass Lanczos filter.

    Parameters
    ----------
    window: int
        The length of the filter window.

    cutoff: float
        The cutoff frequency in inverse time steps.
    cutoff_2: float
        The second cutoff frequency in inverse time steps. Only used if ftype
        is 'bandpass'
    ftype: str
        The type of cutoff filtering: 'lowpass', 'highpass' or 'bandpass'.

    Returns
    -------
    wgts: vector
        Array with calculated weights.

    """
    def _low_pass_filter(win, cut):
        order = ((win - 1) // 2) + 1
        nwts = 2 * order + 1
        w = np.zeros([nwts])
        n = nwts // 2
        w[n] = 2 * cut
        k = np.arange(1., n)
        sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
        firstfactor = np.sin(2. * np.pi * cut * k) / (np.pi * k)
        w[n-1:0:-1] = firstfactor * sigma
        w[n+1:-1] = firstfactor * sigma
        return w[1:-1]

    if ftype == 'lowpass':
        wgts_out = _low_pass_filter(window, cutoff)
    elif ftype == 'highpass':
        wgts_out = (-1) * _low_pass_filter(window, cutoff)
    elif ftype == 'bandpass':
        assert cutoff_2 is not None, "cutoff_2 must be set for ftype bandpass"
        wgts1 = _low_pass_filter(window, cutoff)
        wgts2 = _low_pass_filter(window, cutoff_2)
        wgts_out = wgts2 - wgts1

    return wgts_out


def fft_prep(array, kernel, fill_value, boundary='fill', psf_pad=False,
             fft_pad=True):
    """
    Prepare data array and kernel for fft computation.

    Parameters
    ----------
    boundary : {'fill', 'wrap'}, optional
        A flag indicating how to handle boundaries:

            * 'fill': set values outside the array boundary to fill_value
              (default)
            * 'wrap': periodic boundary
    fft_pad : bool, optional
        Default on.  Zero-pad image to the nearest 2^n
    psf_pad : bool, optional
        Default off.  Zero-pad image to be at least the sum of the image sizes
        (in order to avoid edge-wrapping when smoothing)
    """

    arrayshape = array.shape
    kernshape = kernel.shape

    # mask catching - masks must be turned into NaNs for use later
    if np.ma.is_masked(array):
        mask = array.mask
        array = np.array(array)
        array[mask] = np.nan
    if np.ma.is_masked(kernel):
        mask = kernel.mask
        kernel = np.array(kernel)
        kernel[mask] = np.nan

    # NaN and inf catching
    nanmaskarray = np.isnan(array) + np.isinf(array)
    array[nanmaskarray] = 0
    nanmaskkernel = np.isnan(kernel) + np.isinf(kernel)
    kernel[nanmaskkernel] = 0

    if boundary is None:
        psf_pad = True
    elif boundary == 'fill':
        # create a boundary region at least as large as the kernel
        psf_pad = True
    elif boundary == 'wrap':
        psf_pad = False
        fft_pad = False
        fill_value = 0  # force zero; it should not be used
    elif boundary == 'extend':
        raise NotImplementedError("The 'extend' option is not implemented "
                                  "for fft-based convolution")

    # find ideal size (power of 2) for fft.
    # Can add shapes because they are tuples
    if fft_pad:         # default=True
        if psf_pad:     # default=False
            # add the dimensions and then take the max (bigger)
            fsize = 2 ** np.ceil(np.log2(
                np.max(np.array(arrayshape) + np.array(kernshape))))
        else:
            # add the shape lists (max of a list of length 4) (smaller)
            # also makes the shapes square
            fsize = 2 ** np.ceil(np.log2(np.max(arrayshape + kernshape)))
        newshape = np.array([fsize for ii in range(array.ndim)], dtype=int)
    else:
        if psf_pad:
            # just add the biggest dimensions
            newshape = np.array(arrayshape) + np.array(kernshape)
        else:
            newshape = np.array([np.max([imsh, kernsh])
                                for imsh, kernsh
                                in zip(arrayshape, kernshape)])

    # separate each dimension by the padding size...  this is to determine the
    # appropriate slice size to get back to the input dimensions
    arrayslices = []
    kernslices = []
    for ii, (newdimsize, arraydimsize, kerndimsize)\
            in enumerate(zip(newshape, arrayshape, kernshape)):
        center = newdimsize - (newdimsize + 1) // 2
        arrayslices += [slice(center - arraydimsize // 2,
                              center + (arraydimsize + 1) // 2)]
        kernslices += [slice(center - kerndimsize // 2,
                             center + (kerndimsize + 1) // 2)]

    if not np.all(newshape == arrayshape):
        bigarray = np.ones(newshape, dtype=np.complex) * fill_value
        bigarray[arrayslices] = array
    else:
        bigarray = array

    if not np.all(newshape == kernshape):
        bigkernel = np.zeros(newshape, dtype=np.complex)
        bigkernel[kernslices] = kernel
    else:
        bigkernel = kernel

    return bigarray, bigkernel, arrayslices,\
        kernslices, newshape, nanmaskarray, nanmaskkernel


def convolve_fft(array, kernel, boundary='fill', fill_value=0, crop=True,
                 return_fft=False, fft_pad=True, psf_pad=False,
                 interpolate_nan=False, quiet=False, ignore_edge_zeros=False,
                 min_wt=0.0, normalize_kernel=False, allow_huge=True,
                 fftn=np.fft.fftn, ifftn=np.fft.ifftn):
    """
    Convolve an ndarray with an nd-kernel.  Returns a convolved image with
    shape = array.shape.  Assumes kernel is centered.

    `convolve_fft` differs from `scipy.signal.fftconvolve` in a few ways:

    * It can treat ``NaN`` values as zeros or interpolate over them.
    * ``inf`` values are treated as ``NaN``
    * (optionally) It pads to the nearest 2^n size to improve FFT speed.
    * Its only valid ``mode`` is 'same' (i.e. the same shape array is returned)
    * It lets you use your own fft, e.g.,
      `pyFFTW <http://pypi.python.org/pypi/pyFFTW>` or
      `pyFFTW3 <http://pypi.python.org/pypi/PyFFTW3/0.2.1>`,
      which can lead to performance improvements, depending on your system
      configuration. pyFFTW3 is threaded, and therefore may yield significant
      performance benefits on multi-core machines at the cost of greater
      memory requirements. Specify the ``fftn`` and ``ifftn`` keywords to
      override the default, which is `numpy.fft.fft` and `numpy.fft.ifft`.

    Parameters
    ----------
    array : `numpy.ndarray`
        Array to be convolved with ``kernel``
    kernel : `numpy.ndarray`
        Will be normalized if ``normalize_kernel`` is set.  Assumed to be
        centered (i.e., shifts may result if your kernel is asymmetric)
    boundary : {'fill', 'wrap'}, optional
        A flag indicating how to handle boundaries:
        * 'fill': set values outside the array boundary to fill_value (default)
        * 'wrap': periodic boundary

    interpolate_nan : bool, optional
        The convolution will be re-weighted assuming ``NaN`` values are meant
        to be ignored, not treated as zero.  If this is off, all ``NaN`` values
        will be treated as zero.
    ignore_edge_zeros : bool, optional
        Ignore the zero-pad-created zeros.
        This will effectively decrease the kernel area on the edges but will
        not re-normalize the kernel. This parameter may result in
        'edge-brightening' effects if you're using a normalized kernel
    min_wt : float, optional
        If ignoring ``NaN`` / zeros, force all grid points with a weight less
        than this value to ``NaN`` (the weight of a grid point with *no*
        ignored neighbors is 1.0). If ``min_wt`` is zero, then all zero-weight
        points will be set to zero instead of ``NaN`` (which they would be
        otherwise, because 1/0 = nan). See the examples below
    normalize_kernel : function or boolean, optional
        If specified, this is the function to divide kernel by to normalize it.
        e.g., ``normalize_kernel=np.sum`` means that kernel will be modified
        to be:
        ``kernel = kernel / np.sum(kernel)``.  If True, defaults to
        ``normalize_kernel = np.sum``.

    Other Parameters
    ----------------
    fft_pad : bool, optional
        Default on.  Zero-pad image to the nearest 2^n
    psf_pad : bool, optional
        Default off.  Zero-pad image to be at least the sum of the image sizes
        (in order to avoid edge-wrapping when smoothing)
    crop : bool, optional
        Default on.  Return an image of the size of the largest input image.
        If the images are asymmetric in opposite directions, will return the
        largest image in both directions.
        For example, if an input image has shape [100,3] but a kernel with
        shape [6,6] is used, the output will be [100,6].
    return_fft : bool, optional
        Return the fft(image)*fft(kernel) instead of the convolution (which is
        ifft(fft(image)*fft(kernel))).  Useful for making PSDs.
    fftn, ifftn : functions, optional
        The fft and inverse fft functions.  Can be overridden to use your own
        ffts, e.g. an fftw3 wrapper or scipy's fftn, e.g.
        ``fftn=scipy.fftpack.fftn``
    complex_dtype : np.complex, optional
        Which complex dtype to use.  `numpy` has a range of options, from 64 to
        256.
    quiet : bool, optional
        Silence warning message about NaN interpolation
    allow_huge : bool, optional
        Allow huge arrays in the FFT?  If False, will raise an exception if the
        array or kernel size is >1 GB

    Raises
    ------
    ValueError:
        If the array is bigger than 1 GB after padding, will raise this
        exception unless allow_huge is True

    See Also
    --------
    convolve : Convolve is a non-fft version of this code.  It is more
               memory efficient and for small kernels can be faster.

    Returns
    -------
    default : ndarray
        **array** convolved with ``kernel``.
        If ``return_fft`` is set, returns fft(**array**) * fft(``kernel``).
        If crop is not set, returns the image, but with the fft-padded size
        instead of the input size
    """

    array = np.asarray(array, dtype=np.complex)
    kernel = np.asarray(kernel, dtype=np.complex)

    # Check that the number of dimensions is compatible
    if array.ndim != kernel.ndim:
        raise ValueError("Image and kernel must have same number of "
                         "dimensions")

    arrayshape = array.shape

    if not allow_huge:
        array_size_B = (np.product(arrayshape, dtype=np.int64) *
                        np.dtype(np.complex).itemsize)
        if array_size_B > 1024**3:
            raise ValueError("Size Error: Use allow_huge=True to override\
                             this exception.")

    bigarray, bigkernel, arrayslices, kernslices,\
        newshape, nanmaskarray, nanmaskkernel =\
        fft_prep(array, kernel, fill_value, boundary='fill',
                 psf_pad=psf_pad, fft_pad=fft_pad)

    arrayfft = fftn(bigarray)
    # need to shift the kernel so that, e.g., [0,0,1,0] -> [1,0,0,0] = unity
    kernfft = fftn(np.fft.ifftshift(bigkernel))
    fftmult = arrayfft * kernfft

    if (interpolate_nan or ignore_edge_zeros):
        if ignore_edge_zeros:
            bigimwt = np.zeros(newshape, dtype=np.complex)
        else:
            bigimwt = np.ones(newshape, dtype=np.complex)
        bigimwt[arrayslices] = 1.0 - nanmaskarray * interpolate_nan
        wtfft = fftn(bigimwt)
        # I think this one HAS to be normalized (i.e., the weights can't be
        # computed with a non-normalized kernel)
        wtfftmult = wtfft * kernfft / kernel.sum()
        wtsm = ifftn(wtfftmult)

        # need to re-zero weights outside of the image (if it is padded, we
        # still don't weight those regions)
        bigimwt[arrayslices] = wtsm.real[arrayslices]
        # curiously, at the floating-point limit, can get slightly negative
        # numbers they break the min_wt=0 "flag" and must therefore be removed
        bigimwt[bigimwt < 0] = 0
    else:
        bigimwt = 1

    if np.isnan(fftmult).any():
        # this check should be unnecessary; call it an insanity check
        raise ValueError("Encountered NaNs in convolve.  This is disallowed.")

    # restore NaNs in original image (they were modified inplace earlier)
    # We don't have to worry about masked arrays - if input was masked, it was
    # copied
    array[nanmaskarray] = np.nan
    kernel[nanmaskkernel] = np.nan

    if return_fft:
        return fftmult

    if interpolate_nan or ignore_edge_zeros:
        rifft = (ifftn(fftmult)) / bigimwt
        if not np.isscalar(bigimwt):
            rifft[bigimwt < min_wt] = np.nan
            if min_wt == 0.0:
                rifft[bigimwt == 0.0] = 0.0
    else:
        rifft = (ifftn(fftmult))

    if crop:
        result = rifft[arrayslices].real
        return result
    else:
        return rifft.real
