"""
Functions for various arithmetic calculations.

Created: Autumn 2016
Authors: Petter Lind & David Lindstedt
"""

import numpy as np

def run_mean(x, N, mode='valid'):
    """
    Calculate running mean

    Return running mean of data vector x where
    N is the window size.
    mode key word argument describes how the edges should be handled.
    See numpy.convolve for more information.
    """
    return np.convolve(x, np.ones((N,))/N, mode='valid')
