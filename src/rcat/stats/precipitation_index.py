import numpy as np


def ranked_cumsum(data, axis=0, keepdims=False):
    """
    Rank the data and calculate the cumulative sum, starting from the highest
    values.
    """
    # Rank data
    ranked = np.flip(np.sort(data, axis=axis), axis=axis)

    # Normalize
    norm = np.divide(ranked, ranked.sum(axis=axis, keepdims=True))
    csum = np.cumsum(norm, axis=axis)

    return csum


def precip_amount_survival_fraction(data, pctls, keepdims=False):
    """
    Calculate the frequency distribution, normalize by total precipitation, and
    sum the fractions. It answers the question 'what fraction of total
    precipitation occurs beyond the top p percentile of days in a period?',
    where p is any percentile of interest.
    """
    pctls = np.percentile(data, pctls)
    return [data[data >= p].sum() for p in pctls]/np.nansum(data)
