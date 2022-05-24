import numpy as np
from scipy.stats import sem
from scipy.stats import t


def ttest_1d(data1, data2, alpha):
    """
    Function for calculating the t-test for two independent samples
    """

    # calculate means
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)

    # calculate standard errors
    se1 = sem(data1)
    se2 = sem(data2)

    # standard error on the difference between the samples
    sed = np.sqrt(se1**2.0 + se2**2.0)

    # calculate the t statistic
    t_stat = (mean1 - mean2) / sed

    # degrees of freedom
    df = len(data1) + len(data2) - 2

    # calculate the critical value
    cv = t.ppf(1.0 - alpha, df)

    # calculate the p-value
    p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0

    return t_stat, df, cv, p
