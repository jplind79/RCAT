"""
Atmospheric Function Module
---------------------------
Functions for calculations of various physical properties

Created: Autumn 2017
Authors: Petter Lind & David Lindstedt
"""

import numpy as np

# Constants/parameters
L = 2.501e6     # latent heat of vaporization
R = 287.04      # gas constant air
Rv = 461.5      # gas constant vapor
eps = R/Rv
cp = 1005.
cv = 718.


def rh2sh(rh, T, P=1013.25):
    """
    Convert relative humidity to specific humidity
    Code from:
    https://github.com/PecanProject/pecan/blob/master/modules/data.atmosphere/R/metutils.R
    Equation for sh from standard literature, e.g. K. Emanuel (1994; eq. 4.1.4)
    Reference:
    * Emanuel, K. A. (1994):  Atmospheric Convection.  New York, NY:
      Oxford University Press, 580 pp.


    Parameters
    ----------
    rh, float/array of floats
        Relative humidity in proportion, not percent
    T, float/array of floats
        Absolute temperature in Kelvin
    P, float/array of floats
        Pressure in hPa (mb)

    Returns
    -------
    sh,
        Specific humidity in kg/kg

    """
    es = calc_es(T)
    e = rh * es
    sh = (eps * e) / (P - e*(1 - eps))

    return(sh)


def sh2rh(sh, T, P=1013.25):
    """
    Convert relative humidity to specific humidity
    Code from:
    https://github.com/PecanProject/pecan/blob/master/modules/data.atmosphere/R/metutils.R

    Parameters
    ----------
    sh, float/array of floats
        Specific humidity in kg/kg
    T, float/array of floats
        Absolute temperature in Kelvin
    P, float/array of floats
        Pressure in hPa (mb)

    Returns
    -------
    rh,
        Relative humidity in fraction (not percent)

    """

    es = calc_es(T)
    e = calc_e_from_sh(sh, P)
    rh = e / es
    if isinstance(rh, (np.int, np.float)):
        rh = 1 if rh > 1 else rh
        rh = 0 if rh < 0 else rh
    else:
        rh[rh > 1] = 1
        rh[rh < 0] = 0

    return(rh)


def td2sh(Td, P):
    """
    Convert dew point temperature to specific humidity
    https://github.com/PecanProject/pecan/blob/master/modules/data.atmosphere/R/metutils.R

    Parameters
    ----------
    Td, float/array of floats
        Absolute dew point temperature in Kelvin
    P, float/array of floats
        Pressure in mb

    Returns
    -------
    p, float/array of floats
        Specific humidity in g/kg
    """

    Td_C = Td - 273.15
    e = 6.112 * np.exp((17.67 * Td_C) / (Td_C + 243.5))
    q = (eps * e)/(P - (0.378 * e))

    return(q)


def sh2td(sh, p):
    """
    Returns dew point temperature (K) at mixing ratio sh (kg/kg) and
    pressure p (Pa). Insert Td in 2.17 in Rogers&Yau and solve for Td
    """

    ep = calc_e_from_sh(sh, p)
    return 243.5 * np.log(ep/611.2)/(17.67-np.log(ep/611.2)) + 273.15


def vpd(rh, T):
    """
    Calculate VPD - vapor pressure deficit from relative humidity and
    temperature.
    Code from:
    https://github.com/PecanProject/pecan/blob/master/modules/data.atmosphere/R/metutils.R

    Parameters
    ----------
    rh, float/array of floats
        Relative humidity in fraction (not percent)
    T, float/array of floats
        Absolute temperature in Kelvin

    Returns
    -------
    vpd,
        Vapor pressure deficit (in hPa/mb)

    """

    e_sat = calc_es(T)
    vpd = (((100 - rh)/100) * e_sat)

    return vpd


def calc_es(T):
    """
    Returns saturation vapor pressure in hPa at temperature T (Kelvin)
    Formula 2.17 in Rogers&Yau
    """

    Tc = T - 273.15  # Temperature in degrees Celcius

    return 6.112 * np.exp(17.67 * Tc / (Tc + 243.5))


def calc_e_from_w(w, P=1013.25):
    """
    Vapor pressure over liquid surface can be calculated from a variety of
    different combinations of state variables.
    Here, vapor pressure is calculated from mixing ratio w (kg/kg)
    and pressure P. Formula is given in standard literature; e.g. eq. 2.18 in
    Rogers&Yau (Cloud physics) and eq. 4.1.2 in Emanuel (1994).
    Reference:
    * Emanuel, K. A. (1994):  Atmospheric Convection.  New York, NY:
      Oxford University Press, 580 pp.
    """

    return w * P / (eps + w)


def calc_e_from_sh(sh, P=1013.25):
    """
    Vapor pressure over liquid surface can be calculated from a variety of
    different combinations of state variables.
    Here, vapor pressure is calculated from specific humidity sh and pressure
    P. Formula is given in standard literature; e.g. eq. 2.19 in Rogers & Yau
    (Cloud physics) and eq. 4.1.4 in Emanuel (1994).
    Reference:
    * Emanuel, K. A. (1994):  Atmospheric Convection.  New York, NY:
      Oxford University Press, 580 pp.
    """

    return sh * P / (sh*(1 - eps) + eps)


def td(e):
    """
    Returns dew point temperature (C) at vapor pressure e (Pa)
    Insert Td in 2.17 in Rogers&Yau and solve for Td
    """

    return 243.5 * np.log(e/611.2)/(17.67-np.log(e/611.2))


def wind2uv(ws, wd, dir_unit='rad'):
    """
    Converts wind speed and direction to u and v components.

    Parameters
    ----------
    ws, float/array of floats
        Wind speed data
    wd, float/array of floats
        Wind direction data
    dir_deg, string
        Units of the wind direction data; 'rad' (default) or 'deg'.

    Returns
    -------
    (u, v), floats/arrays of floats
        u and v wind components

    """

    if dir_unit == 'deg':
        RperD = 4.0*np.arctan(1)/180
        rad = wd*RperD
    else:
        rad = wd

    u = -ws*np.sin(rad)
    v = -ws*np.cos(rad)

    return u, v


def uv2wind(u, v):
    """
    Converts u and v components to wind speed and direction.

    Parameters
    ----------
    u, float/array of floats
        east/west wind component
    v, float/array of floats
        north/south wind component

    Returns
    -------
    (ws, wd), floats/arrays of floats
        wind speed and wind direction respectively.

    """

    from math import pi

    DperR = 180/pi

    ws = np.sqrt(u**2 + v**2)
    wd = 270 - (np.arctan2(v, u) * DperR)

    return ws, wd


def calc_vaisala(ddict, model, lower_plevel, upper_plevel):
    """
    Calculate Brunt-Vaisala frequency in layer bounded by two pressure levels.
    Pressure must be given in hPa, temperature (T) in Kelvin and specific
    humidity (q) in kg/kg.

    Parameters
    ----------
    ddict: dictionary
        Data dictionary
    model: str
        model data to use
    lower_plevel: float
        lower pressure surface
    upper_plevel: float
        upper pressure surface

    Returns
    -------
    N2: array with floats
        The squared Brunt-Vaisala frequency
    """

    g = 9.81
    R = 287
    cp = 1005

    # Virtual theta at lower plevel
    p_lwr = lower_plevel
    q_lwr = ddict['hus{}'.format(p_lwr)][model]
    T_lwr = ddict['ta{}'.format(p_lwr)][model]
    theta_lwr = T_lwr * (1000/p_lwr) ** (R/cp)
    theta_v_lwr = (1 + 0.61*q_lwr) * theta_lwr

    # Virtual theta at upper plevel
    p_upr = lower_plevel
    q_upr = ddict['hus{}'.format(p_upr)][model]
    T_upr = ddict['ta{}'.format(p_upr)][model]
    theta_upr = T_upr * (1000/p_upr) ** (R/cp)
    theta_v_upr = (1 + 0.61*q_upr) * theta_upr

    dz = (ddict['zg{}'.format(p_upr)][model] -
          ddict['zg{}'.format(p_lwr)][model])/g

    N2 = g * (np.log(theta_v_upr) - np.log(theta_v_lwr)) / dz

    return N2
