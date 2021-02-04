#
#       Observation Meta Data File
#

"""
This file contains information regarding observations used in RCAT.
Paths to and file names (prefixes) of observations as well as some meta data
should be added here.

N.B.!
The obs data filenames must include year and months that are covered by the
specific file.
The file pattern defined for each data set should have 'YYYYMM' at the
locations in the file name where year and month occur.

Ex.
Full data filename: tas_day_ECMWF-ERA5_rean_r1i1p1_19970101-19971231.nc
File pattern to set: tas_day_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc
"""

import numpy as np
import os
import glob


def obs_data():
    """
    Dictionary with variables as top keys and available observations
    directly below. For each observation data set, path and file pattern must
    be defined.
    """

    meta_dict = {

    # ------------------------------------------------------------------------
    # 2m temperature
    'tas': {
        'EOBS': {
            'path': '/home/rossby/imports/obs/EOBS/EOBS17/EUR-22/input/day',
            'file pattern': 'tas_EUR-22_EOBS17_obs_r1i1p1_ECAD_v1_day_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': '/nobackup/rossby20/sm_petli/data/grids/grid_EOBS_022_latlon' # noqa
        },
        'EOBS19': {
            'path': '/home/rossby/imports/obs/EOBS/EOBS19/EUR-10/input/day',
            'file pattern': 'tas_EUR-10_EOBS19e_obs_r1i1p1_ECAD_v1_day_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': None
        },
        'EOBS20': {
            'path': '/home/rossby/imports/obs/EOBS/EOBS20/EUR-10/input/day',
            'file pattern': 'tas_EUR-10_EOBS20e_obs_r1i1p1_ECAD_v1_day_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': None
        },
        'ERA5': {
            'path': '/nobackup/rossby20/sm_petli/data/ERA5/VALIDATION/EUR/day',
            'file pattern': 'tas_day_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': '/nobackup/rossby20/sm_petli/data/grids/grid_ERA5_EUR_latlon' # noqa
        },
        'NGCD': {
            'path': '/nobackup/rossby20/sm_petli/data/NGCD',
            'file pattern': 'tas_NGCD_type2_YYYYMMDD-YYYYMMDD.nc', # noqa
            'grid': None
        },
    },

    'tasmax': {
        'EOBS': {
            'path': '/home/rossby/imports/obs/EOBS/EOBS17/EUR-22/input/day',
            'file pattern': 'tasmax_EUR-22_EOBS17_obs_r1i1p1_ECAD_v1_day_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': '/nobackup/rossby20/sm_petli/data/grids/grid_EOBS_022_latlon' # noqa
        },
        'EOBS19': {
            'path': '/home/rossby/imports/obs/EOBS/EOBS19/EUR-10/input/day',
            'file pattern': 'tasmax_EUR-10_EOBS19e_obs_r1i1p1_ECAD_v1_day_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': None
        },
        'EOBS20': {
            'path': '/home/rossby/imports/obs/EOBS/EOBS20/EUR-10/input/day',
            'file pattern': 'tasmax_EUR-10_EOBS20e_obs_r1i1p1_ECAD_v1_day_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': None
        },
        'ERA5': {
            'path': '/nobackup/rossby20/sm_petli/data/ERA5/VALIDATION/EUR/day',
            'file pattern': 'tasmax_day_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': '/nobackup/rossby20/sm_petli/data/grids/grid_ERA5_EUR_latlon' # noqa
        },
        'NGCD': {
            'path': '/nobackup/rossby20/sm_petli/data/NGCD',
            'file pattern': 'tasmax_NGCD_type2_YYYYMMDD-YYYYMMDD.nc', # noqa
            'grid': None
        },
    },

    'tasmin': {
        'EOBS': {
            'path': '/home/rossby/imports/obs/EOBS/EOBS17/EUR-22/input/day',
            'file pattern': 'tasmin_EUR-22_EOBS17_obs_r1i1p1_ECAD_v1_day_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': '/nobackup/rossby20/sm_petli/data/grids/grid_EOBS_022_latlon' # noqa
        },
        'EOBS19': {
            'path': '/home/rossby/imports/obs/EOBS/EOBS19/EUR-10/input/day',
            'file pattern': 'tasmin_EUR-10_EOBS19e_obs_r1i1p1_ECAD_v1_day_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': None
        },
        'EOBS20': {
            'path': '/home/rossby/imports/obs/EOBS/EOBS20/EUR-10/input/day',
            'file pattern': 'tasmin_EUR-10_EOBS20e_obs_r1i1p1_ECAD_v1_day_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': None
        },
        'ERA5': {
            'path': '/nobackup/rossby20/sm_petli/data/ERA5/VALIDATION/EUR/day',
            'file pattern': 'tasmin_day_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': '/nobackup/rossby20/sm_petli/data/grids/grid_ERA5_EUR_latlon' # noqa
        },
        'NGCD': {
            'path': '/nobackup/rossby20/sm_petli/data/NGCD',
            'file pattern': 'tasmin_NGCD_type2_YYYYMMDD-YYYYMMDD.nc', # noqa
            'grid': None
        },
    },

    # ------------------------------------------------------------------------
    # Precipitation
    'pr': {
        'EOBS': {
            'path': '/home/rossby/imports/obs/EOBS/EOBS17/EUR-22/input/day',
            'file pattern': 'pr_EUR-22_EOBS17_obs_r1i1p1_ECAD_v1_day_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': '/nobackup/rossby20/sm_petli/data/grids/grid_EOBS_022_latlon' # noqa
        },
        'EOBS19': {
            'path': '/home/rossby/imports/obs/EOBS/EOBS19/EUR-10/input/day',
            'file pattern': 'pr_EUR-10_EOBS19e_obs_r1i1p1_ECAD_v1_day_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': None
        },
        'EOBS20': {
            'path': '/home/rossby/imports/obs/EOBS/EOBS20/EUR-10/input/day',
            'file pattern': 'pr_EUR-10_EOBS20e_obs_r1i1p1_ECAD_v1_day_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': None
        },
        'ERA5': {
            'path': '/nobackup/rossby20/sm_petli/data/ERA5/VALIDATION/EUR/1h',
            'file pattern': 'pr_1H_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': '/nobackup/rossby20/sm_petli/data/grids/grid_ERA5_EUR_latlon' # noqa
        },
        'ERAI': {
            'path': '/nobackup/rossby20/sm_petli/data/ERAI/VALIDATION/EUR/day',
            'file pattern': 'pr_day_ECMWF-ERAINT_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': None
        },
        'NGCD': {
            'path': '/nobackup/rossby20/sm_petli/data/NGCD',
            'file pattern': 'pr_NGCD_type2_YYYYMMDD-YYYYMMDD.nc', # noqa
            'grid': None
        },
        'HIPRADv2.0': {
            'path': '/nobackup/rossby20/sm_petli/data/HIPRAD/1h_old_code/masked', # noqa
            # 'path': '/nobackup/rossby20/sm_petli/data/HIPRAD/1h_old_code', # noqa
            'file pattern': 'pr_HIPRAD2_1H_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': None
        },
        'HIPRADv2.1': {
            'path': '/nobackup/rossby20/sm_petli/data/HIPRAD/1h/masked',
            'file pattern': 'pr_HIPRAD2_1H_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': None
        },
        'HIPRADv3': {
            'path': '/nobackup/rossby20/sm_petli/data/HIPRAD/Nordic_v3/1h',
            'file pattern': 'pr_HIPRAD3_Nordic_1H_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': None
        },
        'SENORGE': {
            'path': '/nobackup/rossby20/sm_petli/data/seNorge_pr/orig',
            'file pattern': 'pr_seNorge2_PREC1h_grid_YYYYMM.nc', # noqa
            'grid': None
        },
        'Klimagrid': {
            'path': '/nobackup/rossby20/sm_petli/data/klimagrid/1h',
            'file pattern': 'pr_Klimagrid_Denmark_1h_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': None
        },
        'Spain02': {
            'path': '/nobackup/rossby20/sm_petli/data/Spain02/day',
            'file pattern': 'pr_Spain02_v2_day_YYYYMM01_YYYYMM31.nc', # noqa
            'grid': None
        },
        'SAFRAN': {
            'path': '/nobackup/rossby20/sm_petli/data/SAFRAN/day',
            'file pattern': 'pr_Meteo-France_SAFRAN_day_YYYYMM01_YYYYMM31.nc', # noqa
            'grid': None
        },
        'EURO4M-APGD': {
            'path': '/nobackup/rossby20/sm_petli/data/EURO4M/APGD/day',
            'file pattern': 'pr_EURO4M-APGD_day_YYYYMM01_YYYYMM31.nc', # noqa
            'grid': None
        },
    },

    # ------------------------------------------------------------------------
    # Convective Precipitation
    'cpr': {
        'ERAI': {
            'path': '/nobackup/rossby20/sm_petli/data/ERAI/VALIDATION/EUR/day',
            'file pattern': 'cpr_day_ECMWF-ERAINT_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': None
        },
    },

    # ------------------------------------------------------------------------
    # CAPE
    'cape': {
        'ERA5': {
            'path': '/nobackup/rossby20/sm_petli/data/ERA5/VALIDATION/EUR/day',
            'file pattern': 'cape_day_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': None
        },
    },

    # ------------------------------------------------------------------------
    # MSLP
    'psl': {
        'EOBS': {
            'path': '/home/rossby/imports/obs/EOBS/EOBS17/EUR-22/input/day',
            'file pattern': 'psl_EUR-22_EOBS17_obs_r1i1p1_ECAD_v1_day_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': None # noqa
        },
        'EOBS19': {
            'path': '/home/rossby/imports/obs/EOBS/EOBS19/EUR-10/input/day',
            'file pattern': 'psl_EUR-10_EOBS19e_obs_r1i1p1_ECAD_v1_day_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': None
        },
        'EOBS20': {
            'path': '/home/rossby/imports/obs/EOBS/EOBS20/EUR-10/input/day',
            'file pattern': 'psl_EUR-10_EOBS20e_obs_r1i1p1_ECAD_v1_day_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': None
        },
        'ERA5': {
            # 'path': '/nobackup/rossby20/sm_petli/data/ERA5/VALIDATION/EUR/day',
            'path': '/home/rossby/imports/obs/ECMWF/ERA5/input/day',
            'file pattern': 'psl_day_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': None # noqa
        },
    },

    # ------------------------------------------------------------------------
    # Long-wave down-welling radiation
    'rlds': {
        'ERA5': {
            'path': '/nobackup/rossby20/sm_petli/data/ERA5/VALIDATION/EUR/1h',
            'file pattern': 'rlds_1H_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': '/nobackup/rossby20/sm_petli/data/grids/grid_ERA5_EUR_latlon' # noqa
        },
    },

    # ------------------------------------------------------------------------
    # Short-wave down-welling radiation
    'rsds': {
        'ERA5': {
            'path': '/nobackup/rossby20/sm_petli/data/ERA5/VALIDATION/EUR/day',
            'file pattern': 'rsds_day_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': '/nobackup/rossby20/sm_petli/data/grids/grid_ERA5_EUR_latlon' # noqa
        },
        'CLARA_A2': {
            'path': '/nobackup/rossby20/sm_petli/data/CM_SAF/SW/CLARA-A2/day',  # noqa
            'file pattern': 'rsds_CMSAF_CLARA-A2_day_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': None
        },
    },

    # ------------------------------------------------------------------------
    # Long-wave surface net radiation
    'rlns': {
        'ERA5': {
            'path': '/nobackup/rossby20/sm_petli/data/ERA5/VALIDATION/EUR/1h',
            'file pattern': 'rlns_1H_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': '/nobackup/rossby20/sm_petli/data/grids/grid_ERA5_EUR_latlon' # noqa
        },
    },

    # ------------------------------------------------------------------------
    # Short-wave surface net radiation
    'rsns': {
        'ERA5': {
            'path': '/nobackup/rossby20/sm_petli/data/ERA5/VALIDATION/EUR/1h',
            'file pattern': 'rsns_1H_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': '/nobackup/rossby20/sm_petli/data/grids/grid_ERA5_EUR_latlon' # noqa
        },
    },

    # ------------------------------------------------------------------------
    # TOA longwave emissions
    'rlnt': {
        'METEOSAT': {
            'path': '/nobackup/rossby20/sm_petli/data/CM_SAF/TOA/OLR/MSG/day',  # noqa
            'file pattern': 'rlnt_CMSAF_METEOSAT-MSG2_day_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': None
        },
    },

    # ------------------------------------------------------------------------
    # Surface upward latent heat fluxes
    'hfls': {
        'ERA5': {
            'path': '/nobackup/rossby20/sm_petli/data/ERA5/VALIDATION/EUR/1h',
            'file pattern': 'hfls_1H_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': '/nobackup/rossby20/sm_petli/data/grids/grid_ERA5_EUR_latlon' # noqa
        },
    },

    # ------------------------------------------------------------------------
    # Surface upward sensible heat fluxes
    'hfss': {
        'ERA5': {
            'path': '/nobackup/rossby20/sm_petli/data/ERA5/VALIDATION/EUR/1h',
            'file pattern': 'hfss_1H_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': '/nobackup/rossby20/sm_petli/data/grids/grid_ERA5_EUR_latlon' # noqa
        },
    },

    # ------------------------------------------------------------------------
    # Total Cloud Cover
    'clt': {
        'ERA5': {
            'path': '/nobackup/rossby20/sm_petli/data/ERA5/VALIDATION/EUR/day',
            'file pattern': 'clt_day_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': '/nobackup/rossby20/sm_petli/data/grids/grid_ERA5_EUR_latlon' # noqa
        },
        'CLARA_A2': {
            'path': '/nobackup/rossby20/sm_petli/data/CM_SAF/CLOUD/CLARA-A2/day',  # noqa
            'file pattern': 'clt_CMSAF_CLARA-A2_day_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': None
        },
    },

    # ------------------------------------------------------------------------
    # Low-level Cloud Cover
    'cll': {
        'ERA5': {
            'path': '/nobackup/rossby20/sm_petli/data/ERA5/VALIDATION/EUR/day',
            'file pattern': 'cll_day_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': '/nobackup/rossby20/sm_petli/data/grids/grid_ERA5_EUR_latlon' # noqa
        },
        'CLARA_A2': {
            'path': '/nobackup/rossby20/sm_petli/data/CM_SAF/CLOUD/CLARA-A2/day',  # noqa
            'file pattern': 'cll_CMSAF_CLARA-A2_day_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': None
        },
    },

    # ------------------------------------------------------------------------
    # Middle-level Cloud Cover
    'clm': {
        'ERA5': {
            'path': '/nobackup/rossby20/sm_petli/data/ERA5/VALIDATION/EUR/day',
            'file pattern': 'clm_day_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': '/nobackup/rossby20/sm_petli/data/grids/grid_ERA5_EUR_latlon' # noqa
        },
        'CLARA_A2': {
            'path': '/nobackup/rossby20/sm_petli/data/CM_SAF/CLOUD/CLARA-A2/day',  # noqa
            'file pattern': 'clm_CMSAF_CLARA-A2_day_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': None
        },
    },

    # ------------------------------------------------------------------------
    # High-level Cloud Cover
    'clh': {
        'ERA5': {
            'path': '/nobackup/rossby20/sm_petli/data/ERA5/VALIDATION/EUR/day',
            'file pattern': 'clh_day_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': '/nobackup/rossby20/sm_petli/data/grids/grid_ERA5_EUR_latlon' # noqa
        },
        'CLARA_A2': {
            'path': '/nobackup/rossby20/sm_petli/data/CM_SAF/CLOUD/CLARA-A2/day',  # noqa
            'file pattern': 'clh_CMSAF_CLARA-A2_day_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': None
        },
    },

    # ------------------------------------------------------------------------
    # Integral of cloud water
    'clwvi': {
        'ERA5': {
            'path': '/nobackup/rossby20/sm_petli/data/ERA5/VALIDATION/EUR/1h',
            'file pattern': 'clwvi_1H_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': '/nobackup/rossby20/sm_petli/data/grids/grid_ERA5_EUR_latlon' # noqa
        },
        'CLARA_A2': {
            'path': '/nobackup/rossby20/sm_petli/data/CM_SAF/CLOUD/CLARA-A2/day',  # noqa
            'file pattern': 'clwvi_CMSAF_CLARA-A2_day_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': None
        },
    },

    # ------------------------------------------------------------------------
    # Integral of cloud ice
    'clivi': {
        'ERA5': {
            'path': '/nobackup/rossby20/sm_petli/data/ERA5/VALIDATION/EUR/1h',
            'file pattern': 'clivi_1H_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': '/nobackup/rossby20/sm_petli/data/grids/grid_ERA5_EUR_latlon' # noqa
        },
        'CLARA_A2': {
            'path': '/nobackup/rossby20/sm_petli/data/CM_SAF/CLOUD/CLARA-A2/day',  # noqa
            'file pattern': 'clivi_CMSAF_CLARA-A2_day_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': None
        },
    },

    # ------------------------------------------------------------------------
    # Total column water vapor
    'prw': {
        'ERA5': {
            'path': '/nobackup/rossby20/sm_petli/data/ERA5/VALIDATION/EUR/day',
            'file pattern': 'prw_day_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': '/nobackup/rossby20/sm_petli/data/grids/grid_ERA5_EUR_latlon' # noqa
        },
    },

    # ------------------------------------------------------------------------
    # Snow cover
    'sc': {
        'ERA5': {
            'path': '/nobackup/rossby20/sm_petli/data/ERA5/VALIDATION/EUR/3h',
            'file pattern': 'sc_3H_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': '/nobackup/rossby20/sm_petli/data/grids/grid_ERA5_EUR_latlon' # noqa
        },
    },

    # ------------------------------------------------------------------------
    # Snow depth water equivalent
    'snw_b': {
        'ERA5': {
            'path': '/nobackup/rossby20/sm_petli/data/ERA5/VALIDATION/EUR/3h',
            'file pattern': 'sd_3H_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': '/nobackup/rossby20/sm_petli/data/grids/grid_ERA5_EUR_latlon' # noqa
        },
        'seNorge': {
            'path': '/nobackup/rossby20/sm_petli/data/seNorge_snow/day',
            'file pattern': 'snw_b_seNorge_snowsim_v201_day_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': None
        },
    },

    # ------------------------------------------------------------------------
    # Snow depth
    'snd': {
        'ERA5': {
            'path': '/nobackup/rossby20/sm_petli/data/ERA5_snow',
            'file pattern': 'snd_day_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': '/nobackup/rossby20/sm_petli/data/grids/grid_ERA5_EUR_latlon' # noqa
        },
        'seNorge': {
            'path': '/nobackup/rossby20/sm_petli/data/seNorge_snow/day',
            'file pattern': 'snd_seNorge_snowsim_v201_day_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': None
        },
    },

    # ------------------------------------------------------------------------
    # Geopotential 500hPa
    'phi500': {
        'ERAI': {
            'path': '/nobackup/rossby20/sm_petli/data/ERAI/VALIDATION/EUR/day',
            'file pattern': 'phi500_day_ECMWF-ERAINT_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': None
        },
    },

    # ------------------------------------------------------------------------
    # Temperature 850hPa
    'ta850': {
        'ERAI': {
            'path': '/nobackup/rossby20/sm_petli/data/ERAI/VALIDATION/EUR/day',
            'file pattern': 'ta850_day_ECMWF-ERAINT_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': None
        },
    },

    # ------------------------------------------------------------------------
    # Specific humidity 850hPa
    'hus850': {
        'ERAI': {
            'path': '/nobackup/rossby20/sm_petli/data/ERAI/VALIDATION/EUR/day',
            'file pattern': 'hus850_day_ECMWF-ERAINT_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            'grid': None
        },
    },

    # ------------------------ END OF OBSERVATION LIST -----------------------
    }

    return meta_dict


def get_file_list(var, obsname, start_date, end_date):
    """
    Get a list of data set files that covers the time period defined by
    start_date and end_date provided in the function call.

    Parameters
    ----------
    var: str
        Input variable, e.g. 'tas'
    obsname: str
        Name of dataset to use, e.g. 'EOBS'
    start_date: str
        Start date of time period, format YYYYMM
    end_date: str
        End date of time period, format YYYYMM

    Returns
    -------
    file_list: list
        List of obs data files
    """
    meta_data = obs_data()
    data_dict = meta_data[var][obsname]

    file_pattern = data_dict['file pattern']
    sidx = file_pattern.find('YYYYMM')
    eidx = file_pattern.rfind('YYYYMM')

    obs_path_list = glob.glob(os.path.join(data_dict['path'],
                                           file_pattern[:sidx] + '*.nc'))
    obs_path_list.sort()
    obs_file_list = [l.split('/')[-1] for l in obs_path_list]
    obs_dates = ['{}-{}'.format(f[sidx:sidx+6], f[eidx:eidx+6])
                 for f in obs_file_list]
    idx_start = [d.split('-')[0] <= start_date <= d.split('-')[1]
                 for d in obs_dates]
    msg = "Files not found OR selected start date {} ".format(start_date) +\
          "does not match any obs file dates!"
    assert np.sum(idx_start) != 0, msg
    idx_start = np.where(idx_start)[0][0]

    idx_end = [d.split('-')[0] <= end_date <= d.split('-')[1]
               for d in obs_dates]
    msg = "Files not found OR selected end date {} ".format(end_date) +\
          "does not match any obs file dates!"
    assert np.sum(idx_end) != 0, msg
    idx_end = np.where(idx_end)[0][0]

    return obs_path_list[idx_start: idx_end + 1]
