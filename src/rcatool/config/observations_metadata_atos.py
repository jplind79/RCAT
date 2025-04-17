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
    Dictionary with variables as top keys, followed by observation temporal
    resolution (e.g. 'day', '6hr', '1hr'), and the available observations
    nested below. For each observation data set, path and file pattern must
    be defined.
    """

    meta_dict = {

    # ------------------------------------------------------------------------
    # 2m temperature
    'tas': {
        'day': {
            'EOBS': {
                'path': '/perm/sm0i/data/reference_data/EOBS/EOBS25-0e/EUR-10/input/day',
                'file pattern': 'tas_EUR-10_EOBS25-0e_obs_r1i1p1_ECAD_v1_day_YYYYMM01-YYYYMM31.nc', # noqa
            },
            'ERA5': {
                'path': '/perm/sm0i/data/reference_data/ERA5/day/tas',
                'file pattern': 'tas_day_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            },
            'ERA5-Land': {
                'path': '/perm/sm0i/data/reference_data/ERA5-Land/day/tas',
                'file pattern': 'tas_day_ECMWF-ERA5-Land_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            },
        },
        '1hr': {
            'ERA5-Land': {
                'path': '/perm/sm0i/data/reference_data/ERA5-Land/1hr/tas',
                'file pattern': 'tas_1hr_ECMWF-ERA5-Land_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            },
        },
    },

    'tasmax': {
        'day': {
            'EOBS': {
                'path': '/perm/sm0i/data/reference_data/EOBS/EOBS25-0e/EUR-10/input/day',
                'file pattern': 'tasmax_EUR-10_EOBS25-0e_obs_r1i1p1_ECAD_v1_day_YYYYMM01-YYYYMM31.nc', # noqa
            },
            'ERA5': {
                'path': '/perm/sm0i/data/reference_data/ERA5/day/tasmax',
                'file pattern': 'tasmax_day_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            },
        },
    },

    'tasmin': {
        'day': {
            'EOBS': {
                'path': '/perm/sm0i/data/reference_data/EOBS/EOBS25-0e/EUR-10/input/day',
                'file pattern': 'tasmin_EUR-10_EOBS25-0e_obs_r1i1p1_ECAD_v1_day_YYYYMM01-YYYYMM31.nc', # noqa
            },
            'ERA5': {
                'path': '/perm/sm0i/data/reference_data/ERA5/day/tasmin',
                'file pattern': 'tasmin_day_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            },
        },
    },

    # ------------------------------------------------------------------------
    # Surface skin temperature
    'ts': {
        'day': {
            'ERA5-Land': {
                'path': '/perm/sm0i/data/reference_data/ERA5-Land/day/ts',
                'file pattern': 'ts_day_ECMWF-ERA5-Land_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            },
        },
        '1hr': {
            'ERA5-Land': {
                'path': '/perm/sm0i/data/reference_data/ERA5-Land/1hr/ts',
                'file pattern': 'ts_1hr_ECMWF-ERA5-Land_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            },
        },
    },

    # ------------------------------------------------------------------------
    # Precipitation
    'pr': {
        'day': {
            'EOBS': {
                'path': '/perm/sm0i/data/reference_data/EOBS/EOBS25-0e/EUR-10/input/day',
                'file pattern': 'pr_EUR-10_EOBS25-0e_obs_r1i1p1_ECAD_v1_day_YYYYMM01-YYYYMM31.nc', # noqa
            },
            'ERA5': {
                'path': '/perm/sm0i/data/reference_data/ERA5/day/pr',
                'file pattern': 'pr_day_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            },
            'MSWEP': {
                'path': '/perm/sm0i/data/reference_data/MSWEP/day',
                'file pattern': 'pr_MSWEP_v2_europe_0.1deg_day_YYYYMM0100-YYYYMM3100.nc', # noqa
            },
        },
        '1hr': {
            'ERA5': {
                'path': '/perm/sm0i/data/reference_data/ERA5/1hr/pr',
                'file pattern': 'pr_1H_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            },
        },
    },

    # ------------------------------------------------------------------------
    # MSLP
    'psl': {
        'day': {
            'ERA5': {
                'path': '/perm/sm0i/data/reference_data/ERA5/day/psl',
                'file pattern': 'psl_day_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            },
        },
    },

    # ------------------------------------------------------------------------
    # Surface wind speed
    'sfcWind': {
        'day': {
            'EOBS': {
                'path': '/perm/sm0i/data/reference_data/EOBS/EOBS25-0e/EUR-10/input/day',
                'file pattern': 'sfcWind_EUR-10_EOBS25-0e_obs_r1i1p1_ECAD_v1_day_YYYYMM01-YYYYMM31.nc', # noqa
            },
        },
    },

    # ------------------------------------------------------------------------
    # Near-surface relative humidity
    'hurs': {
        'day': {
            'EOBS': {
                'path': '/perm/sm0i/data/reference_data/EOBS/EOBS25-0e/EUR-10/input/day',
                'file pattern': 'hurs_EUR-10_EOBS25-0e_obs_r1i1p1_ECAD_v1_day_YYYYMM01-YYYYMM31.nc', # noqa
            },
        },
    },

    # ------------------------------------------------------------------------
    # Total evaporation over land
    'evaptl': {
        'day': {
            'GLEAM': {
                'path': '/perm/sm0i/data/reference_data/GLEAM/v3.6a/europe/day/evaptl',
                'file pattern': 'evaptl_GLEAM_v3.6a_europe_YYYYMM01-YYYYMM31.nc', # noqa
            },
        },
    },

    # ------------------------------------------------------------------------
    # Short-wave down-welling radiation
    'rsds': {
        'day': {
            'ERA5': {
                'path': '/perm/smf/obs/ERA5/input/day',
                'file pattern': 'rsds_day_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            },
            'ERA5-Land': {
                'path': '/perm/sm0i/data/reference_data/ERA5-Land/day/rsds',
                'file pattern': 'rsds_day_ECMWF-ERA5-Land_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            },
            'CLARA_A3': {
                'path': '/perm/sm0i/data/reference_data/CLARA_A3/day/rsds',  # noqa
                'file pattern': 'rsds_CMSAF_CLARA-A3_day_YYYYMM01-YYYYMM31.nc', # noqa
            },
            'EOBS': {
                'path': '/perm/sm0i/data/reference_data/EOBS/EOBS25-0e/EUR-10/input/day',
                'file pattern': 'rsds_EUR-10_EOBS25-0e_obs_r1i1p1_ECAD_v1_day_YYYYMM01-YYYYMM31.nc', # noqa
            },
        },
        '1hr': {
            'ERA5-Land': {
                'path': '/perm/sm0i/data/reference_data/ERA5-Land/1hr/rsds',
                'file pattern': 'rsds_1hr_ECMWF-ERA5-Land_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            },
        },
    },

    # ------------------------------------------------------------------------
    # Long-wave down-welling radiation
    'rlds': {
        'day': {
            'ERA5-Land': {
                'path': '/perm/sm0i/data/reference_data/ERA5-Land/day/rlds',
                'file pattern': 'rlds_day_ECMWF-ERA5-Land_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            },
        },
        '1hr': {
            'ERA5-Land': {
                'path': '/perm/sm0i/data/reference_data/ERA5-Land/1hr/rlds',
                'file pattern': 'rlds_1hr_ECMWF-ERA5-Land_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            },
        },
    },

    # ------------------------------------------------------------------------
    # Short-wave surface net radiation
    'rsns': {
        'day': {
            'ERA5': {
                'path': '/perm/sm0i/data/reference_data/ERA5/day/rsns',
                'file pattern': 'rsns_day_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            },
            'ERA5-Land': {
                'path': '/perm/sm0i/data/reference_data/ERA5-Land/day/rsns',
                'file pattern': 'rsns_day_ECMWF-ERA5-Land_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            },
        },
        '1hr': {
            'ERA5-Land': {
                'path': '/perm/sm0i/data/reference_data/ERA5-Land/1hr/rsns',
                'file pattern': 'rsns_1hr_ECMWF-ERA5-Land_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            },
        },
    },

    # ------------------------------------------------------------------------
    # Long-wave surface net radiation
    'rlns': {
        'day': {
            'ERA5': {
                'path': '/perm/sm0i/data/reference_data/ERA5/day/rlns',
                'file pattern': 'rlns_day_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            },
            'ERA5-Land': {
                'path': '/perm/sm0i/data/reference_data/ERA5-Land/day/rlns',
                'file pattern': 'rlns_day_ECMWF-ERA5-Land_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            },
        },
        '1hr': {
            'ERA5-Land': {
                'path': '/perm/sm0i/data/reference_data/ERA5-Land/1hr/rlns',
                'file pattern': 'rlns_1hr_ECMWF-ERA5-Land_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            },
        },
    },

    # ------------------------------------------------------------------------
    # Surface upward latent heat fluxes
    'hfls': {
        'day': {
            'ERA5': {
                'path': '/perm/sm0i/data/reference_data/ERA5/day/hfls',
                'file pattern': 'hfls_day_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            },
            'ERA5-Land': {
                'path': '/perm/sm0i/data/reference_data/ERA5-Land/day/hfls',
                'file pattern': 'hfls_day_ECMWF-ERA5-Land_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            },
        },
        '1hr': {
            'ERA5-Land': {
                'path': '/perm/sm0i/data/reference_data/ERA5-Land/1hr/hfls',
                'file pattern': 'hfls_1hr_ECMWF-ERA5-Land_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            },
        },
    },

    # ------------------------------------------------------------------------
    # Surface upward sensible heat fluxes
    'hfss': {
        'day': {
            'ERA5': {
                'path': '/perm/sm0i/data/reference_data/ERA5/day/hfss',
                'file pattern': 'hfss_day_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            },
            'ERA5-Land': {
                'path': '/perm/sm0i/data/reference_data/ERA5-Land/day/hfss',
                'file pattern': 'hfss_day_ECMWF-ERA5-Land_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            },
        },
        '1hr': {
            'ERA5-Land': {
                'path': '/perm/sm0i/data/reference_data/ERA5-Land/1hr/hfss',
                'file pattern': 'hfss_1hr_ECMWF-ERA5-Land_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            },
        },
    },

    # ------------------------------------------------------------------------
    # Total Cloud Cover
    'clt': {
        'day': {
            'ERA5': {
                'path': '/perm/sm0i/data/reference_data/ERA5/day',
                'file pattern': 'clt_day_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            },
            'CLARA_A3': {
                'path': '/perm/sm0i/data/reference_data/CLARA_A3/day/clt',  # noqa
                'file pattern': 'clt_CMSAF_CLARA-A3_day_YYYYMM01-YYYYMM31.nc', # noqa
            },
        },
    },

    # ------------------------------------------------------------------------
    # Low-level Cloud Cover
    'cll': {
        'day': {
            'CLARA_A3': {
                'path': '/perm/sm0i/data/reference_data/CLARA_A3/day/cll',  # noqa
                'file pattern': 'cll_CMSAF_CLARA-A3_day_YYYYMM01-YYYYMM31.nc', # noqa
            },
        },
    },

    # ------------------------------------------------------------------------
    # Middle-level Cloud Cover
    'clm': {
        'day': {
            'CLARA_A3': {
                'path': '/perm/sm0i/data/reference_data/CLARA_A3/day/clm',  # noqa
                'file pattern': 'clm_CMSAF_CLARA-A3_day_YYYYMM01-YYYYMM31.nc', # noqa
            },
        },
    },

    # ------------------------------------------------------------------------
    # High-level Cloud Cover
    'clh': {
        'day': {
            'CLARA_A3': {
                'path': '/perm/sm0i/data/reference_data/CLARA_A3/day/clh',  # noqa
                'file pattern': 'clh_CMSAF_CLARA-A3_day_YYYYMM01-YYYYMM31.nc', # noqa
            },
        },
    },


    # ------------------------ END OF OBSERVATION LIST -----------------------
    }

    return meta_dict


def get_file_list(var, obsname, obsfreq, start_date, end_date):
    """
    Get a list of data set files that covers the time period defined by
    start_date and end_date provided in the function call.

    Parameters
    ----------
    var: str
        Input variable, e.g. 'tas'
    obsname: str
        Name of dataset to use, e.g. 'EOBS'
    obsfreq: str
        Temporal resolution of dataset to use, e.g. 'day' or '1hr'
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

    data_dict = meta_data.get(var, {}).get(obsfreq, {}).get(obsname)

    # Check if data configuration exists
    if data_dict is None:
        errmsg = f"""\n\t\t** Error **
                 Could not find observation data for:
                 \tobs: {obsname}, var: {var}, freq: {obsfreq}.

                 Please Check settings in the obs meta data file\n"""
        raise ValueError(errmsg)

    file_pattern = data_dict['file pattern']
    sidx = file_pattern.find('YYYYMM')
    eidx = file_pattern.rfind('YYYYMM')

    obs_path_list = glob.glob(os.path.join(data_dict['path'],
                                           file_pattern[:sidx] + '*.nc'))
    obs_path_list.sort()
    obs_file_list = [ln.split('/')[-1] for ln in obs_path_list]
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
