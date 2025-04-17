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
                'path': '/home/rossby/imports/obs/EOBS/EOBS20/EUR-10/input/day',
                'file pattern': 'tas_EUR-10_EOBS20e_obs_r1i1p1_ECAD_v1_day_YYYYMM01-YYYYMM31.nc', # noqa
            },
            'ERA5': {
                'path': '/nobackup/rossby20/sm_petli/data/ERA5/VALIDATION/EUR/day',
                'file pattern': 'tas_day_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            },
        },
        '1hr': {
            'EOBS': {
                'path': '/home/rossby/imports/obs/EOBS/EOBS20/EUR-10/input/1hr',
                'file pattern': 'tas_EUR-10_EOBS20e_obs_r1i1p1_ECAD_v1_1hr_YYYYMM01-YYYYMM31.nc', # noqa
            },
            'ERA5': {
                'path': '/nobackup/rossby20/sm_petli/data/ERA5/VALIDATION/EUR/1hr',
                'file pattern': 'tas_1hr_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            },
        },
    },

    # ------------------------------------------------------------------------
    # Precipitation
    'pr': {
        'day': {
            'EOBS': {
                'path': '/home/rossby/imports/obs/EOBS/EOBS20/EUR-10/input/day',
                'file pattern': 'pr_EUR-10_EOBS20e_obs_r1i1p1_ECAD_v1_day_YYYYMM01-YYYYMM31.nc', # noqa
            },
            'ERA5': {
                'path': '/nobackup/rossby20/sm_petli/data/ERA5/VALIDATION/EUR/1h',
                'file pattern': 'pr_1H_ECMWF-ERA5_rean_r1i1p1_YYYYMM01-YYYYMM31.nc', # noqa
            },
        },
        '1hr': {
            'SENORGE': {
                'path': '/nobackup/rossby20/sm_petli/data/seNorge_pr/orig',
                'file pattern': 'pr_seNorge2_PREC1h_grid_YYYYMM.nc', # noqa
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
