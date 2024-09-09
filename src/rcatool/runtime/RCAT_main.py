#!/usr/bin/env python
# coding: utf-8

# Modules
import sys
import os
import warnings
import glob
import xarray as xa
import xesmf as xe
import pandas as pd
from pandas import to_timedelta
import datetime as dt
from itertools import product
import numpy as np
import re
from dask.distributed import Client
# from dask.diagnostics import ProgressBar
from rcatool.utils import ini_reader
from rcatool.utils.polygons import mask_region
import rcatool.runtime.RCAT_stats as st
import rcatool.utils.grids as gr

import dask
dask.config.set(scheduler="single-threaded")

warnings.filterwarnings("ignore")


# Functions
def get_config_settings(config_file):
    """Retrieve information from main configuration file"""

    conf_dict = ini_reader.get_config_dict(config_file)
    settings_dict = {
        'models': conf_dict['MODELS'],
        'obs metadata file': conf_dict['OBS']['metadata file'],
        'obs time dict': {
            'start year': conf_dict['OBS']['start year'],
            'end year': conf_dict['OBS']['end year'],
            'months': conf_dict['OBS']['months'],
            'date interval start': conf_dict['OBS']['date interval start'],
            'date interval end': conf_dict['OBS']['date interval end']
        },
        'variables': conf_dict['SETTINGS']['variables'],
        'var modification': conf_dict['SETTINGS']['variable modification'],
        'regions': conf_dict['SETTINGS']['regions'],
        'full domain': conf_dict['SETTINGS']['full domain'],
        'requested_stats': conf_dict['STATISTICS']['stats'],
        'stats_conf': st.mod_stats_config(conf_dict['STATISTICS']['stats']),
        'validation plot': conf_dict['PLOTTING']['validation plot'],
        'map projection': conf_dict['PLOTTING']['map projection'],
        'map extent': conf_dict['PLOTTING']['map extent'],
        'map gridlines': conf_dict['PLOTTING']['map gridlines'],
        'map configuration': conf_dict['PLOTTING']['map configuration'],
        'map grid config': conf_dict['PLOTTING']['map grid config'],
        'map plot kwargs': conf_dict['PLOTTING']['map plot kwargs'],
        'map domain': conf_dict['PLOTTING']['map model domain'],
        'line grid setup': conf_dict['PLOTTING']['line grid setup'],
        'line kwargs': conf_dict['PLOTTING']['line kwargs'],
        'cluster type': conf_dict['CLUSTER']['cluster type'],
        'nodes': conf_dict['CLUSTER']['nodes'],
        'cluster kwargs': conf_dict['CLUSTER']['cluster kwargs'],
        'outdir': conf_dict['SETTINGS']['output dir'],
    }

    return settings_dict


def cluster_setup(clr_type, clr_kwargs, nrnodes=1):
    """Set up cluster on local or HPC system

    Parameters
    ----------
    clr_type:
        Which cluster to set up; local, SLURM, ...
    clr_kwargs:
        Keyword arguments for cluster specifications
    nrnodes: int
        Number of nodes to use
    """

    if clr_type == 'local':
        from dask.distributed import LocalCluster
        cluster = LocalCluster(processes=False)
    elif clr_type == 'slurm':
        from dask_jobqueue import SLURMCluster
        cluster = SLURMCluster(**clr_kwargs)
        cluster.scale(jobs=nrnodes)
    else:
        print(f"\n\tCluster type {cluster_type} not implemented! Exiting...")
        sys.exit()

    return cluster


def set_def_args():
    """Set argparse argument"""
    import argparse

    # Configuring argument setup and handling
    parser = argparse.ArgumentParser(
        description='Main script for model/obs validation')
    parser.add_argument('--config', '-c',  metavar='name config file',
                        type=str, help='<Required> Full path to config file',
                        required=True)
    return parser.parse_args()


def get_variable_config(var_config):
    """Retrieve information and settings for variable var as defined in main
    configuration file
    """
    vardict = {
        'var names': var_config['var names'],
        'input resolution': var_config['freq'],
        'units': var_config['units'],
        'scale factor': var_config['scale factor'],
        'offset factor': var_config['offset factor'],
        'obs scale factor': var_config['obs scale factor'] if
        'obs scale factor' in var_config else None,
        'deacc': var_config['accumulated'],
        'regrid': var_config['regrid to'] if 'regrid to' in
        var_config else None,
        'rgr method': var_config['regrid method'] if 'regrid method' in
        var_config else None,
    }
    return vardict


def variabel_modification(dd, nv_dd, funargs, mlist, olist):
    """
    Create new or modify existing variables.
    """
    from inspect import signature
    out_dd = {}
    expression = nv_dd['expression']
    func = eval(f"lambda {funargs}: {expression}")
    args = list(signature(func).parameters)
    invar = [v for k, v in nv_dd['input'].items()]
    _modlist = nv_dd['models']
    _obslist = nv_dd['obs']

    if _modlist is not None:
        modlist = _modlist if isinstance(_modlist, list) else mlist \
                if _modlist == 'all' else [_modlist]
        for m in modlist:
            if 'resample resolution' in nv_dd:
                dd_res = {}
                for v in invar:
                    if v in nv_dd['resample resolution']:
                        tres = nv_dd['resample resolution'][v]
                        dd_res[v] = data_resampling(dd[v][m]['data'], tres)
                    else:
                        dd_res[v] = dd[v][m]['data']
                input_data = [
                    dd_res[nv_dd['input'][a]][nv_dd['input'][a]]
                    for a in args]
            else:
                input_data = [
                    dd[nv_dd['input'][a]][m]['data'][nv_dd['input'][a]]
                    for a in args]

            # 2024-04: Temp fix for EURO-CORDEX upsampling HCLIM-ALADIN data
            time_fix = input_data[0].time
            input_data[1] = input_data[1].reindex(time=time_fix).ffill(
                dim='time')
            _new_data = xa.apply_ufunc(
                func, *input_data, input_core_dims=[[]]*len(input_data),
                dask='parallelized', output_dtypes=[float],
            )
            new_data = _new_data.to_dataset(name=new_var).assign_coords(
                input_data[0].coords)
            out_dd[m] = {'data': new_data}
    if _obslist is not None:
        obslist = _obslist if isinstance(_obslist, list) else olist \
                if _obslist == 'all' else [_obslist]
        for o in obslist:
            input_data = [dd[nv_dd['input'][a]][o]['data'][nv_dd['input'][a]]
                          for a in args]
            _new_data = xa.apply_ufunc(
                func, *input_data, input_core_dims=[[]]*len(input_data),
                dask='parallelized', output_dtypes=[float],
            )
            new_data = _new_data.to_dataset(name=new_var)
            out_dd[o] = {'data': new_data}

    return out_dd


def get_mod_data(model, mconf, tres, var, varnames, factor, offset, deacc):
    """Open and preprocess model data"""
    import re

    print("\t-- Opening {} files\n".format(model.upper()))

    if mconf['date interval start'] is not None:
        start_date = mconf['date interval start']
        end_date = mconf['date interval end']
        date_list = pd.date_range(start_date, end_date, freq='MS').strftime(
            "%Y%m").tolist()
    else:
        start_year = mconf['start year']
        end_year = mconf['end year']
        months = mconf['months']
        date_list = ["{}{:02d}".format(yy, mm) for yy, mm in product(
            range(start_year, end_year+1), months)]

    # Variable name in filename/file
    if varnames is not None:
        if 'all' in varnames:
            readvar = varnames['all']['prfx']
        elif model in varnames:
            readvar = varnames[model]['prfx']
        else:
            readvar = var
    else:
        readvar = var

    file_path = os.path.join(mconf['fpath'],
                             f'{tres}/{readvar}/{readvar}_*.nc')
    _flist = glob.glob(file_path)

    errmsg = (f"Could not find any files at specified location:\n{file_path}")
    if not _flist:
        raise FileNotFoundError(errmsg)

    # Extract dates from listed file names
    _file_dates = [re.split('-|_', f.rsplit('.')[-2])[-2:] for f in _flist]
    file_dates = [(d[0][:6], d[1][:6]) for d in _file_dates]

    # Find which file dates fall into chosen time period
    fidx = [np.where([d[0] <= date <= d[1] for d in file_dates])[0]
            for date in date_list]

    # Select input files
    flist = [_flist[i] for i in np.unique(np.hstack(fidx))]
    flist.sort()

    if np.unique([len(f) for f in flist]).size > 1:
        flngth = np.unique([len(f) for f in flist])
        flist = [f for f in flist if len(f) == flngth[0]]

    # Chunk sizes in temporal and spatial dimensions
    ch_t = mconf['chunks_time']
    ch_x = mconf['chunks_x']
    ch_y = mconf['chunks_y']

    # -- Opening files (possibly with de-accumulation preprocessing)
    if deacc:
        _mdata = xa.open_mfdataset(
            flist, parallel=True, engine='h5netcdf',
            data_vars='minimal', coords='minimal', combine='by_coords',
            chunks={**ch_t, **ch_x, **ch_y},
            preprocess=(lambda arr: arr.diff('time'))).drop_duplicates(
                dim='time', keep='last')
        _mdata = _mdata.chunk({**ch_t}).unify_chunks()

        # Modify time stamps to mid-point
        diff = _mdata.time.values[1] - _mdata.time.values[0]
        nsec = to_timedelta(diff).total_seconds()
        _mdata['time'] = _mdata.time -\
            np.timedelta64(dt.timedelta(seconds=np.round(nsec/2)))
    else:
        _mdata = xa.open_mfdataset(
            flist, parallel=True, engine='h5netcdf',
            data_vars='minimal', coords='minimal', combine='by_coords',
            chunks={**ch_t, **ch_x, **ch_y}).drop_duplicates(
                dim='time', keep='last')
        _mdata = _mdata.chunk({**ch_t}).unify_chunks()

    # EDIT 240805: Is this still needed??
    # Time stamps
    if 'units' in _mdata.time.attrs:
        if _mdata.time.attrs['units'] == 'day as %Y%m%d.%f':
            dates = [pd.to_datetime(d, format='%Y%m%d') +
                     pd.to_timedelta((d % 1)*86400, unit='s').round('H')
                     for d in _mdata.time.values]
            _mdata['time'] = (('time',), dates)
        else:
            raise ValueError(("\n\n\tUnfortunately, at the moment the time "
                              "units in file cannot be treated, change if "
                              "possible"))

    # Extract years and months
    if mconf['date interval start'] is not None:
        mdata = _mdata.sel(time=slice(start_date, end_date))
    else:
        mdata = _mdata.where(((_mdata.time.dt.year >= start_year) &
                              (_mdata.time.dt.year <= end_year) &
                              (np.isin(_mdata.time.dt.month, months))),
                             drop=True)

    # Rename variable if not consistent with name in configuration file
    if varnames is not None:
        if 'all' in varnames:
            mdata = mdata.rename({varnames['all']['vname']: var})
        elif model in varnames:
            mdata = mdata.rename({varnames[model]['vname']: var})

    # Scale data
    if factor is not None:
        mdata[var] *= factor

    # Add/subtract to data
    if offset is not None:
        mdata[var] += offset

    # Model grid
    gridname = 'grid_{}'.format(mconf['grid name'])

    # Labels for spatial coordinates
    xc, yc = _space_coords(mdata)

    # Labels for spatial dimensions
    xd, yd = _space_dim(mdata)

    # Make sure lon/lat elements are in ascending order
    mdata = _coords_in_ascending_order(mdata, xd, xc, yd, yc)

    # Convert from 0->360 to -180->180, if needed
    if mdata[xc].max() > 180:
        mdata.coords[xc] = (mdata.coords[xc] + 180) % 360 - 180
        if mdata.coords[xc].ndim == 1:
            mdata = mdata.sortby(mdata[xc])

    # -- Unrotate grid if needed
    # Certain coordinates & meta data expected to make the transform, otherwise
    # errors will be raised.
    if mconf['grid type'] == 'rot':
        lon_reg, lat_reg = gr.rotated_grid_transform(
            mdata.rlon, mdata.rlat,
            mdata.rotated_pole.grid_north_pole_longitude,
            mdata.rotated_pole.grid_north_pole_latitude)
        mdata = mdata.assign_coords({'lon': (('y', 'x'), lon_reg),
                                     'lat': (('y', 'x'), lat_reg)}).\
            swap_dims({'rlon': 'x', 'rlat': 'y'})
        grid = {'lon': lon_reg, 'lat': lat_reg}
    else:
        grid = {'lon': mdata[xc].values, 'lat': mdata[yc].values}

    model_data = {'data': mdata.unify_chunks(),
                  'grid': grid, 'gridname': gridname}

    return model_data


def get_obs_data(metadata_file, obs, var, factor, time_dict):
    """Open observation data"""

    from importlib.machinery import SourceFileLoader
    obs_meta = SourceFileLoader("obs_meta", metadata_file).load_module()

    if time_dict['date interval start'] is not None:
        start_date = time_dict['date interval start']
        end_date = time_dict['date interval end']
        date_list = pd.date_range(start_date, end_date, freq='MS').strftime(
            "%Y%m").tolist()
    else:
        start_year = time_dict['start year']
        end_year = time_dict['end year']
        months = time_dict['months']
        start_date = '{}{:02d}'.format(start_year, np.min(months))
        end_date = '{}{:02d}'.format(end_year, np.max(months))
        date_list = ["{}{:02d}".format(yy, mm) for yy, mm in product(
            range(start_year, end_year+1), months)]

    print("\t-- Opening {} files\n".format(obs.upper()))

    # Open obs files
    obs_flist = obs_meta.get_file_list(var, obs, start_date, end_date)

    emsg = ("Could not find any {} files at specified location"
            "\n\nexiting ...".format(obs.upper()))
    if not obs_flist:
        print("\t\n{}".format(emsg))
        sys.exit()

    # Extract dates from listed file names
    _file_dates = [re.split('-|_', f.rsplit('.')[-2])[-2:] for f in obs_flist]
    file_dates = [(d[0][:6], d[1][:6]) for d in _file_dates]

    # Find which file dates fall into chosen time period
    fidx = [np.where([d[0] <= date <= d[1] for d in file_dates])[0]
            for date in date_list]
    flist = [obs_flist[i] for i in np.unique(np.hstack(fidx))]
    flist.sort()

    f_obs = xa.open_mfdataset(
        flist, parallel=True, data_vars='minimal', coords='minimal',
        combine='by_coords').unify_chunks()

    # Extract years and months
    if time_dict['date interval start'] is not None:
        obs_data = f_obs.sel(time=slice(start_date, end_date))
    else:
        obs_data = f_obs.where(((f_obs.time.dt.year >= start_year) &
                                (f_obs.time.dt.year <= end_year) &
                                (np.isin(f_obs.time.dt.month, months))),
                               drop=True)
    # Scale data
    if factor is not None:
        obs_data[var] *= factor

    # Labels for spatial coordinates
    xc, yc = _space_coords(obs_data)

    # Labels for spatial dimensions
    xd, yd = _space_dim(obs_data)

    # Make sure lon/lat elements are in ascending order
    obs_data = _coords_in_ascending_order(obs_data, xd, xc, yd, yc)

    # Convert from 0->360 to -180->180 if needed
    if obs_data[xc].max() > 180:
        obs_data.coords[xc] = (obs_data.coords[xc] + 180) % 360 - 180
        if obs_data.coords[xc].ndim == 1:
            obs_data = obs_data.sortby(obs_data[xc])

    lons = obs_data[xc].values
    lats = obs_data[yc].values

    grid = {'lon': lons, 'lat': lats}
    gridname = 'grid_{}'.format(obs.upper())

    outdata = {'data': obs_data, 'grid': grid, 'gridname': gridname}

    return outdata


def get_grid_coords(ds, grid_coords):
    """Read model grid coordinates

    Parameters
    ----------
    ds: xarray dataset

    Returns
    -------
    grid_coords: dict
        Containing data grid domain information
    """

    def _domain(lats, lons, x):
        if x == 0:
            lons_p = np.r_[lons[x, x::], lons[x::, -1-x][1:],
                           lons[-1-x, x::][::-1][1:], lons[x::, x][::-1][1:]]
            lats_p = np.r_[lats[x, x::], lats[x::, -1-x][1:],
                           lats[-1-x, x::][::-1][1:], lats[x::, x][::-1][1:]]
        else:
            lons_p = np.r_[lons[x, x:-x], lons[x:-x, -1-x][1:],
                           lons[-1-x, x:-x][::-1][1:], lons[x:-x, x][::-1][1:]]
            lats_p = np.r_[lats[x, x:-x], lats[x:-x, -1-x][1:],
                           lats[-1-x, x:-x][::-1][1:], lats[x:-x, x][::-1][1:]]

        return list(zip(lons_p, lats_p))

    # If lon/lat is 1D; create 2D meshgrid
    xd, yd = _space_coords(ds)
    lons = ds[xd].values
    lats = ds[yd].values
    lon, lat = np.meshgrid(lons, lats)\
        if lats.ndim == 1 else (lons, lats)

    # Calculate domain mid point if not given
    idx = tuple([int(i/2) for i in lat.shape])
    lat0 = lat[idx]
    lon0 = lon[idx]

    grid_coords['lat_0'] = lat0
    grid_coords['lon_0'] = lon0

    # gp_bfr could be changed > 0 to use a lateral buffer zone for plotting
    # So far hardcoded to zero
    gp_bfr = 0
    grid_coords['crnrs'] = [lat[gp_bfr, gp_bfr], lon[gp_bfr, gp_bfr],
                            lat[-(1 + gp_bfr), -(1 + gp_bfr)],
                            lon[-(1 + gp_bfr), -(1 + gp_bfr)]]
    grid_coords['domain'] = _domain(lat, lon, gp_bfr)

    return grid_coords


def get_grids(data, target_grid, method='bilinear'):
    """Get and modify the source and target grids for interpolation"""
    # EDIT 15/11 2019 by Petter Lind
    # There's no consistent way to handle data when lon_bnds/lat_bnds are
    # already available. They may have different formats not all handled by
    # xesmf regridding tool. Thus, for now, calculation of grid corners is
    # always performed when method requires it (e.g. 'conservative')

    # Spatial coordinates
    xd, yd = _space_coords(data)

    if method == 'conservative':
        slon_b, slat_b = gr.fnCellCorners(data[xd].values, data[yd].values)
        s_grid = {'lon': data[xd].values, 'lat': data[yd].values,
                  'lon_b': slon_b, 'lat_b': slat_b}
        tlon_b, tlat_b = gr.fnCellCorners(target_grid['lon'],
                                          target_grid['lat'])
        t_grid = {'lon': target_grid['lon'], 'lat': target_grid['lat'],
                  'lon_b': tlon_b, 'lat_b': tlat_b}
    else:
        s_grid = {'lon': data[xd].values, 'lat': data[yd].values}
        t_grid = target_grid

    return s_grid, t_grid


def data_interpolation(dd, var, varconf, modnames, obsnames, griddict):
    if varconf['regrid'] is None:
        griddict.update(
            {'lon': {m: dd[m]['grid']['lon'] for m in modnames},
             'lat': {m: dd[m]['grid']['lat'] for m in modnames}})
        if None not in obsnames:
            for obs in obsnames:
                griddict['lon'].update({obs: dd[obs]['grid']['lon']})
                griddict['lat'].update({obs: dd[obs]['grid']['lat']})
        gridname = 'native_grid'
    else:
        if isinstance(varconf['regrid'], dict):
            target_grid = xa.open_dataset(varconf['regrid']['file'])
            gridname = "grid_{}".format(varconf['regrid']['name'])
            griddict.update({'lon': {gridname: target_grid['lon'].values},
                            'lat': {gridname: target_grid['lat'].values}})
            for mod in modnames:
                dd[mod]['data'] = regrid_calc(dd, mod, var, target_grid,
                                              varconf['rgr method'])
            if None not in obsnames:
                for obs in obsnames:
                    dd[obs]['data'] = regrid_calc(dd, obs, var, target_grid,
                                                  varconf['rgr method'])
        elif varconf['regrid'] in obsnames:
            rgr_oname = varconf['regrid']
            target_grid = dd[rgr_oname]['grid']
            gridname = dd[rgr_oname]['gridname']
            griddict.update({'lon': {rgr_oname: target_grid['lon']},
                             'lat': {rgr_oname: target_grid['lat']}})
            for m in modnames:
                dd[m]['data'] = regrid_calc(dd, m, var, target_grid,
                                            varconf['rgr method'])
            if len(obsnames) > 1:
                obslist = obsnames.copy()
                obslist.remove(rgr_oname)
                for obs in obslist:
                    dd[obs]['data'] = regrid_calc(dd, obs, var, target_grid,
                                                  varconf['rgr method'])
        elif varconf['regrid'] in modnames:
            mname = varconf['regrid']
            modlist = modnames.copy()
            modlist.remove(mname)
            target_grid = dd[mname]['grid']
            gridname = dd[mname]['gridname']
            griddict.update({'lon': {mname: target_grid['lon']},
                            'lat': {mname: target_grid['lat']}})
            for mod in modlist:
                dd[mod]['data'] = regrid_calc(dd, mod, var, target_grid,
                                              varconf['rgr method'])
            if None not in obsnames:
                for obs in obsnames:
                    dd[obs]['data'] = regrid_calc(dd, obs, var, target_grid,
                                                  varconf['rgr method'])
        else:
            raise ValueError(("\n\n\tTarget grid name not found!\n"
                              "Check 'regrid to' option in main config file"))

    griddict.update({'gridname': gridname})

    return dd, griddict


def regrid_calc(data, data_name, var, target_grid, method):
    """Perform interpolation between grids"""

    print("\t\t** Regridding {} data **\n".format(data_name.upper()))

    indata = data[data_name]['data']

    # Get grid info of source and target grids
    sgrid, tgrid = get_grids(indata, target_grid, method)

    # Regridding
    regridder = xe.Regridder(sgrid, tgrid, method, unmapped_to_nan=True,
                             ignore_degenerate=True)
    darr_rgr = regridder(indata[var])
    dset = darr_rgr.to_dataset(name=var)

    return dset


def data_resampling(data, resample_conf):
    """Resample data according to chosen time frequency and resample method."""

    errmsg = ("\t\t\nResample resolution argument must be a list of two "
              "or three items. Please re-check the configuration.")
    assert len(resample_conf) in (2, 3), errmsg

    if resample_conf[0] in ('select hours', 'select dates'):
        resample_val = resample_conf[1]
        ll_val = resample_val if isinstance(resample_val, list) else\
            [resample_val]
        if resample_conf[0] == 'select hours':
            resampled_data = data.sel(time=np.isin(data['time.hour'], ll_val))
        else:
            resampled_data = data.sel(time=ll_val)
    else:
        diff = data.time.values[1] - data.time.values[0]
        nsec = to_timedelta(diff).total_seconds()
        tres, freq = _get_freq(resample_conf[0])
        sec_resample = to_timedelta(tres, freq).total_seconds()

        if resample_conf[1] == 'interpolate':
            expr = (f"data.resample(time='{resample_conf[0]}')"
                    f".interpolate('{resample_conf[2]}')")
        else:
            expr = (f"data.resample(time='{resample_conf[0]}')"
                    f".{resample_conf[1]}('time').dropna('time', how='all')")
        if nsec != sec_resample:
            resampled_data = eval(expr)
            time_comp = (np.unique(resampled_data['time.month']) ==
                         np.unique(data['time.month']))
            if not np.all(time_comp):
                print("\t\t** The resampling added months - removing these **")
                resampled_data = resampled_data.isel(
                    time=np.isin(resampled_data['time.month'],
                                 np.unique(data['time.month'])))
            if sec_resample < 24*3600:
                data = eval(
                    f"data.resample(time='{resample_conf[0]}', "
                    f"label='right').{resample_conf[1]}('time')."
                    f"dropna('time', how='all')")
                # EDIT 210608: Should the time stamp be set to midpoint?
                resampled_data['time'] = resampled_data.time + np.timedelta64(
                    dt.timedelta(seconds=np.round(sec_resample/2)))
        else:
            print("\t\tData is already at target resolution, skipping "
                  "resampling ...\n")
            resampled_data = data.copy()
    return resampled_data


def conditional_data_selection(dd_condition, cond_var, cond_data,
                               indata, in_var):
    """Sub-select data conditional on other data"""
    import operator

    def _percentile_func(arr, axis=0, q=95, thr=None):
        if thr is not None:
            arr[arr < thr] = np.nan
        pctl = np.nanpercentile(arr, axis=axis, q=q)
        if axis == -1 and pctl.ndim > 2:
            pctl = np.moveaxis(pctl, 0, -1)
        return pctl

    def _get_cond_mask(darr, oper, q, expand=None, axis=0):
        mask = ops[oper](darr, q)
        if expand is not None:
            mask = np.repeat(mask, expand, axis=axis)

        return mask

    # Relational operators
    ops = {'>': operator.gt,
           '<': operator.lt,
           '>=': operator.ge,
           '<=': operator.le,
           '=': operator.eq}

    # Number of time steps in input data
    ts = indata.time.size

    # Resample data
    if 'resample resolution' in dd_condition:
        print("\t\tResampling conditional data ...\n")
        tres = dd_condition['resample resolution']
        cond_data = data_resampling(cond_data, tres)
    else:
        tres = None

    # Type of conditional: from file, static value or percentile
    cond_type = dd_condition['type']

    # Get relational operator for conditional selection
    operator = dd_condition['operator']

    if cond_type == 'file':
        wrnmsg = ("\t\t\n*** WARNING:\n Variable of conditional file data is"
                  " not the same as input data variable! ***\n")
        errmsg = ("\t\t\nConditional data from file must be 2D!\n")
        file_in = dd_condition['file in']
        file_var = dd_condition['file var']
        if file_var != in_var:
            warnings.warn(wrnmsg)
        with xa.open_dataset(file_in) as fopen:
            q = fopen[file_var].squeeze()
            assert q.ndim == 2, errmsg
        sub_data = indata.where(ops[operator](indata, q))
        sub_data.attrs['Conditional_subselection:'] =\
            (f"Type: file ({file_in}) | File var: {file_var} | "
             f"Operator: {operator} | Resampled cond data: {tres}")
    else:
        q = float(dd_condition['value'])

        if cond_data.time.size != ts:
            expand = ts/cond_data.time.size
        else:
            expand = None
        if cond_type == 'static':
            mask = xa.apply_ufunc(
                _get_cond_mask, cond_data[cond_var], input_core_dims=[[]],
                output_core_dims=[[]], dask='parallelized',
                output_dtypes=[bool], kwargs={
                    'oper': operator, 'q': q, 'expand': expand, 'axis': -1})
            if cond_var != in_var:
                try:
                    xa.testing.assert_identical(indata.time, mask.time)
                except AssertionError:
                    mask = mask.assign_coords({'time': indata.time})

            sub_data = indata.where(mask, drop=False)
            sub_data.attrs['Conditional_subselection:'] =\
                (f"Type: Static | Value: {q} | Operator: {operator} | "
                 f"Cond variable: {cond_var} | Resampled cond data: {tres}")
        elif cond_type == 'percentile':
            # Re-chunk in space dim if needed
            if len(cond_data.chunks['time']) > 1:
                cond_data = manage_chunks(cond_data, 'space')
            pctl = xa.apply_ufunc(
                _percentile_func, cond_data[cond_var],
                input_core_dims=[['time']], output_core_dims=[[]],
                dask='parallelized', output_dtypes=[float],
                kwargs={'q': q, 'axis': -1})
            if cond_var != in_var:
                mask = xa.where(ops[operator](cond_data[cond_var], pctl),
                                x=True, y=False)
                sub_data = indata.where(mask, drop=False)
            else:
                sub_data = indata.where(ops[operator](indata, pctl),
                                        drop=False)
            sub_data.attrs['Conditional_subselection:'] =\
                (f"Type: Percentile | Value: {q} | Operator: {operator} | "
                 f"Cond variable: {cond_var} | Resampled cond data: {tres}")
        else:
            raise ValueError(f"Unknown conditional selec type:\t{cond_type}")

    return sub_data


def calculate_statistics(ddict, varlist, stat, pool, chunk_dim,
                         stats_config, regions, fulldomain):
    """ Calculate statistics for all models/obs and variables"""

    stats_data = {}
    for v in varlist:
        stats_data[v] = {}
        for m in ddict[v]:
            print("\tCalculate {} {} for {}\n".format(v, stat, m))

            indata = ddict[v][m]['data']
            stats_data[v][m] = {}

            # Remove additional variables in data set
            if len(indata.data_vars) > 2:
                indata = indata[v].to_dataset()

            # Chunking of data
            indata = manage_chunks(indata, chunk_dim)

            # ----- Resampling of data
            if stats_config[stat]['resample resolution'] is not None:
                resample_args = stats_config[stat]['resample resolution']
                if isinstance(resample_args, dict):
                    if v in resample_args:
                        print("\t\tResampling input data ...\n")
                        indata = data_resampling(indata, resample_args[v])
                    else:
                        pass
                else:
                    print("\t\tResampling input data ...\n")
                    indata = data_resampling(indata, resample_args)

                # Check chunking of data
                indata = manage_chunks(indata, chunk_dim)

            # ----- Conditional analysis; sample data according to condition
            if stats_config[stat]['cond analysis'] is not None:
                cond_dict = stats_config[stat]['cond analysis']
                if v in cond_dict:
                    print("\t\tPerforming conditional sub-selection\n")
                    cond_calc = cond_dict[v]
                    cond_var = cond_calc['cvar']

                    if cond_var == v:
                        cond_data = indata
                    else:
                        # assert vcond in stats_data, msg.format(vcond, v)
                        cond_data = ddict[cond_var][m]['data']

                    # Extract sub selection of data
                    sub_data = conditional_data_selection(
                        cond_calc, cond_var, cond_data, indata, v)

                    # Check chunking of data
                    indata = manage_chunks(sub_data, chunk_dim)

            # ----- Calculate stats
            if regions:
                xd, yd = _space_coords(indata)
                masks = {
                    r: mask_region(
                        indata[xd].values, indata[yd].values, r,
                        cut_data=False) for r in regions}
                # if pool:
                #     mdata = {r: get_masked_data(data, v, masks[r])
                #              for r in regions}

                #     st_mdata = {r: st.calc_statistics(mdata[r], v, stat,
                #                                       stats_config)
                #                 for r in regions}
                if fulldomain:
                    stats_data[v][m]['domain'] = st.calc_statistics(
                        indata, v, stat, stats_config)
                    st_mdata = {r: get_masked_data(
                        stats_data[v][m]['domain'], v, masks[r])
                                for r in regions}
                else:
                    mdata = {r: get_masked_data(indata, v, masks[r])
                             for r in regions}

                    st_mdata = {r: st.calc_statistics(mdata[r], v, stat,
                                                      stats_config)
                                for r in regions}

                stats_data[v][m]['regions'] = st_mdata
            else:
                stats_data[v][m]['domain'] = st.calc_statistics(
                    indata, v, stat, stats_config)

    return stats_data


def get_time_suffix_string(date_type, years=None, months=None,
                           date_start=None, date_end=None):
    """Return time suffix string"""

    month_dict = {
        1: 'J', 2: 'F', 3: 'M', 4: 'A', 5: 'M', 6: 'J', 7: 'J',
        8: 'A', 9: 'S', 10: 'O', 11: 'N', 12: 'D'}
    month_dict_l = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

    if date_type == 'years months':
        if len(months) == 12:
            mstring = 'ANN'
        elif len(months) == 1:
            mstring = month_dict_l[months[0]]
        else:
            if months in ([1, 2, 12], (1, 2, 12)):
                months = (12, 1, 2)
            mstring = ''.join(month_dict[m] for m in months)

        tsuffix_string = f'{years[0]}-{years[1]}_{mstring}'

    elif date_type == 'date interval':
        ds = date_start.replace('-', '')
        de = date_end.replace('-', '')
        tsuffix_string = f'{ds}-{de}'

    return tsuffix_string


def _get_freq(tf):
    """Return temporal frequency (number of time units)"""
    from functools import reduce

    d = [j.isdigit() for j in tf]
    if np.any(d):
        freq = int(reduce((lambda x, y: x+y), [x for x, y in zip(tf, d) if y]))
    else:
        freq = 1

    unit = reduce((lambda x, y: x+y), [x for x, y in zip(tf, d) if not y])

    if unit in ('M', 'Y'):
        freq = freq*30 if unit == 'M' else freq*365
        unit = 'D'
    elif unit[0] == 'Q':
        # Quarterly frequencies - assuming length of 90 days
        freq = 90
        unit = 'D'

    return freq, unit


def _space_dim(ds):
    """
    Return labels for space dimensions in data set.

    Space dimensions in different data sets may have different names. This
    dictionary is created to account for that (to some extent), but in the
    future this might be change/removed. For now, it's hard coded.
    """
    spcdim = {'x': ['lon', 'rlon', 'longitude', 'x', 'X', 'i'],
              'y': ['lat', 'rlat', 'latitude', 'y', 'Y', 'j']}
    xd = [x for x in ds.dims if x in spcdim['x']][0]
    yd = [y for y in ds.dims if y in spcdim['y']][0]

    return xd, yd


def _space_coords(ds):
    """
    Return labels for space coordinates in data set.

    Space coordinates in different data sets may have different names. This
    dictionary is created to account for that (to some extent), but in the
    future this might be change/removed. For now, it's hard coded.
    """
    spccrd = {'x': ['lon', 'longitude', 'x', 'X', 'rlon', 'i'],
              'y': ['lat', 'latitude', 'y', 'Y', 'rlat', 'j']}
    xc = [x for x in spccrd['x'] if x in ds.coords][0]
    yc = [y for y in spccrd['y'] if y in ds.coords][0]

    return xc, yc


def _coords_in_ascending_order(ds, xd, xc, yd, yc):
    """Make sure coordinates in ascending order"""
    if ds[xc].ndim == 1:
        if np.diff(ds[xc])[0] < 0:
            ds = ds.reindex({xd: np.flipud(ds[xd])})
        if np.diff(ds[yc])[0] < 0:
            ds = ds.reindex({yd: np.flipud(ds[yd])})
    elif ds[xc].ndim == 2:
        if np.diff(ds[xc][0, :])[0] < 0:
            ds = ds.reindex({xd: np.flipud(ds[xd])})
        if np.diff(ds[yc][:, 0])[0] < 0:
            ds = ds.reindex({yd: np.flipud(ds[yd])})

    return ds


def save_to_disk(data, label, stat, odir, var, grid, time_suffix, stat_dict,
                 tres, thr='', regs=None, fulldomain=True):
    """Saving data to netcdf files"""

    # Encoding for time variable. EDIT 240826: Often not applicable
    # encoding = {'time': {'dtype': 'i4'}}

    if stat in ('annual cycle', 'seasonal cycle', 'diurnal cycle'):
        tstat = '_' + stat_dict['stat method'].replace(' ', '')
    else:
        tstat = ''
    stat_name = stat.replace(' ', '_')
    if stat in ('diurnal cycle'):
        stat_fn = "{}_{}".format(stat_name, stat_dict['dcycle stat'])
    else:
        stat_fn = stat_name

    if regs:
        for r in regs:
            rn = r.replace(' ', '_')
            data['regions'][r].attrs['Analysed time'] =\
                f"{time_suffix.replace('_', ' ')}"
            fname = '{}_{}_{}_{}{}{}_{}_{}_{}.nc'.format(
                label, stat_fn, var, thr, tres, tstat, rn, grid, time_suffix)
            data['regions'][r].to_netcdf(os.path.join(odir, stat_name, fname))
        if fulldomain:
            fname = '{}_{}_{}_{}{}{}_{}_{}.nc'.format(
                label, stat_fn, var, thr, tres, tstat, grid, time_suffix)
            data['domain'].attrs['Analysed time'] =\
                f"{time_suffix.replace('_', ' ')}"
            data['domain'].to_netcdf(os.path.join(odir, stat_name, fname))
    else:
        fname = '{}_{}_{}_{}{}{}_{}_{}.nc'.format(
            label, stat_fn, var, thr, tres, tstat, grid, time_suffix)
        data['domain'].attrs['Analysed time'] =\
            f"{time_suffix.replace('_', ' ')}"
        data['domain'].to_netcdf(os.path.join(odir, stat_name, fname))


def get_masked_data(data, var, mask):
    """Mask region in data set"""
    mask_in = xa.DataArray(
        np.broadcast_to(mask, data[var].shape), dims=data[var].dims
    )
    return data.where(mask_in, drop=True)

    # Petter Lind 2019-10-29
    # N.B. For masking of large data sets, the code below might be more
    # efficient.
    #
    # def _mask_func(arr, axis=0, lons=None, lats=None, region=''):
    #     iter_3d = arr.shape[axis]
    #     mdata = mask_region(lons, lats, region, arr, iter_3d=iter_3d,
    #                               cut_data=True)[0]
    #     return mdata

    # mask = mask_region(data.lon.values, data.lat.values, reg,
    #                          cut_data=True)

    # imask = mask[0]
    # mask_data = da.map_blocks(_mask_func, data[var].data, dtype=float,
    #                           chunks=(data.chunks['time'],
    #                                   imask.shape[0], imask.shape[1]),
    #                           lons=data.lon.values, lats=data.lat.values,
    #                           region=reg)
    # lon_d = 'x' if mask[1].ndim == 1 else ['y', 'x']
    # lat_d = 'y' if mask[2].ndim == 1 else ['y', 'x']
    # mask_data_out = xa.Dataset(
    #     {var: (['time', 'y', 'x'],  mask_data)},
    #     coords={'lon': (lon_d, mask[1]), 'lat': (lat_d, mask[2]),
    #             'time': data.time.values})
    # return mask_data_out


def manage_chunks(data, chunk_dim):
    """Re-chunk the data in specified dimension."""

    xd, yd = _space_dim(data)
    xsize = data[xd].size
    ysize = data[yd].size

    data = data.unify_chunks()

    # Rule of thumb: chunk size should at least be 1e6 elements
    # http://xarray.pydata.org/en/stable/dask.html#chunking-and-performance
    # Max chunksize just an arbitrary value to assert 'reasonable' chunking
    min_chunksize = 1e6
    max_chunksize = 1e8
    if chunk_dim == 'space':
        chunksize = np.mean(data.chunks[xd]) * np.mean(data.chunks[yd])\
                * data.time.size
        if chunksize < min_chunksize or chunksize > max_chunksize:
            sub_size = np.sqrt(min_chunksize/data.time.size)
            _csize_x = int((xsize/sub_size))
            csize_x = 5 if _csize_x < 5 else _csize_x
            _csize_y = int((ysize/sub_size))
            csize_y = 5 if _csize_y < 5 else _csize_y
            data = data.chunk({'time': -1, xd: csize_x, yd: csize_y})
        else:
            data = data.chunk({'time': -1})
    else:
        chunksize = xsize * ysize * np.mean(data.chunks['time'])
        if chunksize < min_chunksize or chunksize > max_chunksize:
            sub_size = max_chunksize/(xsize*ysize)
            _csize_t = int(data.time.size/sub_size)
            csize_t = 5 if _csize_t < 5 else _csize_t
            data = data.chunk({'time': csize_t, xd: xsize, yd: ysize})
        else:
            data = data.chunk({xd: xsize, yd: ysize})

    return data


def get_time_resolution_string(resample, cdict, v, mod, obs):
    """
    Generate a dictionary with time resolution string for models and obs
    """
    tres_out_obs = None
    plot_tres = ''
    if resample is not None:
        if isinstance(resample, dict):
            if v in resample:
                if resample[v][0] in ('select dates', 'select hours'):
                    tres_out = resample[v][0].replace(' ', '_')
                else:
                    tres_out = "_".join(resample[v][0:2])
                tres_out_obs = tres_out
                # The tres used in plot file names (if plotting)
                plot_tres = f"_{tres_out}"
            else:
                tres_out = cdict['variables'][v]['freq']
        else:
            if resample[0] in ('select dates', 'select hours'):
                tres_out = resample[0].replace(' ', '_')
            else:
                tres_out = "_".join(resample[0:2])
            tres_out_obs = tres_out
            plot_tres = f"_{tres_out}"
    else:
        tres_out = cdict['variables'][v]['freq']

    if isinstance(tres_out, list):
        msg = "The number of input frequencies must match number of models!"
        assert len(tres_out) == len(mod), msg
        tres_dd = {m: tr for m, tr in zip(mod, tres_out)}
    elif isinstance(tres_out, str):
        tres_dd = {m: tres_out for m in mod}
    else:
        msg = (f"Unknown format on time frequency: {tres_out}. "
               f"Should be either string or list.")
        raise ValueError(msg)

    tres_dd.update({'plot tres': plot_tres})

    # Observations
    if obs[0] is not None:
        if tres_out_obs is None:
            tres_out_o = cdict['variables'][v]['obs freq']
            if isinstance(tres_out_o, list):
                msg = ("The number of input frequencies must match "
                       "number of observations!")
                assert len(tres_out_o) == len(obslist), msg
                tres_dd.update({o: tr for o, tr in zip(obslist, tres_out_o)})
            elif isinstance(tres_out_o, str):
                tres_dd.update({o: tres_out_o for o in obslist})
            else:
                msg = (f"Unknown format on obs time frequency: {tres_out_o}. "
                       f"Should be either string or list.")
                raise ValueError(msg)
        else:
            tres_dd.update({o: tres_out_obs for o in obslist})

    return tres_dd


def get_plot_dict(cdict, var, grid_coords, models, obs, tsuffix_dict, tres,
                  img_outdir, stat_outdir, stat):
    """
    Create a dictionary with settings for a validation plot procedure.
    """
    # Settings and meta data
    vconf = get_variable_config(cdict['variables'][var])
    grdnme = grid_coords['target grid'][var]['gridname']
    st = stat.replace(' ', '_')
    if stat == 'diurnal cycle':
        stnm = "{}_{}".format(st, cdict['stats_conf'][stat]['dcycle stat'])
    else:
        stnm = st
    if stat in ('annual cycle', 'seasonal cycle', 'diurnal cycle'):
        tstat = '_' + cdict['stats_conf'][stat]['stat method'].replace(
            ' ', '')
    else:
        tstat = ''
    thrlg = (('thr' in cdict['stats_conf'][stat]) and
             (cdict['stats_conf'][stat]['thr'] is not None) and
             (var in cdict['stats_conf'][stat]['thr']))
    if thrlg:
        thrstr = "thr{}_".format(str(cdict['stats_conf'][stat]['thr'][v]))
    else:
        thrstr = ''

    # Create dictionaries with list of files for models and obs
    _fm_list = {stat: [glob.glob(os.path.join(
        stat_outdir, f'{st}', '{}_{}_{}_{}{}{}_{}_{}.nc'.format(
            m, stnm, var, thrstr, tres[m], tstat, grdnme, tsuffix_dict[m])))
        for m in models]}
    fm_list = {s: [y for x in ll for y in x] for s, ll in _fm_list.items()}

    obs_list = [obs] if not isinstance(obs, list) else obs
    if obs is not None:
        _fo_list = {stat: [glob.glob(os.path.join(
            stat_outdir, f'{st}', '{}_{}_{}_{}{}{}_{}_{}.nc'.format(
                o, stnm, var, thrstr, tres[o], tstat, grdnme,
                tsuffix_dict[o]))) for o in obs_list]}
        fo_list = {s: [y for x in ll for y in x] for s, ll in _fo_list.items()}
    else:
        fo_list = {stat: None}

    # Create a dictionary with settings for plotting procedure
    plot_dict = {
        'mod files': fm_list,
        'obs files': fo_list,
        'models': models,
        'observation': obs,
        'variable': var,
        'time res': tres['plot tres'],
        'stat method': tstat,
        'units': vconf['units'],
        'grid coords': grid_coords,
        'map projection': cdict['map projection'],
        'map extent': cdict['map extent'],
        'map gridlines': cdict['map gridlines'],
        'map config': cdict['map configuration'],
        'map grid setup': cdict['map grid config'],
        'map plot kwargs': cdict['map plot kwargs'],
        'map domain': cdict['map domain'],
        'line grid setup': cdict['line grid setup'],
        'line kwargs': cdict['line kwargs'],
        'regions': cdict['regions'],
        'time suffix dict': tsuffix_dict,
        'img dir': os.path.join(img_outdir, st)
    }

    # If there are regions, create list of files for these as well
    # Then also update plot dictionary
    if cdict['regions'] is not None:
        _fm_listr = {stat: {r:  [glob.glob(os.path.join(
            stat_outdir, f'{st}', '{}_{}_{}_{}{}{}_{}_{}_{}.nc'.format(
                m, stnm, var, thrstr, tres[m], tstat, r.replace(' ', '_'),
                grdnme, tsuffix_dict[m])))
            for m in models] for r in cdict['regions']}}
        fm_listr = {s: {r: [y for x in _fm_listr[s][r] for y in x]
                        for r in _fm_listr[s]} for s in _fm_listr}
        if obs is not None:
            _fo_listr = {stat: {r: [glob.glob(os.path.join(
                stat_outdir, f'{st}', '{}_{}_{}_{}{}{}_{}_{}_{}.nc'.format(
                    o, stnm, var, thrstr, tres[o], tstat, r.replace(' ', '_'),
                    grdnme, tsuffix_dict[o]))) for o in obs_list]
                for r in cdict['regions']}}
            fo_listr = {s: {r: [y for x in _fo_listr[s][r] for y in x]
                            for r in _fo_listr[s]} for s in _fo_listr}
        else:
            fo_listr = {stat: None}

        plot_dict.update({
            'mod reg files': fm_listr,
            'obs reg files': fo_listr,
        })

    return plot_dict


# Read config file
args = set_def_args()
config_file = args.config
if not os.path.isfile(config_file):
    raise ValueError(f"\nConfig file, '{config_file}', does not exist!")
cdict = get_config_settings(config_file)

###############################################
#                                             #
#            0. CREATE DIRECTORIES            #
#                                             #
###############################################


# Create dirs
stat_outdir = os.path.join(cdict['outdir'], 'stats')
stat_names = [s.replace(' ', '_') for s in cdict['requested_stats']]
img_outdir = os.path.join(cdict['outdir'], 'imgs')

if os.path.exists(cdict['outdir']):
    msg = ("\nOutput folder\n\n{}\n\nalready exists!\nDo you want "
           "to overwrite? y/n: ".format(cdict['outdir']))
    overwrite = input(msg)
    if overwrite == 'y':
        [os.makedirs(os.path.join(stat_outdir, t), exist_ok=True)
         for t in stat_names]
        [os.makedirs(os.path.join(img_outdir, t), exist_ok=True)
         for t in stat_names]
    else:
        sys.exit()
else:
    [os.makedirs(os.path.join(stat_outdir, t)) for t in stat_names]
    [os.makedirs(os.path.join(img_outdir, t)) for t in stat_names]


# Set up distributed client
cluster_type = cdict['cluster type']
cluster_kwargs = cdict['cluster kwargs']
nnodes = cdict['nodes']

cluster = cluster_setup(cluster_type, cluster_kwargs, nnodes)
client = Client(cluster)

####################################################
#                                                  #
#       1. OPEN DATA FILES, GRID INTERPOLATION     #
#                                                  #
####################################################

grid_coords = {}
grid_coords['meta data'] = {}
grid_coords['target grid'] = {}
data_dict = {}
time_suffix_dict = {}

print("\n=== READ & PRE-PROCESS DATA ===")

for var in cdict['variables']:
    data_dict[var] = {}
    grid_coords['meta data'][var] = {}
    grid_coords['target grid'][var] = {}
    time_suffix_dict[var] = {}

    var_conf = get_variable_config(cdict['variables'][var])

    tres = cdict['variables'][var]['freq']
    tres = ([tres]*len(cdict['models'].keys())
            if not isinstance(tres, list) else tres)

    print(f"\n\tvariable: {var}  |  input resolution: "
          f"{', '.join(np.unique(tres))}")

    # Model data
    mod_names = []
    mod_scale_factors = var_conf['scale factor']
    mod_scale_factors = ([mod_scale_factors]*len(cdict['models'].keys())
                         if not isinstance(mod_scale_factors, list) else
                         mod_scale_factors)
    mod_offsets = var_conf['offset factor']
    mod_offsets = ([mod_offsets]*len(cdict['models'].keys())
                   if not isinstance(mod_offsets, list) else mod_offsets)
    deacc = var_conf['deacc']
    deacc = ([deacc]*len(cdict['models'].keys())
             if not isinstance(deacc, list) else deacc)
    for (mod_name, settings), tr, scf, ofs, dc\
            in zip(cdict['models'].items(), tres, mod_scale_factors,
                   mod_offsets, deacc):
        mod_names.append(mod_name)
        mod_data = get_mod_data(
            mod_name, settings, tr, var, var_conf['var names'],
            scf, ofs, dc)
        data_dict[var][mod_name] = mod_data

        # Update grid information for plotting purposes
        if mod_name not in grid_coords['meta data'][var]:
            grid_coords['meta data'][var][mod_name] = {}
            get_grid_coords(mod_data['data'],
                            grid_coords['meta data'][var][mod_name])

    # Observational data
    obs_metadata_file = cdict['obs metadata file']
    obs_names = cdict['variables'][var]['obs']
    obs_list = [obs_names] if not isinstance(obs_names, list) else obs_names
    obs_scale_factors = var_conf['obs scale factor']
    obs_scale_factors = ([obs_scale_factors]*len(obs_list)
                         if not isinstance(obs_scale_factors, list) else
                         obs_scale_factors)
    if obs_names is not None:
        for obsname, cfactor in zip(obs_list, obs_scale_factors):
            obs_data = get_obs_data(
                obs_metadata_file, obsname, var, cfactor,
                cdict['obs time dict'])
            data_dict[var][obsname] = obs_data

    # Time period suffix
    for m in mod_names:
        if cdict['models'][m]['date interval start'] is not None:
            time_suffix_dict[var][m] = get_time_suffix_string(
                'date interval',
                date_start=cdict['models'][m]['date interval start'],
                date_end=cdict['models'][m]['date interval end'])
        else:
            time_suffix_dict[var][m] = get_time_suffix_string(
                'years months',
                years=(cdict['models'][m]['start year'],
                       cdict['models'][m]['end year']),
                months=cdict['models'][m]['months'])
    if obs_names is not None:
        if cdict['obs time dict']['date interval start'] is not None:
            time_suffix_dict[var].update({o: get_time_suffix_string(
                'date interval',
                date_start=cdict['obs time dict']['date interval start'],
                date_end=cdict['obs time dict']['date interval end'])
                for o in obs_list})
        else:
            time_suffix_dict[var].update({o: get_time_suffix_string(
                'years months',
                years=(cdict['obs time dict']['start year'],
                       cdict['obs time dict']['end year']),
                months=cdict['obs time dict']['months'])
                for o in obs_list})

    ##################################
    #  1B REMAP DATA TO COMMON GRID  #
    ##################################
    data_interpolation(data_dict[var], var, var_conf, mod_names, obs_list,
                       grid_coords['target grid'][var])

#############################################
#  1C MODIFICATION & CREATION OF VARIABLES  #
#############################################
if cdict['var modification'] is not None:
    # Modification/manipulation of variables
    for new_var, nv_dict in cdict['var modification'].items():
        arglist = list(nv_dict['input'].keys())
        inargs = ",".join(arglist)
        data_dict[new_var] = variabel_modification(data_dict, nv_dict, inargs,
                                                   mod_names, obs_list)

        # Change parameters and dictionaries accordingly
        cdict['variables'][new_var] = \
            cdict['variables'][nv_dict['input'][arglist[0]]]
        time_suffix_dict[new_var] = time_suffix_dict[
            nv_dict['input'][arglist[0]]]
        grid_coords['target grid'][new_var] =\
            grid_coords['target grid'][nv_dict['input'][arglist[0]]]
        grid_coords['meta data'][new_var] =\
            grid_coords['meta data'][nv_dict['input'][arglist[0]]]

        # Remove input variables if 'replace' in config_main.ini is set to True
        if nv_dict['replace']:
            [data_dict.pop(v) for k, v in nv_dict['input'].items()]
            [cdict['variables'].pop(v) for k, v in nv_dict['input'].items()]


###############################################
#                                             #
#          2. CALCULATE STATISTICS            #
#                                             #
###############################################

print("\n=== STATISTICS ===\n")
stats_dict = {}
for stat in cdict['stats_conf']:
    st_var = cdict['stats_conf'][stat]['vars']
    varlist = list(cdict['variables'].keys()) if not st_var \
        else [st_var] if not isinstance(st_var, list) else st_var
    pool = cdict['stats_conf'][stat]['pool data']
    chunk_dim = cdict['stats_conf'][stat]['chunk dimension']
    stats_dict[stat] = calculate_statistics(data_dict, varlist, stat, pool,
                                            chunk_dim, cdict['stats_conf'],
                                            cdict['regions'],
                                            cdict['full domain'])


###############################################
#                                             #
#           3. WRITE DATA TO DISK             #
#                                             #
###############################################

print("\n=== SAVE OUTPUT ===")
tres_str = {}
for stat in cdict['stats_conf']:
    tres_str[stat] = {}
    resample_res = cdict['stats_conf'][stat]['resample resolution']
    for v in stats_dict[stat]:
        # Time resolution
        obs = cdict['variables'][v]['obs']
        obslist = [obs] if not isinstance(obs, list) else obs
        tres_str[stat][v] = get_time_resolution_string(
            resample_res, cdict, v, mod_names, obslist)
        # Threshold
        thrlg = (('thr' in cdict['stats_conf'][stat]) and
                 (cdict['stats_conf'][stat]['thr'] is not None) and
                 (v in cdict['stats_conf'][stat]['thr']))
        if thrlg:
            thrval = str(cdict['stats_conf'][stat]['thr'][v])
            thrstr = "thr{}_".format(thrval)
        else:
            thrstr = ''

        # Grid name
        gridname = grid_coords['target grid'][v]['gridname']

        for m in stats_dict[stat][v]:
            print(f"\n\twriting {m.upper()} - {v} - {stat} to disk ...")
            save_to_disk(stats_dict[stat][v][m], m, stat, stat_outdir, v,
                         gridname, time_suffix_dict[v][m],
                         cdict['stats_conf'][stat], tres_str[stat][v][m],
                         thrstr, cdict['regions'], cdict['full domain'])


###############################################
#                                             #
#                 4. PLOTTING                 #
#                                             #
###############################################

if cdict['validation plot']:
    print('\n=== PLOTTING ===')
    import rcatool.runtime.RCAT_plots as rplot
    statnames = list(stats_dict.keys())
    for sn in statnames:
        for v in stats_dict[sn]:
            plot_dict = get_plot_dict(
                cdict, v, grid_coords, mod_names, cdict['variables'][v]['obs'],
                time_suffix_dict[v], tres_str[sn][v], img_outdir,
                stat_outdir, sn)
            print("\t\n** Plotting: {} for {} **".format(sn, v))
            rplot.plot_main(plot_dict, sn)
