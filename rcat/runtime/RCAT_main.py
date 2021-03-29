#!/usr/bin/env python
# coding: utf-8

# Import modules

import sys
import os
import glob
import xarray as xa
import xesmf as xe
import pandas as pd
from itertools import product
# import dask.array as da
import numpy as np
import re
from dask.distributed import Client
from rcat.utils import ini_reader
from rcat.utils.polygons import mask_region
import rcat.runtime.RCAT_stats as st
import rcat.utils.grids as gr

import warnings
warnings.filterwarnings("ignore")


# Functions
def get_settings(config_file):
    """
    Retrieve information from main configuration file
    """
    conf_dict = ini_reader.get_config_dict(config_file)
    d = {
        'models': conf_dict['MODELS'],
        'obs metadata file': conf_dict['OBS']['metadata file'],
        'obs start year': conf_dict['OBS']['start year'],
        'obs end year': conf_dict['OBS']['end year'],
        'obs months': conf_dict['OBS']['months'],
        'variables': conf_dict['SETTINGS']['variables'],
        'var modification': conf_dict['SETTINGS']['variable modification'],
        'regions': conf_dict['SETTINGS']['regions'],
        'requested_stats': conf_dict['STATISTICS']['stats'],
        'stats_conf': st.mod_stats_config(conf_dict['STATISTICS']['stats']),
        'validation plot': conf_dict['PLOTTING']['validation plot'],
        'map configure': conf_dict['PLOTTING']['map configure'],
        'map grid setup': conf_dict['PLOTTING']['map grid setup'],
        'map kwargs': conf_dict['PLOTTING']['map kwargs'],
        'line grid setup': conf_dict['PLOTTING']['line grid setup'],
        'line kwargs': conf_dict['PLOTTING']['line kwargs'],
        'cluster type': conf_dict['CLUSTER']['cluster type'],
        'nodes': conf_dict['CLUSTER']['nodes'],
        'cluster kwargs': conf_dict['CLUSTER']['cluster kwargs'],
        'outdir': conf_dict['SETTINGS']['output dir'],
    }

    return d


def local_cluster_setup():
    """
    Set up local-pc cluster

    """
    from dask.distributed import LocalCluster
    cluster = LocalCluster(processes=False)
    return cluster


def slurm_cluster_setup(nodes=1, **kwargs):
    """
    Set up SLURM cluster

    Parameters
    ----------
    nodes: int
        Number of nodes to use
    **kwargs:
        Keyword arguments for cluster specifications
    """
    from dask_jobqueue import SLURMCluster
    cluster = SLURMCluster(**kwargs)
    cluster.scale(nodes)
    return cluster


def get_args():
    """
    Read configuration file
    Parameters
    ----------
    -
    Returns
    -------
    Input arguments
    """
    import argparse

    # Configuring argument setup and handling
    parser = argparse.ArgumentParser(
        description='Main script for model/obs validation')
    parser.add_argument('--config', '-c',  metavar='name config file',
                        type=str, help='<Required> Full path to config file',
                        required=True)
    return parser.parse_args()


def get_grid_coords(nc, grid_coords):
    """
    Read model grid coordinates

    Parameters
    ----------
    nc: xarray dataset
    Returns
    -------
    grid_coords: dict
    """

    def _domain(lats, lons, x):
        lons_p = np.r_[lons[x, x:-x], lons[x:-x, -1-x][1:],
                       lons[-1-x, x:-x][::-1][1:], lons[x:-x, x][::-1][1:]]
        lats_p = np.r_[lats[x, x:-x], lats[x:-x, -1-x][1:],
                       lats[-1-x, x:-x][::-1][1:], lats[x:-x, x][::-1][1:]]
        return list(zip(lons_p, lats_p))

    # If lon/lat is 1D; create 2D meshgrid
    lons = nc.lon.values
    lats = nc.lat.values
    lon, lat = np.meshgrid(lons, lats)\
        if lats.ndim == 1 else (lons, lats)

    # Calculate domain mid point if not given
    idx = tuple([np.int(i/2) for i in lat.shape])
    lat0 = lat[idx]
    lon0 = lon[idx]

    grid_coords['lat_0'] = lat0
    grid_coords['lon_0'] = lon0

    gp_bfr = 1
    grid_coords['crnrs'] = [lat[gp_bfr, gp_bfr], lon[gp_bfr, gp_bfr],
                            lat[-gp_bfr, -gp_bfr], lon[-gp_bfr, -gp_bfr]]
    grid_coords['domain'] = _domain(lat, lon, gp_bfr)

    return grid_coords


def get_grids(nc, target_grid, method='bilinear'):
    """
    Get and/or modify the source and target grids for interpolation
    """
    # EDIT 15/11 2019 by Petter Lind
    # There's no consistent way to handle data where lon_bnds/lat_bnds are
    # already available, They may have different formats not all handled by
    # xesmf regridding tool. Thus, for now, calculation of grid corners is
    # always performed when method='conservative'
    if method == 'conservative':
        slon_b, slat_b = gr.fnCellCorners(nc.lon.values, nc.lat.values)
        s_grid = {'lon': nc.lon.values, 'lat': nc.lat.values,
                  'lon_b': slon_b, 'lat_b': slat_b}
        tlon_b, tlat_b = gr.fnCellCorners(target_grid['lon'],
                                          target_grid['lat'])
        t_grid = {'lon': target_grid['lon'], 'lat': target_grid['lat'],
                  'lon_b': tlon_b, 'lat_b': tlat_b}
    else:
        s_grid = {'lon': nc.lon.values, 'lat': nc.lat.values}
        t_grid = target_grid
    if t_grid['lon'].ndim == 1:
        outdims = (t_grid['lat'].size, t_grid['lon'].size)
        outdims_shape = {'lon': ('x',), 'lat': ('y',)}
    else:
        outdims = (t_grid['lon'].shape[0], t_grid['lon'].shape[1])
        outdims_shape = {'lon': ('y', 'x'), 'lat': ('y', 'x')}

    return s_grid, t_grid, outdims, outdims_shape


def regrid_func(dd, v, vconf, mnames, onames, gdict):
    if vconf['regrid'] is None:
        gdict.update(
            {'lon': {m: dd[m]['grid']['lon'] for m in mnames},
             'lat': {m: dd[m]['grid']['lat'] for m in mnames}})
        if None not in onames:
            for obs in onames:
                gdict['lon'].update({obs: dd[obs]['grid']['lon']})
                gdict['lat'].update({obs: dd[obs]['grid']['lat']})
        gridname = 'native_grid'
    else:
        if isinstance(vconf['regrid'], dict):
            target_grid = xa.open_dataset(vconf['regrid']['file'])
            gridname = vconf['regrid']['name']
            gdict.update({'lon': {gridname: target_grid['lon'].values},
                          'lat': {gridname: target_grid['lat'].values}})
            for mod in mnames:
                dd[mod]['data'] = regrid_calc(dd, mod, v, target_grid,
                                              vconf['rgr method'])
            if None not in onames:
                for obs in onames:
                    dd[obs]['data'] = regrid_calc(dd, obs, v, target_grid,
                                                  vconf['rgr method'])
        elif vconf['regrid'] in onames:
            oname = vconf['regrid']
            target_grid = dd[oname]['grid']
            gridname = dd[oname]['gridname']
            gdict.update({'lon': {oname: target_grid['lon']},
                          'lat': {oname: target_grid['lat']}})
            for m in mnames:
                dd[m]['data'] = regrid_calc(dd, m, v, target_grid,
                                            vconf['rgr method'])
            if len(onames) > 1:
                obslist = onames.copy()
                obslist.remove(oname)
                for obs in obslist:
                    dd[obs]['data'] = regrid_calc(dd, obs, v, target_grid,
                                                  vconf['rgr method'])
        elif vconf['regrid'] in mnames:
            mname = vconf['regrid']
            modlist = mnames.copy()
            modlist.remove(mname)
            target_grid = dd[mname]['grid']
            gridname = dd[mname]['gridname']
            gdict.update({'lon': {mname: target_grid['lon']},
                          'lat': {mname: target_grid['lat']}})
            for mod in modlist:
                dd[mod]['data'] = regrid_calc(dd, mod, v, target_grid,
                                              vconf['rgr method'])
            if None not in onames:
                for obs in onames:
                    dd[obs]['data'] = regrid_calc(dd, obs, v, target_grid,
                                                  vconf['rgr method'])
        else:
            raise ValueError(("\n\n\tTarget grid name not found!\n"
                              "Check 'regrid to' option in main config file"))

    gdict.update({'gridname': gridname})

    return dd, gdict


def regrid_calc(data, data_name, var, target_grid, method):
    print("\t\t** Regridding {} data **\n".format(data_name.upper()))

    indata = data[data_name]['data']

    # Get grid info
    sgrid, tgrid, dim_val, dim_shape = get_grids(indata, target_grid, method)

    # Regridding
    regridder = gr.add_matrix_NaNs(xe.Regridder(sgrid, tgrid, method))
    # regridder.clean_weight_file()
    ds_rgr = regridder(indata[var])
    dset = ds_rgr.to_dataset()

    return dset


def resampling(data, v, tresample):
    """
    Resample data to chosen time frequency and resample method.
    """
    from pandas import to_timedelta
    diff = data.time.values[1] - data.time.values[0]
    nsec = to_timedelta(diff).total_seconds()
    tr, fr = _get_freq(tresample[0])
    sec_resample = to_timedelta(tr, fr).total_seconds()
    if nsec != sec_resample:
        data = eval("data.resample(time='{}').{}('time').\
                      dropna('time', 'all')".format(
                          tresample[0], tresample[1]))
    else:
        print("\t\tData is already at target resolution, skipping "
              "resampling ...\n")
    return data


def conditional_data(condition, cvar, cdata, indata):
    """
    Sub-select data conditional on other data
    """
    import operator

    def _get_cond_mask(darr, thr_type, relate, q=95, thr=0,
                       expand=None, axis=0):
        ops = {'>': operator.gt,
               '<': operator.lt,
               '>=': operator.ge,
               '<=': operator.le,
               '=': operator.eq}
        if thr_type == 'percentile':
            mask = np.apply_along_axis(
                # (lambda x: ops[relate](x, np.percentile(x, q=q))),
                (lambda x: x >= np.percentile(x, q=q)),
                axis=axis, arr=darr)
        elif thr_type == 'static':
            mask = np.apply_along_axis((lambda x: ops[relate](x, thr)),
                                       axis=axis, arr=darr)
        if expand is not None:
            mask = np.repeat(mask, expand, axis=axis)

        return mask

    if 'percentile' in condition['condition'][1]:
        inthr = 'percentile'
        q = float(condition['condition'][1].split(' ')[1])
    else:
        inthr = 'static'
        q = condition['condition'][1]
    relate = condition['condition'][0]

    if 'resample resolution' in condition:
        print("\t\tResampling conditional data ...\n")
        tres = condition['resample resolution']
        cdata = resampling(cdata, tres)

    cdata = manage_chunks(cdata, 'space')

    if cdata.time.size != indata.time.size:
        expand = indata.time.size/cdata.time.size
    else:
        expand = None

    _mask = xa.apply_ufunc(
        _get_cond_mask, cdata[cvar], input_core_dims=[['time']],
        output_core_dims=[['time']], dask='parallelized',
        output_dtypes=[float], exclude_dims={'time'},
        dask_gufunc_kwargs={'output_sizes': {'time': indata.time.size}},
        kwargs={'thr_type': inthr, 'relate': relate, 'q': q,
                'expand': expand, 'axis': -1})
    dims = list(_mask.dims)
    mask = _mask.transpose(dims[-1], dims[0], dims[1])
    sub_data = indata.where(mask)

    return sub_data


def calc_stats(ddict, vlist, stat, pool, chunk_dim, stats_config, regions):
    """
    Calculate statistics for variables and models/obs
    """
    st_data = {}
    for v in vlist:
        st_data[v] = {}
        for m in ddict[v]:
            print("\tCalculate {} {} for {}\n".format(v, stat, m))

            if ddict[v][m] is None:
                print("\t* Data not available for model {}, moving on".format(
                    m))
            else:
                indata = ddict[v][m]['data']
                st_data[v][m] = {}
                if len(indata.data_vars) > 2:
                    indata = indata[v].to_dataset()

                # Resampling of data
                if stats_config[stat]['resample resolution'] is not None:
                    print("\t\tResampling input data ...\n")
                    indata = resampling(
                        indata, v, stats_config[stat]['resample resolution'])

                # Check chunking of data
                data = manage_chunks(indata, chunk_dim)

                # Conditional analysis; sample data according to condition
                # (static threshold or percentile).
                if stats_config[stat]['cond analysis'] is not None:
                    cond = stats_config[stat]['cond analysis']
                    vcond, ccond = zip(*cond.items())
                    vcond = vcond[0]
                    ccond = ccond[0]
                    # assert vcond in st_data, msg.format(vcond, v)
                    print("\t\tPerforming conditional sub-selection\n")
                    frq = [v for v, k in ddict.items()
                           if vcond in list(k.keys())][0]
                    sub_data = conditional_data(
                        ccond, vcond, ddict[frq][vcond][m]['data'], indata)

                    data = manage_chunks(sub_data, chunk_dim)

                # Calculate stats
                st_data[v][m]['domain'] = st.calc_statistics(data, v, stat,
                                                             stats_config)

                if regions:
                    masks = {
                        r: mask_region(
                            data.lon.values, data.lat.values, r,
                            cut_data=False) for r in regions}
                    if pool:
                        mdata = {r: get_masked_data(data, v, masks[r])
                                 for r in regions}

                        st_mdata = {r: st.calc_statistics(mdata[r], v, stat,
                                                          stats_config)
                                    for r in regions}
                    else:
                        st_mdata = {r: get_masked_data(st_data[v][m]['domain'],
                                                       v, masks[r])
                                    for r in regions}

                    st_data[v][m]['regions'] = st_mdata

    return st_data


def get_month_string(mlist):
    d = {1: 'J', 2: 'F', 3: 'M', 4: 'A', 5: 'M', 6: 'J', 7: 'J', 8: 'A',
         9: 'S', 10: 'O', 11: 'N', 12: 'D'}
    dl = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul',
          8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    if len(mlist) == 12:
        mstring = 'ANN'
    elif len(mlist) == 1:
        mstring = dl[mlist[0]]
    else:
        if mlist in ([1, 2, 12], (1, 2, 12)):
            mlist = (12, 1, 2)
        mstring = ''.join(d[m] for m in mlist)
    return mstring


def _get_freq(tf):
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
        # Quarterly frequencies - assuming average 90 days
        freq = 90
        unit = 'D'

    return freq, unit


def _space_dim(ds):
    """
    Return labels for space dimensions in data set.

    Space dimensions in different observations have different names. This
    dictionary is created to account for that (to some extent), but in the
    future this might be change/removed. For now, it's hard coded.
    """
    spcdim = {'x': ['x', 'X', 'lon', 'rlon'],
              'y': ['y', 'Y', 'lat', 'rlat']}
    xd = [x for x in ds.dims if x in spcdim['x']][0]
    yd = [y for y in ds.dims if y in spcdim['y']][0]

    return xd, yd


def save_to_disk(data, label, stat, odir, var, grid, sy, ey, tsuffix,
                 stat_dict, tres, thr='', regs=None):
    """
    Saving statistical data to netcdf files
    """
    # encoding = {'lat': {'dtype': 'float32', '_FillValue': False},
    #             'lon': {'dtype': 'float32', '_FillValue': False},
    #             var: {'dtype': 'float32', '_FillValue': 1.e20}
    #             }
    if stat in ('annual cycle', 'seasonal cycle', 'diurnal cycle'):
        _tstat = stat_dict['stat method'].partition(' ')[0]
        tstat = '_' + re.sub(r'[^a-zA-Z0-9 \.\n]', '', _tstat).\
                replace(' ', '_')
    else:
        tstat = ''
    stat_name = stat.replace(' ', '_')
    if stat in ('diurnal cycle'):
        stat_fn = "{}_{}".format(stat_name, stat_dict['dcycle stat'])
    else:
        stat_fn = stat_name

    if regs is not None:
        for r in regs:
            rn = r.replace(' ', '_')
            data['regions'][r].attrs['Analysed time'] =\
                "{}-{} | {}".format(sy, ey, tsuffix)
            fname = '{}_{}_{}_{}{}{}_{}_{}_{}-{}_{}.nc'.format(
                label, stat_fn, var, thr, tres, tstat, rn, grid, sy, ey,
                tsuffix)
            data['regions'][r].to_netcdf(os.path.join(odir, stat_name, fname))
    fname = '{}_{}_{}_{}{}{}_{}_{}-{}_{}.nc'.format(
        label, stat_fn, var, thr, tres, tstat, grid, sy, ey, tsuffix)
    data['domain'].attrs['Analysed time'] = "{}-{} | {}".format(sy, ey,
                                                                tsuffix)
    data['domain'].to_netcdf(os.path.join(odir, stat_name, fname))


def get_masked_data(data, var, mask):
    """
    Mask region
    """
    mask_in = xa.DataArray(np.broadcast_to(mask, data[var].shape),
                           dims=data[var].dims)
    return data.where(mask_in, drop=True)

    # Petter Lind 2019-10-29
    # N.B. For masking of large data sets, the code below might be a more
    # viable choice.
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
    # out = xa.Dataset(
    #     {var: (['time', 'y', 'x'],  mask_data)},
    #     coords={'lon': (lon_d, mask[1]), 'lat': (lat_d, mask[2]),
    #             'time': data.time.values})
    # return out


def manage_chunks(data, chunk_dim):
    """
    Re-chunk the data in specified dimension.
    """
    xd, yd = _space_dim(data)
    xsize = data[xd].size
    ysize = data[yd].size

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
            csize_x = int((xsize/sub_size))
            csize_y = int((ysize/sub_size))
            data = data.chunk({'time': -1, xd: csize_x, yd: csize_y})
        else:
            data = data.chunk({'time': -1})
    else:
        chunksize = xsize * ysize * np.mean(data.chunks['time'])
        if chunksize < min_chunksize or chunksize > max_chunksize:
            sub_size = min_chunksize/(xsize*ysize)
            csize_t = int(data.time.size/sub_size)
            data = data.chunk({'time': csize_t, xd: xsize, yd: ysize})
        else:
            data = data.chunk({xd: xsize, yd: ysize})

    return data


def get_variable_config(var_config, var):
    """
    Retrieve configuration info for variable var as defined in main
    configuration file,
    """
    vdict = {
        'var names': var_config['var names'],
        'input resolution': var_config['freq'],
        'units': var_config['units'],
        'scale factor': var_config['scale factor'],
        'obs scale factor': var_config['obs scale factor'] if
        'obs scale factor' in var_config else None,
        'deacc': var_config['accumulated'],
        'regrid': var_config['regrid to'] if 'regrid to' in
        var_config else None,
        'rgr method': var_config['regrid method'] if 'regrid method' in
        var_config else None,
    }
    return vdict


def variabel_modification(dd, nv_dd, funargs, mlist, olist):
    """
    Create new or modify existing variables.
    """
    from inspect import signature
    out_dd = {}
    expression = nv_dd['expression']
    func = eval(f"lambda {funargs}: {expression}")
    args = list(signature(func).parameters)
    _modlist = nv_dd['models']
    _obslist = nv_dd['obs']
    if _modlist is not None:
        modlist = _modlist if isinstance(_modlist, list) else mlist \
                if _modlist == 'all' else [_modlist]
        for m in modlist:
            input_data = [dd[nv_dd['input'][a]][m]['data'][nv_dd['input'][a]]
                          for a in args]
            _new_data = xa.apply_ufunc(
                func, *input_data, input_core_dims=[[]]*len(input_data),
                dask='parallelized', output_dtypes=[float],
            )
            new_data = _new_data.to_dataset(name=new_var)
            out_dd[m] = {'data': new_data}
    if _obslist is not None:
        obslist = _obslist if isinstance(_obslist, list) else olist \
                if _obslist == 'all' else [_obslist]
        for o in obslist:
            input_data = {k: dd[v][o]['data'][v]
                          for k, v in nv_dd['input'].items()}
            _new_data = func(**input_data)
            new_data = _new_data.to_dataset(name=new_var)
            out_dd[o] = {'data': new_data}

    return out_dd


def get_mod_data(model, mconf, tres, var, vnames, cfactor, deacc):
    """
    Open model data files where file path is dependent on time resolution tres.
    """
    import re

    print("\t-- Opening {} files\n".format(model.upper()))

    fyear = mconf['start year']
    lyear = mconf['end year']
    months = mconf['months']
    date_list = ["{}{:02d}".format(yy, mm) for yy, mm in product(
        range(fyear, lyear+1), months)]

    file_path = os.path.join(mconf['fpath'], f'{tres}/{var}/{var}_*.nc')
    _flist = glob.glob(file_path)

    errmsg = (f"Could not find any files at specified location:\n{file_path} "
              "\n\nexiting ...")
    if not _flist:
        print("\t\n{}".format(errmsg))
        sys.exit()

    _file_dates = [re.split('-|_', f.rsplit('.')[-2])[-2:] for f in _flist]
    file_dates = [(d[0][:6], d[1][:6]) for d in _file_dates]
    fidx = [np.where([d[0] <= date <= d[1] for d in file_dates])[0]
            for date in date_list]
    flist = [_flist[i] for i in np.unique(np.hstack(fidx))]
    flist.sort()

    if np.unique([len(f) for f in flist]).size > 1:
        flngth = np.unique([len(f) for f in flist])
        flist = [f for f in flist if len(f) == flngth[0]]

    # Chunk sizes in temporal and spatial dimensions
    ch_t = mconf['chunks_time']
    ch_x = mconf['chunks_x']
    ch_y = mconf['chunks_y']

    if deacc:
        _mdata = xa.open_mfdataset(
            flist, combine='by_coords', parallel=True,  # engine='h5netcdf',
            chunks={**ch_t, **ch_x, **ch_y},
            preprocess=(lambda arr: arr.diff('time')))
    else:
        _mdata = xa.open_mfdataset(
            flist, combine='by_coords', parallel=True,  # engine='h5netcdf',
            chunks={**ch_t, **ch_x, **ch_y})

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

    # Extract years
    mdata = _mdata.where(((_mdata.time.dt.year >= fyear) &
                          (_mdata.time.dt.year <= lyear) &
                          (np.isin(_mdata.time.dt.month, months))), drop=True)

    # Remove height dim
    if 'height' in mdata.dims:
        mdata = mdata.squeeze()

    # Rename variable if not consistent with name in config_main.ini
    if vnames is not None:
        if model in vnames:
            mdata = mdata.rename({vnames[model]['vname']: var})

    if cfactor is not None:
        mdata[var] *= cfactor

    # Model grid
    gridname = 'grid_{}'.format(mconf['grid name'])

    # - Unrotate grid if needed
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
        grid = {'lon': mdata.lon.values, 'lat': mdata.lat.values}

    outdata = {'data': mdata.unify_chunks(),
               'grid': grid, 'gridname': gridname}

    return outdata


def get_obs_data(metadata_file, obs, var, cfactor, sy, ey, mns):
    """
    Open obs data files.
    """
    from importlib.machinery import SourceFileLoader
    obs_meta = SourceFileLoader("obs_meta", metadata_file).load_module()
    # obs_dict = obs_meta.obs_data()

    sdate = '{}{:02d}'.format(sy, np.min(mns))
    edate = '{}{:02d}'.format(ey, np.max(mns))

    print("\t-- Opening {} files\n".format(obs.upper()))

    # Open obs files
    obs_flist = obs_meta.get_file_list(var, obs, sdate, edate)

    emsg = ("Could not find any {} files at specified location"
            "\n\nexiting ...".format(obs.upper()))
    if not obs_flist:
        print("\t\n{}".format(emsg))
        sys.exit()

    date_list = ["{}{:02d}".format(yy, mm) for yy, mm in product(
        range(sy, ey+1), mns)]
    _file_dates = [re.split('-|_', f.rsplit('.')[-2])[-2:] for f in obs_flist]
    file_dates = [(d[0][:6], d[1][:6]) for d in _file_dates]
    fidx = [np.where([d[0] <= date <= d[1] for d in file_dates])[0]
            for date in date_list]
    flist = [obs_flist[i] for i in np.unique(np.hstack(fidx))]
    flist.sort()
    f_obs = xa.open_mfdataset(flist, combine='by_coords', parallel=True)
    # engine='h5netcdf')

    # Extract years
    obs_data = f_obs.where(((f_obs.time.dt.year >= sy) &
                            (f_obs.time.dt.year <= ey) &
                            (np.isin(f_obs.time.dt.month, mns))),
                           drop=True)
    # Scale data
    if cfactor is not None:
        obs_data[var] *= cfactor

    lons = obs_data.lon.values
    lats = obs_data.lat.values

    # Labels for space dimensions
    xd, yd = _space_dim(obs_data)

    # Make sure lon/lat elements are in ascending order
    if lons.ndim == 1:
        if np.diff(lons)[0] < 0:
            lons = lons[::-1]
            obs_data = obs_data.reindex({xd: obs_data[xd][::-1]})
        if np.diff(lats)[0] < 0:
            lats = lats[::-1]
            obs_data = obs_data.reindex({yd: obs_data[yd][::-1]})
    elif lons.ndim == 2:
        if np.diff(lons[0, :])[0] < 0:
            lons = np.flipud(lons)
            obs_data = obs_data.reindex({xd: np.flipud(obs_data[xd])})
        if np.diff(lats[:, 0])[0] < 0:
            lats = np.flipud(lats)
            obs_data = obs_data.reindex({yd: np.flipud(obs_data[yd])})

    grid = {'lon': lons, 'lat': lats}
    gridname = 'grid_{}'.format(obs.upper())

    outdata = {'data': obs_data, 'grid': grid, 'gridname': gridname}
    return outdata


def get_plot_dict(cdict, var, grid_coords, models, obs, yrs_d, mon_d, tres,
                  img_outdir, stat_outdir, stat):
    """
    Create a dictionary with settings for a validation plot procedure.
    """
    # Get some settings and meta data
    vconf = get_variable_config(cdict['variables'][var], var)
    grdnme = grid_coords['target grid'][var]['gridname']
    st = stat.replace(' ', '_')
    if stat == 'diurnal cycle':
        stnm = "{}_{}".format(st, cdict['stats_conf'][stat]['dcycle stat'])
    else:
        stnm = st
    if stat in ('annual cycle', 'seasonal cycle', 'diurnal cycle'):
        tstat = '_' + cdict['stats_conf'][stat]['stat method'].replace(
            ' ', '_')
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
        stat_outdir, '{}'.format(st), '{}_{}_{}_{}{}{}_{}_{}-{}_{}.nc'.format(
            m, stnm, var, thrstr, tres, tstat, grdnme, yrs_d[m][0],
            yrs_d[m][1], get_month_string(mon_d[m])))) for m in models]}
    fm_list = {s: [y for x in ll for y in x] for s, ll in _fm_list.items()}

    obs_list = [obs] if not isinstance(obs, list) else obs
    if obs is not None:
        _fo_list = {stat: [glob.glob(os.path.join(
            stat_outdir, '{}'.format(st),
            '{}_{}_{}_{}{}{}_{}_{}-{}_{}.nc'.format(
                o, stnm, var, thrstr, tres, tstat, grdnme,
                yrs_d[o][0], yrs_d[o][1], get_month_string(mon_d[o]))))
            for o in obs_list]}
        fo_list = {s: [y for x in ll for y in x] for s, ll in _fo_list.items()}
    else:
        fo_list = {stat: None}

    sy = np.unique([n['start year'] for m, n in cdict['models'].items()])
    ey = np.unique([n['end year'] for m, n in cdict['models'].items()])
    if sy.size > 1:
        years = {'T1': (sy[0], ey[0]), 'T2': (sy[1], ey[1])}
    else:
        years = (sy[0], ey[0])

    # Create a dictionary with settings for plotting procedure
    plot_dict = {
        'mod files': fm_list,
        'obs files': fo_list,
        'models': models,
        'observation': obs,
        'variable': var,
        'time res': tres,
        'stat method': tstat,
        'units': vconf['units'],
        'grid coords': grid_coords,
        'map configure': cdict['map configure'],
        'map grid setup': cdict['map grid setup'],
        'map kwargs': cdict['map kwargs'],
        'line grid setup': cdict['line grid setup'],
        'line kwargs': cdict['line kwargs'],
        'regions': cdict['regions'],
        'years': years,
        'img dir': os.path.join(img_outdir, st)
    }

    # If there are regions, create list of files for these  as well
    # Then also update plot dictionary
    if cdict['regions'] is not None:
        _fm_listr = {stat: {r:  [glob.glob(os.path.join(
            stat_outdir, '{}'.format(st),
            '{}_{}_{}_{}{}{}_{}_{}_{}-{}_{}.nc'.format(
                m, stnm, var, thrstr, tres, tstat, r.replace(' ', '_'), grdnme,
                yrs_d[m][0], yrs_d[m][1], get_month_string(mon_d[m]))))
            for m in models] for r in cdict['regions']}}
        fm_listr = {s: {r: [y for x in _fm_listr[s][r] for y in x]
                        for r in _fm_listr[s]} for s in _fm_listr}
        if obs is not None:
            _fo_listr = {stat: {r: [glob.glob(os.path.join(
                stat_outdir, '{}'.format(st),
                '{}_{}_{}_{}{}{}_{}_{}_{}-{}_{}.nc'.format(
                    o, stnm, var, thrstr, tres, tstat, r.replace(' ', '_'),
                    grdnme, yrs_d[o][0], yrs_d[o][1],
                    get_month_string(mon_d[o])))) for o in obs_list]
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


# Configuration
args = get_args()

# Read config file
config_file = args.config
cdict = get_settings(config_file)

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
if cdict['cluster type'] == 'local':
    cluster = local_cluster_setup()
    client = Client(cluster)
elif cdict['cluster type'] == 'slurm':
    nnodes = cdict['nodes']
    sl_kwargs = cdict['cluster kwargs']
    cluster = slurm_cluster_setup(nodes=nnodes, **sl_kwargs)
    client = Client(cluster)
else:
    print("\n\tCluster type not implemented! Exiting..")
    sys.exit()

###############################################
#                                             #
#       1. OPEN DATA FILES AND REMAPPING      #
#                                             #
###############################################


# Loop over time resolution of input data
grid_coords = {}
grid_coords['meta data'] = {}
grid_coords['target grid'] = {}
data_dict = {}
month_dd = {}
year_dd = {}

print("\n=== READ & PRE-PROCESS DATA ===")
for var in cdict['variables']:
    data_dict[var] = {}
    var_conf = get_variable_config(cdict['variables'][var], var)

    tres = cdict['variables'][var]['freq']
    print("\n\tvariable: {}  |  input resolution: {}".format(var, tres))

    grid_coords['meta data'][var] = {}
    grid_coords['target grid'][var] = {}

    obs_metadata_file = cdict['obs metadata file']
    obs_name = cdict['variables'][var]['obs']
    obs_list = [obs_name] if not isinstance(obs_name, list) else obs_name
    obs_scf = var_conf['obs scale factor']
    obs_scf = ([obs_scf]*len(obs_list)
               if not isinstance(obs_scf, list) else obs_scf)
    if obs_name is not None:
        for oname, cfactor in zip(obs_list, obs_scf):
            obs_data = get_obs_data(
                obs_metadata_file, oname, var, cfactor,
                cdict['obs start year'],
                cdict['obs end year'], cdict['obs months'])
            data_dict[var][oname] = obs_data

    mod_names = []
    mod_scf = var_conf['scale factor']
    mod_scf = ([mod_scf]*len(cdict['models'].keys())
               if not isinstance(mod_scf, list) else mod_scf)
    for (mod_name, settings), scf in zip(cdict['models'].items(), mod_scf):
        mod_names.append(mod_name)
        mod_data = get_mod_data(
            mod_name, settings, tres, var, var_conf['var names'],
            scf, var_conf['deacc'])
        data_dict[var][mod_name] = mod_data

        # Update grid information for plotting purposes
        if mod_name not in grid_coords['meta data'][var]:
            grid_coords['meta data'][var][mod_name] = {}
            get_grid_coords(mod_data['data'],
                            grid_coords['meta data'][var][mod_name])

    month_dd[var] = {m: cdict['models'][m]['months'] for m in mod_names}
    year_dd[var] = {m: (cdict['models'][m]['start year'],
                        cdict['models'][m]['end year']) for m in mod_names}
    if obs_list is not None:
        month_dd[var].update({o: cdict['obs months'] for o in obs_list})
        year_dd[var].update({o: (cdict['obs start year'],
                                 cdict['obs end year']) for o in obs_list})

    ##################################
    #  1B REMAP DATA TO COMMON GRID  #
    ##################################
    regrid_func(data_dict[var], var, var_conf, mod_names, obs_list,
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
        month_dd[new_var] = month_dd[nv_dict['input'][arglist[0]]]
        year_dd[new_var] = year_dd[nv_dict['input'][arglist[0]]]
        grid_coords['target grid'][new_var] =\
            grid_coords['target grid'][nv_dict['input'][arglist[0]]]

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
    vrs = cdict['stats_conf'][stat]['vars']
    varlist = list(cdict['variables'].keys()) if not vrs \
        else [vrs] if not isinstance(vrs, list) else vrs
    pool = cdict['stats_conf'][stat]['pool data']
    chunk_dim = cdict['stats_conf'][stat]['chunk dimension']
    stats_dict[stat] = calc_stats(data_dict, varlist, stat, pool,
                                  chunk_dim, cdict['stats_conf'],
                                  cdict['regions'])


###############################################
#                                             #
#           3. WRITE DATA TO DISK             #
#                                             #
###############################################

print("\n=== SAVE OUTPUT ===")
tres_str = {}
for stat in cdict['stats_conf']:
    tres_str[stat] = {}
    for v in stats_dict[stat]:
        thrlg = (('thr' in cdict['stats_conf'][stat]) and
                 (cdict['stats_conf'][stat]['thr'] is not None) and
                 (v in cdict['stats_conf'][stat]['thr']))
        if thrlg:
            thrval = str(cdict['stats_conf'][stat]['thr'][v])
            thrstr = "thr{}_".format(thrval)
        else:
            thrstr = ''
        if cdict['stats_conf'][stat]['resample resolution'] is not None:
            res_tres = cdict['stats_conf'][stat]['resample resolution']
            tres_str[stat][v] = "_".join(res_tres)
        else:
            tres_str[stat][v] = cdict['variables'][v]['freq']
        gridname = grid_coords['target grid'][v]['gridname']
        for m in stats_dict[stat][v]:
            print(f"\n\twriting {m.upper()} - {v} - {stat} to disk ...")
            time_suffix = get_month_string(month_dd[v][m])
            st_data = stats_dict[stat][v][m]
            save_to_disk(st_data, m, stat, stat_outdir, v, gridname,
                         year_dd[v][m][0], year_dd[v][m][1], time_suffix,
                         cdict['stats_conf'][stat], tres_str[stat][v], thrstr,
                         cdict['regions'])


###############################################
#                                             #
#                 4. PLOTTING                 #
#                                             #
###############################################

if cdict['validation plot']:
    print('\n=== PLOTTING ===')
    import rcat.runtime.RCAT_plots as rplot
    statnames = list(stats_dict.keys())
    for sn in statnames:
        for v in stats_dict[sn]:
            plot_dict = get_plot_dict(cdict, v, grid_coords, mod_names,
                                      cdict['variables'][v]['obs'], year_dd[v],
                                      month_dd[v], tres_str[sn][v], img_outdir,
                                      stat_outdir, sn)
            print("\t\n** Plotting: {} for {} **".format(sn, v))
            rplot.plot_main(plot_dict, sn)
