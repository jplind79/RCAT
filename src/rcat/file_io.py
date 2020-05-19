"""
This module provides functions representing tools to read and write
NetCDF files.

@author Petter Lind
@date   2014-10-20
"""

import sys
import os
import numpy as np
from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/
from datetime import datetime, timedelta
from fractions import Fraction


def ncdump(nc_fid, verb=True):
    '''ncdump outputs dimensions, variables and their attribute information.
    The information is similar to that of NCAR's ncdump utility.
    ncdump requires a valid instance of Dataset.

    Parameters
    ----------
        nc_fid: netCDF4.Dataset
            A netCDF4 dateset object
        verb: Boolean
            whether or not nc_attrs, nc_dims, and nc_vars are printed

    Returns
    -------
        nc_attrs: list
                A Python list of the NetCDF file global attributes
        nc_dims: list
                A Python list of the NetCDF file dimensions
        nc_vars: list
                A Python list of the NetCDF file variables

    '''
    def print_ncattr(key):
        """
        Prints the NetCDF file attributes for a given key

        Parameters
        ----------
        key : unicode
            a valid netCDF4.Dataset.variables key
        """
        try:
            print("\t\ttype:", repr(nc_fid.variables[key].dtype))
            for ncattr in nc_fid.variables[key].ncattrs():
                print('\t\t%s:' % ncattr,
                      repr(nc_fid.variables[key].getncattr(ncattr)))
        except KeyError:
            print("\t\tWARNING: %s does not contain variable attributes" % key)

    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()
    if verb:
        print("NetCDF Global Attributes:")
        for nc_attr in nc_attrs:
            print('\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr)))
    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    # Dimension shape information.
    if verb:
        print("NetCDF dimension information:")
        for dim in nc_dims:
            print("\tName:", dim)
            print("\t\tsize:", len(nc_fid.dimensions[dim]))
            print_ncattr(dim)
    # Variable information.
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    if verb:
        print("NetCDF variable information:")
        for var in nc_vars:
            if var not in nc_dims:
                print('\tName:', var)
                print("\t\tdimensions:", nc_fid.variables[var].dimensions)
                print("\t\tsize:", nc_fid.variables[var].size)
                print_ncattr(var)
    return nc_attrs, nc_dims, nc_vars


def openFile(filename):
    """
    Function to open netcdf file.
    filename:  string with full path to file
    """
    try:
        return Dataset(filename, "r")
    except Exception:
        print("*** Error ***")
        print("Found no file: " + filename)
        sys.exit()


def getDimensions(nc, close=False):
    """ Function to retrieve the dimensions of a netcdf file
        nc: Netcdf object opened with function "openFile"
        close: set True if you want the file to be closed after retrieval.
        Returns lons and lats, time as well as gridsize Nx,Ny """
    try:
        Nx = len(nc.dimensions['x'])
        Ny = len(nc.dimensions['y'])
    except:
        print("File does not have 'x' and 'y' \
        dimensions. Returning None.")
        Nx = None
        Ny = None
    try:
        time = nc.variables['time'][:]
    except:
        print("File does not have a 'time'"
              "variable. Returning None.")
        time = None
    try:
        lons = nc.variables['lon'][:]
        lats = nc.variables['lat'][:]
    except:
        try:
            lons = nc.variables['longitude'][:]
            lats = nc.variables['latitude'][:]
        except:
            print("*** Error ***")
            print("Could not extract longitudes/latitudes.")
            print("Check that file contains these variables"
                  "and that they have standard names")
            sys.exit()

    if close:
        nc.close()
    return Nx, Ny, lons, lats, time


def getParams(nc, params, close=False):
    """ Function to retrieve variables from a netcdf file
        nc: Netcdf object opened with function "openFile"
        params: A list of strings with the parameters to be retrieved
        close: set True if you want the file to be closed after retrieval.
        Returns a list with the given parameters. """

    # Make sure params is a list
    if type(params) != list:
        params = [params]

    varsOut = []
    for vv in range(len(params)):
        try:
            varsOut.append(nc.variables[params[vv]][:])
        except Exception:
            print("*** Error ***")
            print("Variable " + params[vv] + " not found in file!")
            sys.exit()

    if close:
        nc.close()

    return np.array(varsOut).squeeze()


def fracday2datetime(tdata):
    """
    Takes an array of dates given in %Y%m%d.%f format and returns a
    corresponding datetime object
    """
    dates = [datetime.strptime(str(i).split(".")[0], "%Y%m%d").date()
             for i in tdata]
    frac_day = [i - np.floor(i) for i in tdata]
    ratios = [(Fraction(i).limit_denominator().numerator,
               Fraction(i).limit_denominator().denominator) for i in frac_day]
    times = [datetime.strptime(
        str(timedelta(seconds=timedelta(days=i[0]/i[1]).total_seconds())),
        '%H:%M:%S').time() for i in ratios]

    date_times = [datetime.combine(d, t) for d, t in zip(dates, times)]

    return date_times


def write2netcdf(filename, filedir, dim, variables, global_attr=None,
                 nc_format='NETCDF4', compress=False, complevel=4):
    """
    Opens a new NetCDF file to write the input data to. For nc_format,
    you can choose from 'NETCDF3_CLASSIC', 'NETCDF3_64BIT',
    'NETCDF4_CLASSIC', and 'NETCDF4' (default)

    Parameters:
        filename: str
            name of netcdf file to write to
        filedir: str
            directory path to put the file
        dim: dict
            dimensions to be used
        variables: dict
            variables with their values and attributes
        global_attr: dict
            global attributes (optional)
        nc_format: str
            Specify netCDF format
        compress: boolean
            Whether to compress (using 'zlib=True' in the write call).
        complevel: int
            An integer between 1-9 representing the degree of compression to be
            used.

    The dictionaries should be structured as described by the examples
    below:

        dims_dict = {}
        dims_dict['x'] = 154
        dims_dict['y'] = 192
        dims_dict['nv'] = 4
        dims_dict['time'] = None

        vars_dict = {}
        vars_dict = {'lon': {'values': lons, 'dims': ('y', 'x'),
                             'attributes': {'long_name': 'longitude',
                                            'standard_name': 'longitude',
                                            'units': 'degrees_east',
                                            '_CoordinateAxisType': 'Lon'}},
                     'lat': {'values': lats, 'dims': ('y', 'x'),
                             'attributes': {'long_name': 'latitude',
                                            'standard_name': 'latitude',
                                            'units': 'degrees_north',
                                            '_CoordinateAxisType': 'Lat'}},
                     'pr': {'values': pr, 'dims': ('time', 'y', 'x'),
                            'attributes': {'long_name': 'precipitation',
                                        'standard_name': 'precipitation flux',
                                           'units': 'kg m-2 s-1',
                                           'coordinates': 'lon lat',
                                           '_FillValue': -9999.}}}

        glob_attr = {'description': 'some description of file',
                     'history': 'Created ' + time.ctime(time.time()),
                     'experiment': 'Fractions Skill Score analysis',
                     'contact': 'petter.lind@smhi.se',
                     'references': 'http://journals.ametsoc.org/doi/abs/\
                                    10.1175/2007MWR2123.1'}

    """

    # Open file in write mode
    nc = Dataset(os.path.join(filedir, filename), 'w', format=nc_format)

    # Set global attributes
    nc.setncatts(global_attr)

    # Create dimensions
    for d, val in dim.items():
        nc.createDimension(d, val)

    # Create variables, assign values and append attributes
    for var in variables:
        var_tmp = nc.createVariable(var, 'f8', variables[var]['dims'],
                                    zlib=compress, complevel=complevel)
        var_tmp.setncatts(variables[var]['attributes'])
        indata = np.asarray(variables[var]['values'])
        nc.variables[var][:] = indata

    nc.close()  # close the new file
