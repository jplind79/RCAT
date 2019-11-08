#!/usr/bin/env python

"""
Functions to remap data given source and target grids
Some utilities use python tool xESMF (which is dependent
on ESMF and ESMPy, which needs to be installed).

Author: Petter Lind (contributions from A. Prein)
Date: 2019-01-30
"""
import numpy as np
import logging


def fnCellCorners(rgrLon, rgrLat):
    '''
    File name: fnCellBoundaries
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 20.03.2015
    Date last modified: 20.03.2015

    ##############################################################
    Purpose:

    Estimate the cell boundaries from the cell location of regular grids

    returns: rgrLonBND & rgrLatBND --> arrays of dimension [nlon,nlat]
    containing the cell boundaries of each gridcell in rgrlon and rgrlat
    '''
    # from ipdb import set_trace as stop
    logging.debug('fnCellCorners')
    rgLonSize = np.array(rgrLon).shape
    rgLatSize = np.array(rgrLat).shape
    if len(rgLonSize) == 1:
        rgrLat = np.broadcast_to(rgrLat, (rgLonSize[0],
                                          rgLatSize[0])).swapaxes(0, 1)
        rgrLon = np.broadcast_to(rgrLon, (rgLatSize[0], rgLonSize[0]))
    rgiSize = np.array(rgrLon).shape
    rgrLonBND = np.empty((rgiSize[0]+1, rgiSize[1]+1,))
    rgrLonBND[:] = np.NAN
    rgrLatBND = np.empty((rgiSize[0]+1, rgiSize[1]+1,))
    rgrLatBND[:] = np.NAN

    for lo in range(rgiSize[0]+1):
        for la in range(rgiSize[1]+1):
            if lo < rgiSize[0]-1 and la < rgiSize[1]-1:
                # All points except at the boundaries
                rgrLonBND[lo, la] = rgrLon[lo, la] -\
                    (rgrLon[lo+1, la+1]-rgrLon[lo, la])/2
                rgrLatBND[lo, la] = rgrLat[lo, la] -\
                    (rgrLat[lo+1, la+1]-rgrLat[lo, la])/2
            elif lo >= rgiSize[0]-1 and la < rgiSize[1]-1:
                # reight boundary second last row
                rgrLonBND[lo, la] = rgrLon[lo-1, la] +\
                    (rgrLon[lo-1, la]-rgrLon[lo-2, la+1])/2
                rgrLatBND[lo, la] = rgrLat[lo-1, la] -\
                    (rgrLat[lo-2, la+1]-rgrLat[lo-1, la])/2
            elif lo < rgiSize[0]-1 and la >= rgiSize[1]-1:
                # upper boundary second last row
                rgrLonBND[lo, la] = rgrLon[lo, la-1] -\
                    (rgrLon[lo+1, la-2]-rgrLon[lo, la-1])/2
                rgrLatBND[lo, la] = rgrLat[lo, la-1] -\
                    (rgrLat[lo+1, la-2]-rgrLat[lo, la-1])/2
            elif lo >= rgiSize[0]-1 and la >= rgiSize[1]-1:
                # upper right grid cells
                rgrLonBND[lo, la] = rgrLon[lo-1, la-1] -\
                    (rgrLon[lo-2, la-2]-rgrLon[lo-1, la-1])/2
                rgrLatBND[lo, la] = rgrLat[lo-1, la-1] -\
                    (rgrLat[lo-2, la-2]-rgrLat[lo-1, la-1])/2

    if len(rgLonSize) == 1:
        rgrLonBND = rgrLonBND[0, :]
        rgrLatBND = rgrLatBND[:, 0]

    return(rgrLonBND, rgrLatBND)


def calc_vertices(lons, lats, write_to_file=False, filename=None):
    """
    Estimate the cell boundaries from the cell location of regular grids

    Parameters
    ----------
    lons, lats: arrays
        Longitude and latitude values
    write_to_file: bool
        If True lat/lon information, including vertices, is written to file
        following the structure given by cdo commmand 'griddes'
    filename: str
        Name of text file for the grid information. Only used if write_to_file
        is True. If not provided, a default name will be used.

    Returns
    -------
    lon_bnds, lat_bnds: arrays
        Arrays of dimension [4, nlat, nlon] containing cell boundaries of each
        gridcell in lons and lats
    """

    # Dimensions lats/lons
    nlon = lons.shape[1]
    nlat = lats.shape[0]

    # Rearrange lat/lons
    lons_row = lons.flatten()
    lats_row = lats.flatten()

    # Allocate lat/lon corners
    lons_cor = np.zeros((lons_row.size*4))
    lats_cor = np.zeros((lats_row.size*4))

    lons_crnr = np.empty((lons.shape[0]+1, lons.shape[1]+1))
    lons_crnr[:] = np.nan
    lats_crnr = np.empty((lats.shape[0]+1, lats.shape[1]+1))
    lats_crnr[:] = np.nan

    # -------- Calculating corners --------- #

    # Loop through all grid points except at the boundaries
    for lat in range(1, lons.shape[0]):
        for lon in range(1, lons.shape[1]):
            # SW corner for each lat/lon index is calculated
            lons_crnr[lat, lon] = (lons[lat-1, lon-1] + lons[lat, lon-1] +
                                   lons[lat-1, lon] + lons[lat, lon])/4.
            lats_crnr[lat, lon] = (lats[lat-1, lon-1] + lats[lat, lon-1] +
                                   lats[lat-1, lon] + lats[lat, lon])/4.

    # Grid points at boundaries
    lons_crnr[0, :] = lons_crnr[1, :] - (lons_crnr[2, :] - lons_crnr[1, :])
    lons_crnr[-1, :] = lons_crnr[-2, :] + (lons_crnr[-2, :] - lons_crnr[-3, :])
    lons_crnr[:, 0] = lons_crnr[:, 1] + (lons_crnr[:, 1] - lons_crnr[:, 2])
    lons_crnr[:, -1] = lons_crnr[:, -2] + (lons_crnr[:, -2] - lons_crnr[:, -3])

    lats_crnr[0, :] = lats_crnr[1, :] - (lats_crnr[2, :] - lats_crnr[1, :])
    lats_crnr[-1, :] = lats_crnr[-2, :] + (lats_crnr[-2, :] - lats_crnr[-3, :])
    lats_crnr[:, 0] = lats_crnr[:, 1] - (lats_crnr[:, 1] - lats_crnr[:, 2])
    lats_crnr[:, -1] = lats_crnr[:, -2] + (lats_crnr[:, -2] - lats_crnr[:, -3])

    # ------------ DONE ------------- #

    # Fill in counterclockwise and rearrange
    count = 0
    for lat in range(lons.shape[0]):
        for lon in range(lons.shape[1]):

            lons_cor[count] = lons_crnr[lat, lon]
            lons_cor[count+1] = lons_crnr[lat, lon+1]
            lons_cor[count+2] = lons_crnr[lat+1, lon+1]
            lons_cor[count+3] = lons_crnr[lat+1, lon]

            lats_cor[count] = lats_crnr[lat, lon]
            lats_cor[count+1] = lats_crnr[lat, lon+1]
            lats_cor[count+2] = lats_crnr[lat+1, lon+1]
            lats_cor[count+3] = lats_crnr[lat+1, lon]

            count += 4

    lons_bnds = lons_cor.reshape(nlat, nlon, 4)
    lats_bnds = lats_cor.reshape(nlat, nlon, 4)

    if write_to_file:
        _write_grid_info(lons_row, lons_cor, lats_row, lats_cor,
                         nlon, nlat, filename=filename)

    return lons_bnds, lats_bnds


def _write_grid_info(lons_row, lons_cor, lats_row, lats_cor, nlon, nlat,
                     filename):
    """
    Write grid info to file
    """

    print("Writing grid info to disk ...")

    if filename is None:
        from datetime import datetime
        dtime = datetime.now().strftime('%Y-%m-%dT%H%M%S')
        fname = './grid_{}x{}_latlon_bounds_{}'.format(nlon, nlat, dtime)

    lt_row = np.array_split(lats_row, np.ceil(lats_row.size/6).astype(np.int))
    lt_row_str = "\n".join([" ".join(str(item) for item in arr)
                            for arr in lt_row])
    lt_cor = np.array_split(lats_cor, np.ceil(lats_cor.size/6).astype(np.int))
    lt_cor_str = "\n".join([" ".join(str(item) for item in arr)
                            for arr in lt_cor])
    ln_row = np.array_split(lons_row, np.ceil(lons_row.size/6).astype(np.int))
    ln_row_str = "\n".join([" ".join(str(item) for item in arr)
                            for arr in ln_row])
    ln_cor = np.array_split(lons_cor, np.ceil(lons_cor.size/6).astype(np.int))
    ln_cor_str = "\n".join([" ".join(str(item) for item in arr)
                            for arr in ln_cor])

    grid_txt = ("#\n# gridID 0\n#\ngridtype  = curvilinear\ngridsize  = {}\n"
                "xname     = lon\nxlongname = longitude\nxunits    = "
                "degrees_east\nyname     = lat\nylongname = latitude\nyunits"
                "    = degrees_north\nxsize     = {}\nysize     = {}\nxvals "
                "    =\n{}\nxbounds     =\n{}\nyvals     =\n{}\nybounds     "
                "=\n{}".format(
                    lons.size, lons.shape[1], lons.shape[0],
                    ln_row_str, ln_cor_str, lt_row_str, lt_cor_str))

    # Write to file
    with open(fname, 'w') as outfile:
            outfile.write(grid_txt)


def fnRemapConOperator(rgrLonS, rgrLatS, rgrLonT, rgrLatT, rgrLonSBNDS=None,
                       rgrLatSBNDS=None, rgrLonTBNDS=None, rgrLatTBNDS=None):
    """
    File name: fnRemapConOperator
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 26.05.2017
    Date last modified: 26.05.2017

    ##############################################################
    Purpose:

    Generates an opperator to coservatively remapp data from a source
    rectangular grid to an target rectangular grid.

    Parameters
    ----------
    rgrLonS,rgrLatS: arrays
        Source grid longitude and latitude values
    rgrLonT,rgrLatT: arrays
        Target grid longitude and latitude values
    rgrLonSBNDS,rgrLatSBNDS: arrays
        Source grid longitude and latitude grid point boundaries (corners).
        These must be given in the structure (lat, lon, vertices) where
        vertices are the four corners of each grid point. If not provided
        (default) then corners are calculated using fnCellCorners.
    rgrLonTBNDS,rgrLatTBNDS: arrays
        Target grid longitude and latitude grid point boundaries (corners).
        See above for more info.

    Returns
    -------
    grConRemapOp: dictionary
        opperator that contains the grid cells and their wheights of the
        source grid for each target grid cell
    """

    from shapely.geometry import Polygon
    logging.debug('fnRemapConOperator')

    # check if the grids are given in 2D
    if len(rgrLonS.shape) == 1:
        rgrLonS1 = np.asarray(([rgrLonS, ]*rgrLatS.shape[0]))
        rgrLatS = np.asarray(([rgrLatS, ]*rgrLonS.shape[0])).transpose()
        rgrLonS = rgrLonS1
    if len(rgrLonT.shape) == 1:
        rgrLonT1 = np.asarray(([rgrLonT, ]*rgrLatT.shape[0]))
        rgrLatT = np.asarray(([rgrLatT, ]*rgrLonT.shape[0])).transpose()
        rgrLonT = rgrLonT1

    # All lon grids have to go from -180 to +180 --> convert now!'
    if np.min(rgrLonS) > 180:
        rgi180 = np.where(rgrLonS > 180)
        rgrLonS[rgi180] = rgrLonS[rgi180] - 360.
    if np.min(rgrLonT) > 180:
        rgi180 = np.where(rgrLonT > 180)
        rgrLonT[rgi180] = rgrLonT[rgi180] - 360.

    if rgrLonSBNDS is None:
        # get boundarie estimations for the grid cells since the center points
        # are given
        rgrLonSB, rgrLatSB = fnCellCorners(rgrLonS, rgrLatS)
    else:
        rgrLonSB = rgrLonSBNDS
        rgrLatSB = rgrLatSBNDS
        # All lon grids have to go from -180 to +180 --> convert now!'
        if np.min(rgrLonSB) > 180:
            rgi180 = np.where(rgrLonSB > 180)
            rgrLonSB[rgi180] = rgrLonSB[rgi180] - 360.
    if rgrLonTBNDS is None:
        rgrLonTB, rgrLatTB = fnCellCorners(rgrLonT, rgrLatT)
    else:
        rgrLonTB = rgrLonTBNDS
        rgrLatTB = rgrLatTBNDS
        if np.min(rgrLonTB) > 180:
            rgi180 = np.where(rgrLonTB > 180)
            rgrLonTB[rgi180] = rgrLonTB[rgi180] - 360.

    # get approximate grid spacing of source and target grid
    rGsS = (abs(np.mean(rgrLonS[:, 1:]-rgrLonS[:, 0:-1])) +
            abs(np.mean(rgrLatS[1:, :]-rgrLatS[0:-1, :])))/2
    rGsT = (abs(np.mean(rgrLonT[:, 1:]-rgrLonT[:, 0:-1])) +
            abs(np.mean(rgrLatT[1:, :]-rgrLatT[0:-1, :])))/2
    rRadius = ((rGsS+rGsT)*1.2)/2.

    # loop over the target grid cells and calculate the weights
    grRemapOperator = {}
    for la in range(rgrLonT.shape[0]):
        for lo in range(rgrLonT.shape[1]):
            rgbGCact = ((rgrLonS > rgrLonT[la, lo]-rRadius) &
                        (rgrLonS < rgrLonT[la, lo]+rRadius) &
                        (rgrLatS > rgrLatT[la, lo]-rRadius) &
                        (rgrLatS < rgrLatT[la, lo]+rRadius))
            if np.sum(rgbGCact) > 0:
                rgrLaLoArea = np.array([])
                # produce polygon for target grid cell
                if rgrLonTBNDS is None:
                    points = [(rgrLonTB[la, lo], rgrLatTB[la, lo]),
                              (rgrLonTB[la, lo+1], rgrLatTB[la, lo+1]),
                              (rgrLonTB[la+1, lo+1], rgrLatTB[la+1, lo+1]),
                              (rgrLonTB[la+1, lo], rgrLatTB[la+1, lo])]
                else:
                    points = [(x, y) for x, y in zip(rgrLonTB[la, lo, :],
                                                     rgrLatTB[la, lo, :])]
                pT = Polygon(points)

                # loop over source grid cells
                rgiGCact = np.where(rgbGCact)
                for sg in range(np.sum(rgbGCact)):
                    laS = rgiGCact[0][sg]
                    loS = rgiGCact[1][sg]
                    if rgrLonSBNDS is None:
                        points = [
                            (rgrLonSB[laS, loS], rgrLatSB[laS, loS]),
                            (rgrLonSB[laS, loS+1], rgrLatSB[laS, loS+1]),
                            (rgrLonSB[laS+1, loS+1], rgrLatSB[laS+1, loS+1]),
                            (rgrLonSB[laS+1, loS], rgrLatSB[laS+1, loS])]
                    else:
                        points = [(x, y)
                                  for x, y in zip(rgrLonSB[laS, loS, :],
                                                  rgrLatSB[laS, loS, :])]
                    pS = Polygon(points)
                    rIntArea = pS.intersection(pT).area/pS.area
                    if rIntArea != 0:
                        if len(rgrLaLoArea) == 0:
                            rgrLaLoArea = np.array([[laS, loS, rIntArea]])
                        else:
                            rgrLaLoArea = np.append(
                                rgrLaLoArea, np.array([[laS, loS, rIntArea]]),
                                axis=0)
                grRemapOperator[str(la)+','+str(lo)] = rgrLaLoArea

    return grRemapOperator


def fnRemapCon(rgrLonS, rgrLatS, rgrLonT, rgrLatT, grOperator, rgrData):
    """
    File name: fnRemapCon
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 13.06.2017
    Date last modified: 13.06.2017

    ##############################################################
    Purpose:

    Uses the remap operator generated by the function fnRemapConOperator to
    remap data to a target grid conservatively

    Parameters
    ----------
    rgrLonS,rgrLatS: arrays
        Source grid longitude and latitude values
    rgrLonT,rgrLatT: arrays
        Target grid longitude and latitude values
    grOperator: dictionary
        Remapping operator returned from fnRemapConOperator.
    rgrData: 3D/4D array
        Data to be regridded, structured as (time, lat, lon) or (time,
        variables, lat, lon).

    Returns
    -------
    rgrTarData: array
        Remapped data matrix
    """

    from ipdb import set_trace as stop
    logging.debug('fnRemapCon')

    if len(rgrData.shape) == 3:
        if len(rgrLonT.shape) == 1:
            rgrTarData = np.zeros((rgrData.shape[0], rgrLatT.shape[0],
                                   rgrLonT.shape[0]))
            rgrTarData[:] = np.nan
        elif len(rgrLonT.shape) == 2:
            rgrTarData = np.zeros((rgrData.shape[0], rgrLatT.shape[0],
                                   rgrLatT.shape[1]))
            rgrTarData[:] = np.nan
        for gc in grOperator:
            rgiGcT = np.array(gc.split(',')).astype('int')
            rgrSource = grOperator[gc]
            if len(rgrSource) != 0:
                try:
                    rgrTarData[:, rgiGcT[0], rgiGcT[1]] = np.average(
                        rgrData[:, rgrSource[:, 0].astype('int'),
                                rgrSource[:, 1].astype('int')],
                        weights=rgrSource[:, 2], axis=1)
                except:
                    logging.warn("stop")
                    stop()
    if len(rgrData.shape) == 4:
        # the data has to be in [time, variables, lat, lon]
        rgrTarData = np.zeros((rgrData.shape[0], rgrData.shape[1],
                               rgrLatT.shape[0], rgrLonT.shape[0]))
        rgrTarData[:] = np.nan
        for gc in grOperator:
            rgiGcT = np.array(gc.split(',')).astype('int')
            rgrSource = grOperator[gc]
            rgrTarData[:, :, rgiGcT[0], rgiGcT[1]] = np.average(
                rgrData[:, :, rgrSource[:, 0].astype('int'),
                        rgrSource[:, 1].astype('int')],
                weights=rgrSource[:, 2], axis=2)

    return rgrTarData


def add_matrix_NaNs(regridder):
    """
    Replace zero values of cells in the new grid that are outside
    the old grid's domain with NaN's.

    Parameters
    ----------
    regridder: Object from xESMF Regridder function

    Returns
    -------
    regridder
        Modified regridder where zero valued cells (outside source grid)
        has been replaced with NaN's.
    """
    from scipy import sparse
    X = regridder.A
    M = sparse.csr_matrix(X)
    num_nonzeros = np.diff(M.indptr)
    M[num_nonzeros == 0, 0] = np.NaN
    regridder.A = sparse.coo_matrix(M)

    return regridder


def get_remap_method(variable):
    """
    Return interpolation method based on input variable.
    """
    method_dict = {
        'tas': 'bilinear',
        'tas_ol': 'bilinear',
        'tasmax': 'bilinear',
        'tasmin': 'bilinear',
        'psl': 'bilinear',
        'uas': 'bilinear',
        'vas': 'bilinear',
        'clt': 'bilinear',
        'cll': 'bilinear',
        'clm': 'bilinear',
        'clh': 'bilinear',
        'clwvi': 'bilinear',
        'clivi': 'bilinear',
        'prw': 'bilinear',
        'snc': 'bilinear',
        'snd': 'bilinear',
        'snw_b': 'bilinear',
        'cape': 'bilinear',
        'pcape': 'bilinear',
        'pcin': 'bilinear',
        'pr': 'conservative',
        'prsnow': 'conservative',
        'prgrpl': 'conservative',
        'prsolid': 'conservative',
        'ssn': 'conservative',
        'csn': 'conservative',
        'cpr': 'conservative',
        'rlds': 'conservative',
        'rsds': 'conservative',
        'rlnt': 'conservative',
        'hfls': 'conservative',
        'hfss': 'conservative',
        'phi500': 'bilinear',
        'ta1000': 'bilinear',
        'ta950': 'bilinear',
        'ta850': 'bilinear',
        'ta700': 'bilinear',
        'ta500': 'bilinear',
        'ta350': 'bilinear',
        'hus1000': 'bilinear',
        'hus950': 'bilinear',
        'hus850': 'bilinear',
        'hus700': 'bilinear',
        'hus500': 'bilinear',
        'hus350': 'bilinear',
        'w500': 'bilinear',
        'w700': 'bilinear',
        'w850': 'bilinear',
        'ua850': 'bilinear',
        'va850': 'bilinear',
        'sfcWind': 'bilinear',
    }

    return method_dict[variable]
