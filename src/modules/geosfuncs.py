#!/usr/bin/env python
#
#
#  Routine for Masking Data with Polygons ##
#
#  By: Petter Lind
#      2014-03-02  v1.0
#      2016-11-04  v2.0
#
import sys
import numpy as np
import os


def regions(area="", reg_print=False):
    """
    See available and retrieve regions from a region dictionary.

    Parameters
    ----------
    area: string
        name of region to be extracted
    reg_print: boolean
        if available regions should be printed

    Returns
    -------
    area_file: string
        path to the text file with polygon data
    """

    # -------- Dictionary of predfined regions -------- #
    reg_dir = "/home/sm_petli/dev/code/polygons/"

    reg_dict = {
            "Sweden": "sweden_poly_6-12km.txt",
            "Sweden flat": "sweden_no_mountains.txt",
            "Sweden east coast": "sweden_east_coast.txt",
            "Sweden north east coast": "sweden_northeast_coast.txt",
            "North Sweden": "north_sweden.txt",
            "South Sweden": "south_sweden.txt",
            "Stockholm region": "stockholm_region.txt",
            "Helsinki region": "helsinki_region.txt",
            "Oslo region": "oslo_region.txt",
            "Switzerland": "switzerland_poly_6-12km.txt",
            "Norway": "norway_poly_6-12km.txt",
            "South Norway": "south_norway_v2.txt",
            "Mid Norway": "mid_norway.txt",
            "Norwegian Sea": "norwegian_sea.txt",
            "Finland": "finland_poly.txt",
            "Denmark": "denmark_poly.txt",
            "Netherlands": "netherlands.txt",
            "Germany": "germany_poly_6-12km.txt",
            "South Germany": "south_germany_poly.txt",
            "Iberian peninsula": "iberian_poly_6-12km.txt",
            "Spain": "spain_poly_6-12km.txt",
            "France": "france_poly_6-12km.txt",
            "Utd Kingdom": "UK_poly_6-12km.txt",
            "EURO4M-APGD": "alp_poly.txt",
            "Scandinavia": "Scand_poly.txt",
            "Scandinavia Interior": "scandinavia_interior.txt",
            "S Scandinavia": "south_scandinavia.txt",
            "S Scandinavia Land": "south_scandinavia_land.txt",
            "N Scandinavia Land": "north_scandinavia_land.txt",
            "NE Scandinavia": "northeast_scandinavia.txt",
            "NW Scandinavia": "northwest_scandinavia.txt",
            "Central Europe": "MidEur_poly.txt",
            "East Europe": "EastEur_poly.txt",
            "S-E Europe": "SEastEur_poly.txt",
            "West Europe": "WestEur_poly.txt",
            "British Isles": "BIsles_poly_LS.txt",
            "Mediterranean": "MedSea_poly.txt",
            "USIS Sthlm": "usis_sthlm.txt",
            "USIS Bologna": "usis_bologna.txt",
            "USIS Amsterdam": "usis_amsterdam.txt",
            "Crete": "crete_poly.txt",
            "Crete domain": "crete_domain_v2.txt",
            "NorCP analysis domain": "norcp_analysis_domain.txt",
            "IMPRX SW Europe": "imprx_sweur_arome3_reduced100.txt",
            "IMPRX SW Europe Land": "imprx_sweur_arome3_reduced100_land.txt",
            "IMPRX SW Europe Sea": "imprx_sweur_arome3_reduced100_sea.txt",
            "IMPRX CE Europe": "imprx_ceeur_arome3_reduced100.txt",
            "IMPRX Nordic": "imprx_nordic_arome3_reduced100.txt",
            "IMPRX Nordic Land": "imprx_nordic_arome3_reduced100_land.txt",
            "IMPRX Nordic Sea": "imprx_nordic_arome3_reduced100_sea.txt",
            "IMPRX Balticum": "imprx_balticum.txt",
            "IMPRX PolBelUkr": "imprx_PolBelUkr.txt",
            "IMPRX Po Valley N Adriatic": "imprx_Po_valley_N_AdriaticSea.txt",
            "IMPRX West Mediterranean": "imprx_west_mediterranean.txt",
            "IMPRX Hungary plains": "imprx_hungary_plains.txt",
            "IMPRX Mid-Sweden": "imprx_mid_sweden.txt",
            "Northernmost Scandinavia": "northernmost_scandinavia.txt",
            }

    if reg_print:
        print
        print("Available regions:")
        print
        print([k for k in reg_dict])
    else:
        try:
            area_file = os.path.join(reg_dir, reg_dict[area])
            return area_file
        except ValueError:
            print
            print("Error! \n {0} is not a pre-defined area.".format(area))


def reg_mask(xp, yp, area, data=None, iter_3d=None, cut_data=False):
    """
    Routine to mask grid points outside a specified polygon.
    Parameters
    ----------
    xp,yp: numpy arrays
        eastward and northward grid points respectively, normally lon/lat
        arrays.
    area: string or list with tuples
        Either name of predefined region to extract (see reg_dict function for
        more info.), or a list with tuples defining a polygon;
        e.g. [(lon1, lat1), (lon2, lat2),...,(lonN, latN), (lon1, lat1)].
        NB! First and last tuple must be the same, closing the polygon.
    data: 2D/3D numpy array, optional
        data matrix
    iter_3d: int, optional
        If set, masking is performed along the zeroth dimension of array. Thus
        array must be 3D and have shape [iter_3d, y, x].
    cut_data: boolean, optional
        If True, the data will be cut to a box that covers the input area,
        leaving one extra (masked) grid point beyond each limiting
        north/south/east/west grid point. This option will
        return not only masked and cut data but also cut lon/lat data as well
        as the box edges (as indices) in the x,y plane.
    Returns
    -------
    mask_out: boolean array
        A 2D mask with True inside region of interest, False outside. Returned
        if data is None.
    masked_data: 2D/3D numpy array
        If data is passed to function, this is the returned masked data.
    xp_cut/yp_cut: numpy arrays
        If cut_data is True, these arrays are the cut xp/yp arrays
    reg_x_edges, reg_y_edges: tuples
        If cut_data is True, the tuples contain index of the edges of cropped
        region in x and y directions respectively.
    """
    from matplotlib.path import Path
    if data is not None:
        if iter_3d is not None:
            msg = ("\nERROR!\nIf iter_3d is set data array must be "
                   "three-dimensional with shape (iter_3d, y, x).")
            assert data.ndim == 3, msg
        else:
            msg = ("\nERROR!\nIf iter_3d is NOT set data array "
                   "must be two-dimensional with shape (y, x).")
            assert data.ndim == 2, msg

    # Dimensions of grid
    if xp.ndim == 2:
        ny, nx = xp.shape
    else:
        nx = xp.size
        ny = yp.size

    # Rearrange grid points into a list of tuples
    points = [(xp[i, j], yp[i, j]) if xp.ndim == 2 else (xp[j], yp[i]) for i in
              range(ny) for j in range(nx)]
    if type(area).__name__ == 'str':
        def coord_return(line):
            s = line.split()
            return list(map(float, s))
        reg_file = regions(area)
        with open(reg_file, 'r') as ff:
            ff.readline()       # Skip first line
            poly = [coord_return(l) for l in ff.readlines()]
    else:
        poly = area
    reg_path = Path(np.array(poly))
    mask = reg_path.contains_points(points)
    mask = np.invert(mask.reshape(ny, nx))
    if cut_data:
        # Cut out region
        yp_r, xp_r = [(np.min(np.where(~mask)[i]),
                       np.max(np.where(~mask)[i]))
                      for i in range(2)]
        # yp_r, xp_r = [(np.min(np.where(~mask)[i])-1,
        #                np.max(np.where(~mask)[i])+1)
        #               for i in range(2)]
        reg_x_edges, reg_y_edges = xp_r, yp_r
        xr = np.int(np.diff(xp_r)+1)
        yr = np.int(np.diff(yp_r)+1)
        if xp.ndim == 2:
            xp_cut = xp[yp_r[0]:yp_r[1]+1, xp_r[0]:xp_r[1]+1]
            yp_cut = yp[yp_r[0]:yp_r[1]+1, xp_r[0]:xp_r[1]+1]
        else:
            xp_cut = xp[xp_r[0]:xp_r[1]+1]
            yp_cut = yp[yp_r[0]:yp_r[1]+1]
        mask_cut = mask[yp_r[0]:yp_r[1]+1, xp_r[0]:xp_r[1]+1]
        mask_out = np.invert(mask_cut)
        if data is not None:
            if iter_3d is not None:
                masked_data = np.zeros((iter_3d, yr, xr))
                masked_data[:] = data[:, yp_r[0]:yp_r[1]+1, xp_r[0]:xp_r[1]+1]
                mask_3d = np.repeat(mask_cut[np.newaxis, :, :],
                                    iter_3d, axis=0)
                masked_data[mask_3d] = np.nan
            else:
                masked_data = np.zeros((yr, xr))
                masked_data[:] = data[yp_r[0]:yp_r[1]+1, xp_r[0]:xp_r[1]+1]
                masked_data[mask_cut] = np.nan
            masked_out = (masked_data, xp_cut, yp_cut, reg_x_edges,
                          reg_y_edges)
        else:
            masked_out = (mask_out, xp_cut, yp_cut, reg_x_edges,
                          reg_y_edges)
    else:
        if data is not None:
            masked_data = data.copy()
            if iter_3d is not None:
                mask_3d = np.repeat(mask[np.newaxis, :, :], iter_3d, axis=0)
                masked_data[mask_3d] = np.nan
            else:
                masked_data[mask] = np.nan
            masked_out = masked_data
        else:
            masked_out = np.invert(mask)
    return masked_out


def get_poly(print_areas=False):
    """
    Retrieve polygon arbitrarily drawn interactively on a map
    by mouse clicking.

    Parameters
    ----------
    print_areas: Boolean
        Available regions are shown if set True.

    Returns
    -------
    poly: list
        List with tuples of lat/lon coordinates for drawn polygon
    """
    import matplotlib.pyplot as plt
    import draw_polygon
    from mpl_toolkits.basemap import Basemap

    def get_map(area, map_dict=None):
        # Create map object
        if area == 'latlon':
            s1 = "Type in southern-most latitude"
            s2 = "Type in northern-most latitude"
            s3 = "Type in western-most longitude"
            s4 = "Type in eastern-most longitude"
            lat1 = input('\n{}\n>> '.format(s1))
            lat2 = input('\n{}\n>> '.format(s2))
            lon1 = input('\n{}\n>> '.format(s3))
            lon2 = input('\n{}\n>> '.format(s4))
            lat1 = float(lat1)
            lat2 = float(lat2)
            lon1 = float(lon1)
            lon2 = float(lon2)
            m = Basemap(ax=ax,
                        llcrnrlat=lat1, urcrnrlat=lat2,
                        llcrnrlon=lon1, urcrnrlon=lon2,
                        lat_0=(lat1+lat2)/2, lon_0=(lon1+lon2)/2,
                        projection='stere',
                        resolution='l')
        else:
            m = Basemap(ax=ax,
                        width=map_dict[area]['wdth'],
                        height=map_dict[area]['hght'],
                        projection=map_dict[area]['proj'],
                        resolution='l',
                        lat_0=map_dict[area]['lat0'],
                        lon_0=map_dict[area]['lon0'])
        m.drawmapboundary(fill_color='#e6f2ff')
        m.fillcontinents(color='#e6e6e6', lake_color='#e6f2ff')
        m.drawcoastlines(color='#262626')
        m.drawcountries()
        m.drawstates(color='darkgrey')
        return m

    map_dict = {
            "Europe": {
                        "proj": "lcc",
                        "wdth": 5e6,
                        "hght": 5e6,
                        "lon0": 17,
                        "lat0": 51,
                        },
            "North America": {
                        "proj": "lcc",
                        "wdth": 9e6,
                        "hght": 7e6,
                        "lon0": -110,
                        "lat0": 47,
                        },
            "South America": {
                        "proj": "lcc",
                        "wdth": 7e6,
                        "hght": 9e6,
                        "lon0": -59,
                        "lat0": -22,
                        },
            "Africa": {
                        "proj": "lcc",
                        "wdth": 10e6,
                        "hght": 9.5e6,
                        "lon0": 16,
                        "lat0": 2,
                        },
            "Australia": {
                        "proj": "lcc",
                        "wdth": 7e6,
                        "hght": 6e6,
                        "lon0": 145,
                        "lat0": -25,
                        },
            "South-East Asia": {
                        "proj": "lcc",
                        "wdth": 7e6,
                        "hght": 6e6,
                        "lon0": 98,
                        "lat0": 12,
                        },
            "East Asia": {
                        "proj": "lcc",
                        "wdth": 7e6,
                        "hght": 4.5e6,
                        "lon0": 111,
                        "lat0": 35,
                        },
            "Central Asia": {
                        "proj": "lcc",
                        "wdth": 4.8e6,
                        "hght": 3e6,
                        "lon0": 54,
                        "lat0": 49,
                        },
            "Middle East": {
                        "proj": "lcc",
                        "wdth": 5e6,
                        "hght": 4e6,
                        "lon0": 51,
                        "lat0": 27,
                        },
            }

    fig = plt.figure(1, (16, 16))
    ax = fig.add_subplot(111)

    # Create Canvas object and mouse click settings
    cnv = draw_polygon.Canvas(ax)
    plt.connect('motion_notify_event', cnv.set_location)
    plt.connect('button_press_event', cnv.update_path)

    print("Welcome!\n")
    s1 = "What area would you like to show on map?"
    s2 = "Type 'print areas' to see selectable areas"
    s3 = "Type 'latlon' to choose area manually"
    area = input('{}\n{}\n{}\n\n>> '.format(s1, s2, s3))

    if area == 'print areas':
        print()
        print("Available map areas:")
        [print('{}\n'.format(ar)) for ar in map_dict.keys()]

        area = input("Ok, so what area have you chosen?\n\n>> ")
        m = get_map(area, map_dict)
    elif area == 'latlon':
        m = get_map(area)
    else:
        try:
            m = get_map(area, map_dict)
        except ValueError:
            print("The area you provided is not available.\nPlease, try"
                  " again ...")
            area = input("What area would you like to show on map?\n\n>> ")
            try:
                m = get_map(area, map_dict)
            except ValueError:
                print("Nope! Something is wrong ... exiting")
                sys.exit()

    s1 = "It's time to choose a polygon in the map soon to be shown ..."
    s2 = "This is done in two steps:"
    s3 = ("With left mouse button click points for polygon; "
          "right click to connect end points.")
    s4 = "When finished, close map window"
    print('\n\n{}\n{}\n   1) {}\n   2) {}\n'.format(s1, s2, s3, s4))
    input("To continue, please press enter...")

    plt.show()

    verts = np.array(cnv.vert)
    lonpt, latpt = m(verts[:, 0], verts[:, 1], inverse=True)

    poly = [(x, y) for x, y in zip(lonpt, latpt)]

    print()
    print()
    s1 = "Do you want to write the polygon to disk?"
    s2 = "Then, type 'write', it will be saved as 'poly.txt'"
    s3 = "If not, just press enter ..."
    write = input('{}\n{}\n{}\n\n>> '.format(s1, s2, s3))

    if write == 'write':
        f = open('poly.txt', 'w')
        f.write(' '.join(str(s) for s in ('x', 'y')) + '\n')
        for t in poly:
            f.write(' '.join(str(s) for s in t) + '\n')
        f.close()

    return poly


def topo_mask(data, orog, orog_int):
    """
    Rotuine to mask grid points outside a specified height interval.

    Parameters
    ----------
    data:  2D numpy array
        the data matrix
    orog:  2D numpy array
        array containing topographical heights (same resolution as data)
    orog_int: 1D array,list,tuple
        the height interval outside which data should be masked.

    Returns
    -------
    masked_data: 2D numpy array
        the masked data

    """

    assert data.ndim == 2, \
        "Error! data array must be two-dimensional."

    assert data.shape() == orog.shape(), \
        "Error! data and orography arrays must have identical grids"

    # Create topographical mask
    orog_mask = np.ma.masked_outside(orog, orog_int[0], orog_int[1])
    orog_mask = np.ma.getmask(orog_mask)

    masked_data = np.ma.masked_where(orog_mask, data)

    return masked_data


def find_geo_indices(lons, lats, x, y):
    """
    Search for nearest decimal degree in an array of decimal degrees and
    return the index.
    np.argmin returns the indices of minimum value along an axis.
    Subtract dd from all values in dd_array, take absolute value and find index
    of minimum.

    Parameters
    ----------
    lons/lats: 1D/2D numpy array
        Latitude and longitude values defining the grid.
    x, y: int/float
        Coordinates of searched for data point

    Returns
    -------
    lat_idx/lon_idx: ints
        The latitude and longitude indices where closest to searched
        data point.
    """
    if lons.ndim == 1:
        lons2d = np.repeat(lons[np.newaxis, :], lats.size, axis=0)
        lats2d = np.repeat(lats[:, np.newaxis], lons.size, axis=1)
    else:
        lons2d = lons
        lats2d = lats

    lonlat = np.dstack([lons2d, lats2d])
    delta = np.abs(lonlat-[x, y])
    ij_1d = np.linalg.norm(delta, axis=2).argmin()
    lat_idx, lon_idx = np.unravel_index(ij_1d, lons2d.shape)
    return lat_idx, lon_idx
