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
from mpl_toolkits.basemap import Basemap
import argparse


def polygons(area="", poly_print=False):
    """
    See available polygons and retrieve path to polygon files.

    Parameters
    ----------
    area: string
        name of polygon to be extracted
    poly_print: boolean
        if available polygons should be printed

    Returns
    -------
    area_file: string
        path to the text file with polygon data
    """

    # -------- Dictionary of predfined regions -------- #
    file_dir = os.path.dirname(os.path.abspath(__file__))
    # p_dir = os.path.dirname(file_dir)
    polypath = os.path.join(file_dir, 'polygon_files')

    errmsg = "Folder to polygons: {}, does not seem to exist!".format(polypath)
    assert os.path.exists(polypath), errmsg

    polygons = os.listdir(polypath)
    poly_dict = {s.split('.')[0].replace('_', ' '): s for s in polygons}

    if poly_print:
        print("\nAvailable polygons/regions:\n")
        [print('\t{}'.format(ar)) for ar in poly_dict]
    else:
        # try:
        #     area_file = os.path.join(polypath, poly_dict[area])
        #     return area_file
        # except ValueError:
        #     print
        #     print("ERROR! \n {0} is not a pre-defined area.".format(area))
        errmsg = ("\n\n\tOohps! '{0}' is not a pre-defined area. Check polygon"
                  " folder for the correct name or create a new "
                  "polygon.").format(area)
        assert area in poly_dict, errmsg

        area_file = os.path.join(polypath, poly_dict[area])
        return area_file


def mask_region(xp, yp, area, data=None, iter_3d=None, cut_data=False):
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
        reg_file = polygons(area)
        with open(reg_file, 'r') as ff:
            ff.readline()       # Skip first line
            poly = [coord_return(ln) for ln in ff.readlines()]
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


def create_polygon():
    """
    Retrieve polygon arbitrarily drawn interactively on a map
    by mouse clicking.

    Parameters
    ----------
    print_zoom_areas: Boolean
        Prints available zoom regions for polygon selection.

    Returns
    -------
    poly: list
        List with tuples of lat/lon coordinates for drawn polygon
    """
    import matplotlib.pyplot as plt
    from rcat.utils import draw_polygon

    def get_map(area, map_resolution='l', map_dict=None):
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
                        resolution=map_resolution)
        else:
            m = Basemap(ax=ax,
                        width=map_dict[area]['wdth'],
                        height=map_dict[area]['hght'],
                        projection=map_dict[area]['proj'],
                        resolution=map_resolution,
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

    mres = input("\nOne more thing before continuing -- what resolution would "
                 "you like for the map?\nOptions are 'c' (crude), 'l' (low), "
                 "'i' (intermediate), 'h' (high), 'f' (full).\nDefault is low"
                 " -- if this is ok just press enter ...\n>> ")
    mres = 'l' if not mres else mres

    if area == 'print areas':
        print()
        print("Available map areas:\n")
        [print('{}'.format(ar)) for ar in map_dict.keys()]

        area = input("\n\nOk, so what area have you chosen?\n>> ")
        m = get_map(area, mres, map_dict)
    elif area == 'latlon':
        m = get_map(area, mres)
    else:
        try:
            m = get_map(area, mres, map_dict)
        except ValueError:
            print("The area you provided is not available.\nPlease, try"
                  " again ...")
            area = input("What area would you like to show on map?\n\n>> ")
            try:
                m = get_map(area, mres, map_dict)
            except ValueError:
                print("Sorry! Something is wrong ... exiting")
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

    s1 = "Do you want to write the polygon to disk?"
    s2 = "Then, type 'write' and instructions will follow."
    s3 = "If not, just press enter ..."
    write = input('\n\n{}\n{}\n{}\n>> '.format(s1, s2, s3))

    if write == 'write':
        s1 = "Type file directory path and file name as 'fdir, fname'"
        s2 = "Make sure 'fname' is an appropriate name for the polygon."
        s3 = ("N.B.\nIf polygon should be added to RCAT, make sure directory "
              "path is set to: <path-to-rcat>/src/polygons")
        file_info = input('\n{}\t\n{}\t\n{}\n>> '.format(s1, s2, s3))
        file_info = file_info.split(',')

        fdir = file_info[0].strip()
        _fname = file_info[1].strip()
        fname = "{}.txt".format(_fname.replace(' ', '_'))
        with open(os.path.join(fdir, fname), 'w') as f:
            f.write(' '.join(str(s) for s in ('x', 'y')) + '\n')
            for t in poly:
                f.write(' '.join(str(s) for s in t) + '\n')

    return poly


def plot_polygon(polygon, map_resolution='l', savefig=False, figpath=None):
    """
    Plot polygon on map.

    Parameters
    ----------
    polygon: string or list
        Name of polygon as defined by poly_dict dictionary in 'polygons'
        function, or list with polygon coordinates [[lon1, lat1], [lon2, lat2],
        ..., [lon1, lat1]].
    map_resolution: string
        The resolution for the plotted map (resolution of boundary database to
        use). Can be 'c' (crude), 'l' (low), 'i' (intermediate), 'h' (high),
        'f' (full). Crude and low are installed by default. If higher
        resolution is wanted ('i', 'h' or 'f') high-res data must be installed.
        In conda this can be achieved by running:
        conda install -c conda-forge basemap-data-hires
    savefig: boolean
        If True, figure is saved to 'figpath' location ('figpath' must be set).
        If false, figure is displayed on screen.
    figpath: string
        Path to folder for saved polygon figure.
    """
    import matplotlib.pyplot as plt
    from datetime import datetime

    # Colors
    water = 'lightskyblue'
    earth = 'cornsilk'

    # Read polygon
    if type(polygon).__name__ == 'str':
        def coord_return(line):
            s = line.split()
            return list(map(float, s))
        reg_file = polygons(polygon)
        with open(reg_file, 'r') as ff:
            ff.readline()       # Skip first line
            poly = [coord_return(ln) for ln in ff.readlines()]
        fname = '{}_polygon_plot.png'.format(polygon.replace(' ', '_'))
    else:
        poly = polygon
        now = datetime.now().strftime('%y%m%dT%H%M')
        fname = 'polygon_plot_{}.png'.format(now)

    # Lats/lons
    lons = [p[0] for p in poly]
    lon_incr = (max(lons) - min(lons))*.2
    lon_0 = (max(lons) + min(lons))/2

    lats = [p[1] for p in poly]
    lat_incr = (max(lats) - min(lats))*.2
    lat_0 = (max(lats) + min(lats))/2

    # Initalize figure
    fig = plt.figure(figsize=[12, 13])
    ax = fig.add_subplot(111)

    mm = Basemap(resolution=map_resolution, projection='stere', ellps='WGS84',
                 lon_0=lon_0, lat_0=lat_0,
                 llcrnrlon=min(lons)-lon_incr, llcrnrlat=min(lats)-lat_incr,
                 urcrnrlon=max(lons)+lon_incr, urcrnrlat=max(lats)+lat_incr,
                 ax=ax)
    mm.drawmapboundary(fill_color=water)
    try:
        mm.drawcoastlines()
    except ValueError:
        pass
    try:
        mm.fillcontinents(color=earth, lake_color=water)
    except ValueError:
        pass
    try:
        mm.drawcountries()
    except ValueError:
        pass

    _draw_screen_poly(lats, lons, ax, mm, linewidth=4, color='m')

    if savefig:
        errmsg = "Error! 'figpath' must be set if saving figure"
        assert figpath is not None, errmsg
        plt.savefig(os.path.join(figpath, fname))
    else:
        plt.show()


def _draw_screen_poly(lats, lons, ax, m, color='k', linewidth=3, alpha=1.0):
    from matplotlib.patches import Polygon

    x, y = m(lons, lats)
    xy = list(zip(x, y))
    poly = Polygon(xy, edgecolor=color, facecolor='none',
                   lw=linewidth, alpha=alpha)
    ax.add_patch(poly)


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Script to create or plot polygons')
    parser.add_argument('--purpose', '-p',  metavar='PURPOSE', type=str,
                        help=('<Required> Purpose of application; either '
                              '"create", "plot" or "printareas" for creating, '
                              'plotting or printing available polygons '
                              'respectively.'),
                        required=True)
    parser.add_argument('--area', '-a', metavar='POLYGON', type=str,
                        help=('If purpose is to plot, this is the name of '
                              'polygon to be plotted'))
    parser.add_argument('--resolution', '-r', metavar='MAP_RESOLUTION',
                        default='l', type=str,
                        help=(
                            'If purpose is to plot, this is the resolution '
                            'of the drawn map (resolution of boundary database'
                            ' to use). Can be "c" (crude), "l" (low), "i" '
                            '(intermediate), "h" (high), "f" (full). Crude '
                            'and low are installed by default. If higher '
                            'resolution is wanted ("i", "h" or "f") high-res'
                            ' data must be installed. In conda this can be '
                            'achieved by running:\n conda install -c '
                            'conda-forge basemap-data-hires.'))
    parser.add_argument('--save', metavar='BOOLEAN', default=False,
                        type=bool, help=('Wether to save polygon plot. Provide'
                                         ' also figpath (--figpath) if needed '
                                         '(defaults to ./)'))
    parser.add_argument('--figpath', metavar='FILEPATH', default='./',
                        type=str, help='Where to save polygon plot')
    args = parser.parse_args()

    if args.purpose == 'create':
        create_polygon()
    elif args.purpose == 'plot':
        area = args.area
        errmsg = ("\n\tWhen plotting polygon, polygon name must be provided")
        assert area is not None, errmsg

        plot_polygon(area, args.resolution, args.save, args.figpath)
    elif args.purpose == 'printareas':
        polygons(poly_print=True)
    else:
        print("Unknown purpose of application (--purpose)!\nExiting...")
        sys.exit()
