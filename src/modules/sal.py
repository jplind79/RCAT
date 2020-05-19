"""
SAL module
---------------
Functions for calculation of SAL statistics.

Based on Wernli et al 2008
http://journals.ametsoc.org/doi/abs/10.1175/2008MWR2415.1

Created: Spring 2018
Authors: Petter Lind & David Lindstedt
"""

# Global modules
import sys
import xarray as xa
import numpy as np
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import remove_small_objects
import multiprocessing as mp


def A_stat(data, refdata):
    """
    Calculate the amplitude component (A).

    Parameters
    ----------
    data, refdata: arrays
        2D data arrays to be compared, where refdata is the reference data e.g.
        observations.

    Returns
    -------
    A: float
        The calculated amplitude component
    """
    if isinstance(data, np.ma.MaskedArray):
        data_mean = np.ma.mean(data)
    else:
        data_mean = np.nanmean(data)
    if isinstance(refdata, np.ma.MaskedArray):
        refdata_mean = np.ma.mean(refdata)
    else:
        refdata_mean = np.nanmean(refdata)

    A = (data_mean - refdata_mean) / (0.5 * (data_mean + refdata_mean))

    return A


def S_stat(data, data_label, refdata, refdata_label, obj_prop=True,
           lsmask=None):
    """
    Function to calculate the structure component (S).  The basic idea is to
    compare the volume of the normalized precipitation objects. This property
    captures information of the geometrical characteristics (size and shape)
    and how they differ between model (M) and reference (O).

    Parameters
    ----------
    data, refdata: arrays
        2D data arrays to be compared, where refdata is the reference data e.g.
        observations.
    data_label, refdata_label: arrays
        Arrays with labeled objects. Returned from the label() function.
    obj_prop: boolean
        If True, individual object (rain fall area) properties are calculated
        and returned.
    lsmask: array
        Land/sea mask (2d boolean array) to characterize identified objects as
        land, ocean or coastal objects.

    Returns
    -------
    S: float
        The calculated structure component.
    area_props/ref_area_props: dictionary
        Dictionary containing properties of identified objects in data and
        refdata respectively.
    """

    def get_object_properties(darray, lbl, area_prop):
        # Properties of identified rain areas
        props = regionprops(lbl, darray)

        # Number of rain areas
        nobj = len(props)

        if area_prop:
            obj_cm_x = np.zeros(nobj)
            obj_cm_y = np.zeros(nobj)
            obj_areas = np.zeros(nobj)
            obj_mxint = np.zeros(nobj)
            obj_int_ratio = np.zeros(nobj)
            obj_major_ax = np.zeros(nobj)
            obj_minor_ax = np.zeros(nobj)
            obj_extent = np.zeros(nobj)
            if lsmask is not None:
                obj_ident = np.zeros(nobj)
        else:
            obj_dict = None

        data_Rn = np.zeros(nobj)
        data_RnVn = np.zeros(nobj)

        # Loop over objects
        for i, obj in enumerate(props):
            obj_data = np.array([darray[tuple(coord)] for coord in obj.coords])

            Rn = obj_data.sum()      # Object mean value
            Rmax = obj.max_intensity   # Object max value

            data_Rn[i] = Rn
            data_RnVn[i] = (Rn * Rn) / Rmax

            if area_prop:
                obj_cm_y[i] = obj.centroid[0]
                obj_cm_x[i] = obj.centroid[1]
                obj_areas[i] = obj.area
                obj_mxint[i] = obj.max_intensity
                obj_int_ratio[i] = obj.max_intensity/obj.mean_intensity
                obj_major_ax[i] = obj.major_axis_length
                obj_minor_ax[i] = obj.minor_axis_length
                obj_extent[i] = obj.extent
                if lsmask is not None:
                    assert lsmask.shape == darray.shape,\
                            """--- ERROR ---\n Land sea mask must have same """
                    """dimension as input data! Now {} vs {}""".format(
                        lsmask.shape, darray.shape)
                    objmask = lbl == obj.label
                    ocean_pts = lsmask[objmask].sum()
                    ratio = ocean_pts/obj.area
                    if ratio >= 0.90:
                        objid = 0
                    elif ratio <= 0.10:
                        objid = 1
                    else:
                        objid = 2
                    obj_ident[i] = objid

        if area_prop:
            obj_dict = {'cm_x': obj_cm_x, 'cm_y': obj_cm_y, 'size': obj_areas,
                        'extent': obj_extent, 'major axis': obj_major_ax,
                        'minor axis': obj_minor_ax, 'max intensity': obj_mxint,
                        'intensity ratio': obj_int_ratio}
            if lsmask is not None:
                obj_dict.update({'obj_id': obj_ident})

        # Calculate the weighted mean of all objects' scaled volume
        data_V = np.nansum(data_RnVn) / np.nansum(data_Rn)

        return data_V, obj_dict

    # Get object properties
    data_vol, area_props = get_object_properties(data, data_label, obj_prop)
    refdata_vol, ref_area_props = get_object_properties(refdata, refdata_label,
                                                        obj_prop)

    # S component
    S = (data_vol - refdata_vol) / (0.5 * (data_vol + refdata_vol))

    return S, {'data': area_props, 'refdata': ref_area_props}


def L_stat(data, data_label, refdata, refdata_label):
    """
    Function to determine the location component (L). It consists of two
    components, L1 and L2.
    L1:
        measures the normalized distance between the centers of mass of the
        modelled and observed fields.
    L2:
        The second considers the averaged distance between the center of mass
        of the total field and individual field objects.

    Parameters
    ----------
    data, refdata: arrays
        2D data arrays to be compared, where refdata is the reference data e.g.
        observations.
    data_label, refdata_label: arrays
        Arrays with labeled objects. Returned from the label() function.

    Returns
    -------
    L1, L2, L: dictionary
        Dictionary with the calculated location components L1 and L2 as well as
        its composite L (L1 + L2).
    """

    from scipy.ndimage.measurements import center_of_mass
    from scipy.spatial.distance import pdist

    def calc_L2(darray, lbl, data_CM):
        # Properties of identified rain areas
        props = regionprops(lbl, darray)

        # Number of rain areas
        nobj = len(props)
        obj_Rn = np.zeros(nobj)
        obj_CM = np.zeros(nobj)
        for i, obj in enumerate(props):
            obj_data = np.array([darray[tuple(coord)] for coord in obj.coords])

            Rn = obj_data.sum()      # Object mean value
            R_CM = distfunc(center_of_mass(darray, lbl, obj.label))

            obj_Rn[i] = Rn
            obj_CM[i] = R_CM

        R = np.nansum((obj_Rn * np.abs(obj_CM - data_CM))) / np.nansum(obj_Rn)

        return R

    # Center of mass of modelled total field: X=1/R * sum( Rn * dn ),
    # where R is domain total field, Rn object total field and
    # dn object distance to reference point.

    # Center of mass for total field
    data_CM = distfunc(center_of_mass(data, data_label))
    refdata_CM = distfunc(center_of_mass(refdata, refdata_label))

    # Find maximum distance in domain
    fld_mask = data.mask if isinstance(data, np.ma.MaskedArray) \
        else ~np.isnan(data)
    poly = find_contours(fld_mask, 0.9)
    if poly:
        bounds = poly[0].astype(int)
        distances = pdist(bounds)
        dmax = distances.max()
    else:
        wrn_msg = """WARNING -- No domain data mask found, using the whole input
        array to define maximum distance 'dmax'."""
        print("\n{}\n".format(wrn_msg))
        dmax = distfunc(fld_mask.shape)

    L1 = np.abs(data_CM - refdata_CM) / dmax

    # Calculate L2 component
    R_data = calc_L2(data, data_label, data_CM)
    R_refdata = calc_L2(refdata, refdata_label, refdata_CM)

    L2 = 2 * (np.abs(R_data - R_refdata) / dmax)

    # Add L1 and L2
    L = L1 + L2

    return {'L': L, 'L1': L1, 'L2': L2}


def threshold(data, thr_type, value):
    """
    Function to calculate the threshold to be used to identify objects.

    Parameters
    ----------
    data: array
        2D data array.
    thr_type: string
        Type of threshold. Can be either "S" for specified (any absolute
        value), "F" for a fraction (between 0 and 1) of the maximum value, and
        "P" for a percentile (95 for the 95th percentile etc).
    value: int/float
        The corresponding value based on the chosen threhold type.
    """

    if (thr_type == "S"):
        T = value
    elif (thr_type == "F"):
        if isinstance(data, np.ma.MaskedArray):
            T = value * np.ma.max(data)
        else:
            T = value * np.nanmax(data)
    elif (thr_type == "P"):
        if isinstance(data, np.ma.MaskedArray):
            T = np.percentile(data.compressed(), value)
        else:
            T = np.percentile(data.ravel(), value)

    # print("Threshold: {}{} --> value: {}".format(thr_type, value, T))

    return T


def distfunc(x):
    """ Calculate distances """
    return np.sqrt(x[0]**2 + x[1]**2)


def remove_large_objects(segments, max_size):
    """
    Remove large objects based on the maximum size limit defined by 'max_size'.

    Parameters
    ----------
    segments: array
        Array with labeled objects. Returned from the label() function.
    max_size: int
        Maximum size (number of grid points)

    Returns
    -------
    out: array
        The segments array with too large objects removed.
    """
    out = np.copy(segments)
    component_sizes = np.bincount(segments.ravel())
    too_large = component_sizes > max_size
    too_large_mask = too_large[segments]
    out[too_large_mask] = 0

    return out


def sal_calc(tstep, data, refdata, thr_t, thr_v, obj_prop=True, olsl=None,
             ousl=None, smlvl=None, land_sea_mask=None):
    """
    Perform the SAL calculation using the S, A, L functions.
    """
    import convolve
    import warnings

    print("\nCalculating SAL stats for time step: {}\n".format(tstep+1))

    # Smoothening of data
    if smlvl is not None:
        kernel = convolve.kernel_gen(smlvl)
        indata = convolve.convolve2Dfunc(data, kernel, fft=False)
        inrefdata = convolve.convolve2Dfunc(refdata, kernel, fft=False)
    else:
        indata = data
        inrefdata = refdata

    # Calculate threshold
    thr_data = threshold(indata, thr_t, thr_v)
    thr_rdata = threshold(inrefdata, thr_t, thr_v)

    # ---- Identify rain areas ---- #
    mask = np.ones(data.shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data_mask = np.logical_and(mask, indata > thr_data)
        refdata_mask = np.logical_and(mask, inrefdata > thr_rdata)

    # Identify separate rain areas
    data_lbl = label(data_mask)
    refdata_lbl = label(refdata_mask)

    if olsl is not None:
        data_lbl = remove_small_objects(data_lbl, olsl)
        refdata_lbl = remove_small_objects(refdata_lbl, olsl)
    if ousl is not None:
        data_lbl = remove_large_objects(data_lbl, ousl)
        refdata_lbl = remove_large_objects(refdata_lbl, ousl)

    if np.all((np.any(data_lbl.astype(bool)),
               np.any(refdata_lbl.astype(bool)))):
        A = A_stat(indata, inrefdata)
        S, obj_prop_dict = S_stat(indata, data_lbl, inrefdata, refdata_lbl,
                                  obj_prop=obj_prop, lsmask=land_sea_mask)
        L = L_stat(indata, data_lbl, inrefdata, refdata_lbl)

        if obj_prop:
            sal_dict = {'S': S, 'A': A, 'L': L, 'obj props': obj_prop_dict}
        else:
            sal_dict = {'S': S, 'A': A, 'L': L}
    else:
        print('Time step has no objects in either or both data sets ...')
        print('Moving on ...')

        sal_dict = None

    return tstep, sal_dict


def write_to_disk(ddict, nt, fname, attrs):
    if fname is None:
        from datetime import datetime
        dtime = datetime.now().strftime('%Y-%m-%dT%H%M%S')
        fn = './sal_analysis_' + dtime + '.nc'
    else:
        fn = fname

    # L component
    L, L1, L2 = zip(*[tuple(ld.values()) for ld in ddict['L']])

    # Object properties
    prop_dict = {}
    for dname in ('data', 'refdata'):
        prop_data = zip(*[tuple(ld[dname].values())
                          for ld in ddict['obj props']])
        prop_dict[dname] =\
            np.array([[item for sublist in ll for item in sublist]
                     for ll in prop_data])
    # Coordinates
    prop_names = list(ddict['obj props'][0]['data'].keys())
    data_nobj = np.arange(prop_dict['data'].shape[1])
    refdata_nobj = np.arange(prop_dict['refdata'].shape[1])

    # Extract object identity
    ds = xa.Dataset({
        "Structure_component": (['t'], ddict['S']),
        "Amplitude_component": (['t'], ddict['A']),
        "Location_total_component": (['t'], list(L)),
        "Location_L1_component": (['t'], list(L1)),
        "Location_L2_component": (['t'], list(L2)),
        "data_area_props": (['p', 'dnobj'], prop_dict['data']),
        "refdata_area_props": (['p', 'rdnobj'], prop_dict['refdata'])},
        coords={"tsteps": (['t'], np.arange(1, nt+1)),
                "object_properties": (['p'], prop_names),
                "N_objects_data": (['dnobj'], data_nobj),
                "N_objects_refdata": (['rdnobj'], refdata_nobj)})

    ds.attrs = attrs

    obj_id_comment = """Identified objects may be interpreted as land, """
    """ocean or coastal objects, if a land-sea mask has been provided """
    """to the analysis. These identifications are represented by an """
    """integer: ocean (0), land (1), coastal (2)."""
    ds.attrs['Comment'] = obj_id_comment

    # Write to disk
    ds.to_netcdf(fn)


def run_sal_analysis(data, refdata, thr_type, thr_val, obj_prop=True,
                     obj_lower_size_limit=None, obj_upper_size_limit=None,
                     smoothening_data_level=None, land_sea_mask=None,
                     write_to_file=False, filename=None, nproc=1):
    """
    Run the SAL analysis on the two data sets 'data' and 'refdata', where the
    latter is supposed to represent the 'truth'.

    Parameters
    ----------
    data/refdata: arrays
        2D data arrays with zeroth dimension representing time steps. Both data
        sets must have the same dimension sizes, i.e. both in time and space.
    thr_type: string
        Type of threshold to use. See 'threshold' function for more
        information.
    thr_val: float/int
        Value of threshold.
    obj_prop: boolean
        If True (default), a number of object area properties are returned for
        each of the identified objects. See 'S_stat' function for more
        information.
    obj_lower_size_limit: int
        If set, all objects with an area (number of connected grid points)
        lower than the value set is removed from analysis.
    obj_upper_size_limit: int
        If set, all objects with an area (number of connected grid points)
        greater than the value set is removed from analysis.
    smoothening_data_level: int
        If set, the number represents the # of grid points of the side of a
        moving window used to smooth the data arrays. Mean value within window
        is calculated.
    land_sea_mask: array/None
        If set, land_sea_mask must be a 2d boolean array with same dimension as
        input data. The land/sea-mask is then used to identify objects as
        either land (1), ocean (0) or coastal (2) in the object
        properties dictionary. Thus, mask only used if obj_prop=True.
        N.B. Mask must have True for ocean points and False for land points.
    write_to_file: boolean
        Whether to write results to disk.
    filename: str
        Name of file for writing to disk.
    nproc: int
        Number of processors to use in calculation. If larger than 1 (default),
        the multiprocessing module is used to distribute the calculation in the
        time dimension.

    Returns
    -------
    out_dict: dictionary
        Dictionary with calculated SAL statistics and area properties (if
        obj_prop is set to True).
    nc: file
        If 'write_to_file' is True, results are written to disk in a netcdf
        file.
    """
    from collections import defaultdict

    err_msg = """\n#--- ERROR ! ---#\nData must be numpy array or numpy masked
    array."""
    assert isinstance(data, (np.ndarray, np.ma.MaskedArray)), err_msg
    err_msg = """\n#--- ERROR ! ---#\nReference data must be numpy array or
    numpy masked array."""
    assert isinstance(refdata, (np.ndarray, np.ma.MaskedArray)), err_msg

    if np.squeeze(data).ndim > 2:
        nt = data.shape[0]
    else:
        nt = 1
        if data.ndim == 2:
            data = np.expand_dims(data, 0)
        if refdata.ndim == 2:
            refdata = np.expand_dims(refdata, 0)

    if nproc > 1:

        nr_procs_set = np.minimum(nproc, mp.cpu_count())

        pool = mp.Pool(processes=nr_procs_set)
        computations = [pool.apply_async(sal_calc,
                        args=(t, data[t, :], refdata[t, :], thr_type, thr_val,
                              obj_prop, obj_lower_size_limit,
                              obj_upper_size_limit, smoothening_data_level,
                              land_sea_mask)) for t in range(nt)]
        pool.close()
        pool.join()

        outp = [k.get() for k in computations]
        outp.sort()

    else:
        outp = [sal_calc(t, data[t, :], refdata[t, :], thr_type, thr_val,
                         obj_prop, obj_lower_size_limit, obj_upper_size_limit,
                         smoothening_data_level, land_sea_mask)
                for t in range(nt)]
    if nt > 1:
        out_dict = defaultdict(list)
        no_data = 0
        for i, d in outp:
            if d is not None:
                for key, value in d.items():
                    out_dict[key].append(value)
            else:
                no_data += 1
        nt = nt - no_data
    else:
        out_dict = outp[0][1]
        if out_dict is None:
            print("No precip data detected for analysis")
            sys.exit()

    if write_to_file:
        attrs = {
            "threshold type": thr_type, "threshold value": thr_val,
            "object size lower limit":
            obj_lower_size_limit if obj_lower_size_limit is not None else "-",
            "object size upper limit":
            obj_upper_size_limit if obj_upper_size_limit is not None else "-",
            "smoothening level (# grid points)":
            smoothening_data_level**2 if smoothening_data_level is not None
            else "-",
        }

        write_to_disk(out_dict, nt, filename, attrs)

    return out_dict
