""" Module script for plotting """

import os
import sys
import xarray as xa
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
import rcatool.plot.plots as rpl
from rcatool.utils.polygons import mask_region
from rcatool.stats.arithmetics import run_mean
from copy import deepcopy

# Colors
set1 = mpl.cm.Set1.colors
set1_m = [[int(255*x) for x in triplet] for triplet in set1]
rel_colors = ['#{:02x}{:02x}{:02x}'.format(s[0], s[1], s[2]) for s in set1_m]
abs_colors = rel_colors[:]
abs_colors.insert(-1, 'k')


def plot_main(pltdict, statistic):
    """
    Main function that controls the plotting procedure.
    """
    pdict = deepcopy(pltdict)
    models = pdict['models']
    nmod = len(models)
    ref_model = models[0]
    obs = pdict['observation']
    var = pdict['variable']
    tres = pdict['time res']
    tstat = pdict['stat method']
    units = pdict['units']
    time_suffix_dd = pdict['time suffix dict']
    regions = pdict['regions']
    img_dir = pdict['img dir']

    grid_coords = pdict['grid coords']
    moments_plot_conf = pdict['moments plot config']
    map_projection = pdict['map projection']
    map_config = pdict['map config']
    map_extent = pdict['map extent']
    map_gridlines = pdict['map gridlines']
    map_axes_conf = _map_grid_setup(
        pdict['map grid setup']
    )
    map_domain = pdict['map domain']
    map_plot_conf = pdict['map plot kwargs']

    line_grid = pdict['line grid setup']
    line_sets = pdict['line kwargs']

    fm_list = pdict['mod files'][statistic]
    fo_list = pdict['obs files'][statistic]
    if regions is not None:
        fm_listr = pdict['mod reg files'][statistic]
        fo_listr = pdict['obs reg files'][statistic]
    else:
        fm_listr = None
        fo_listr = None

    _plots(statistic)(
        fm_list, fo_list, fm_listr, fo_listr, models, nmod, ref_model, obs,
        var, tres, tstat, units, time_suffix_dd, regions, img_dir, grid_coords,
        moments_plot_conf, map_domain, map_projection, map_config, map_extent,
        map_gridlines, map_axes_conf, map_plot_conf, line_grid, line_sets)


def _map_grid_setup(map_grid_set):
    """
    Potentially modify map grid settings for map plots
    """
    if 'cbar_mode' not in map_grid_set:
        map_grid_set.update({'cbar_mode': 'each'})
        map_grid_set.update({'cbar_pad': 0.03})
    if 'axes_pad' not in map_grid_set:
        map_grid_set.update({'axes_pad': 0.5})

    return map_grid_set


def _plots(stat):
    p = {
        'seasonal cycle': map_season,
        'annual cycle': map_ann_cycle,
        'percentile': map_pctls,
        'diurnal cycle': map_diurnal_cycle,
        'pdf': pdf_plot,
        'moments': moments_plot,
        'asop': map_asop,
    }
    return p[stat]


def _round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier


def _round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier


def _mask_data(ds_in, var, mask):
    mdata = ds_in[var].where(mask)
    ds_out = mdata.to_dataset()
    ds_out.attrs = ds_in.attrs
    return ds_out


def round_to_sign_digits(x, sig=2):
    out = round(x, sig-int(np.floor(np.log10(abs(x))))-1) if not x == 0 else 0
    return out


def get_clevs(data, centered=False):
    from scipy.stats import skew
    if centered:
        abs_max = np.nanpercentile(data, 98)
        abs_min = np.nanpercentile(data, 2)
    else:
        if skew(data.ravel()[~np.isnan(data.ravel())]) > 2:
            abs_max = np.nanpercentile(data, 99)
            abs_min = np.nanmin(data)
        else:
            # abs_max = np.nanmax(data)
            abs_max = np.nanpercentile(data, 97)
            abs_min = np.nanmin(data)
    if centered:
        abs_max = np.maximum(abs(abs_max), abs(abs_min))
        abs_min = -abs_max
    if ((abs(abs_max) < 1.0) & (abs(abs_min) < 1.0)):
        nz = np.floor(np.abs(np.log10(np.abs(abs_max))))
        round_max = _round_up(abs_max, nz+1)
        if abs_min == 0.0:
            round_min = 0.0
        else:
            nz = np.floor(np.abs(np.log10(np.abs(abs_min))))
            round_min = _round_down(abs_min, nz+1)
    elif ((abs_max < 10) & (abs_min < 10)):
        round_max = _round_up(abs_max, 0)
        round_min = _round_down(abs_min, 0)
    else:
        round_max = _round_up(abs_max, -1)
        round_min = _round_down(abs_min, -1)

    diff = round_max - round_min
    if diff < 1:
        # nsteps = 15
        # nz = np.floor(np.abs(np.log10(np.abs(diff))))
        # nz = round(diff, -int(np.floor(np.log10(abs(diff)))))
        # step = _round_down(diff/nsteps, nz+2)
        _clevs = np.linspace(round_min, round_max, 14)
        clevs = [round_to_sign_digits(x, 2) for x in _clevs]
    else:
        if 1 <= diff < 10:
            step = .5
        elif 10 <= diff < 25:
            step = 1
        elif 25 <= diff < 50:
            step = 2.5
        elif 50 <= diff < 150:
            step = 5
        elif 150 <= diff < 500:
            step = 10
        else:
            step = 20
        clevs = np.arange(round_min, round_max + step, step)
    return clevs


def _get_colorbar_label_formatting(clevs):
    decimals = [str(x).split('.')[1] for x in clevs]
    number_of_decimals = [len(x) for x in decimals]
    max_decimals = int(np.max(number_of_decimals))
    if max_decimals > 1:
        fmt = "".join(['{:.', f'{max_decimals}', 'f}'])
    elif max_decimals == 1:
        if np.all([x == '0' for x in decimals]):
            fmt = '{:.0f}'
        else:
            fmt = '{:.1f}'

    return fmt


def map_season(fm_list, fo_list, fm_listr, fo_listr, models, nmod, ref_model,
               obs, var, tres, tstat, units, time_suffix_dd, regions, img_dir,
               grid_coords, moments_plot_conf, map_domain, map_projection,
               map_config, map_extent, map_gridlines, map_axes_conf,
               map_plot_conf, line_grid, line_sets):
    """
    Plotting seasonal mean map plot
    """

    ref_mnme = ref_model.upper()
    othr_mod = models.copy()
    othr_mod.remove(ref_model)

    # Obs data list
    obslist = [obs] if not isinstance(obs, list) else obs

    # Map settings
    target_grid_names = list(grid_coords['target grid'][var]['lon'].keys())
    tgname = target_grid_names[0]
    domain_model = map_domain if map_domain else ref_model
    domain = grid_coords['meta data'][var][domain_model]['domain']
    mask = mask_region(
        grid_coords['target grid'][var]['lon'][tgname],
        grid_coords['target grid'][var]['lat'][tgname], domain)

    # Data
    fmod = {m: xa.open_dataset(f) for m, f in zip(models, fm_list)}
    fmod_msk = {m: _mask_data(ds, var, mask) for m, ds in fmod.items()}

    if obslist[0] is not None:
        obslbl = "_".join(s for s in obslist)
        ref_obs = obslist[0]
        fobs = {o: xa.open_dataset(f) for o, f in zip(obslist, fo_list)}
        fobs_msk = {o: _mask_data(ds, var, mask) for o, ds in fobs.items()}
        dlist = [fobs_msk[ref_obs][var].values[i, :] for i in range(4)] +\
                [fmod_msk[m][var].values[i, :] -
                 fobs_msk[ref_obs][var].values[i, :]
                 for m in models for i in range(4)]
        if len(obslist) > 1:
            dlist += [fobs_msk[o][var].values[i, :] -
                      fobs_msk[ref_obs][var].values[i, :]
                      for o in obslist[1:] for i in range(4)]
            ndata = nmod + len(obslist[1:])
            ftitles = [f"{ref_obs} {time_suffix_dd[ref_obs]}"] +\
                [(f"{m} {time_suffix_dd[m]} -\n "
                  f"{ref_obs} {time_suffix_dd[ref_obs]}")
                 for m in models + obslist[1:]]
        else:
            ndata = nmod
            ftitles = [f"{ref_obs} {time_suffix_dd[ref_obs]}"] +\
                [(f"{m} {time_suffix_dd[m]} -\n "
                  f"{ref_obs} {time_suffix_dd[ref_obs]}")
                 for m in models]
    else:
        dlist = [fmod_msk[ref_model][var].values[i, :] for i in range(4)] +\
                [fmod_msk[m][var].values[i, :] -
                 fmod_msk[ref_model][var].values[i, :]
                 for m in othr_mod for i in range(4)]
        ndata = nmod-1
        ftitles = [f"{ref_mnme} {time_suffix_dd[ref_model]}"] +\
            [(f"{m} {time_suffix_dd[m]} -\n "
              f"{ref_mnme} {time_suffix_dd[ref_model]}")
             for m in othr_mod]

    # figure settings
    figshape = (ndata + 1, 4)
    if np.prod(figshape) > 8:
        figsize = (22, 14)
    else:
        figsize = (20, 9)

    tsuffix_ll = [val for key, val in time_suffix_dd.items()]
    tsuffix_fname = "_vs_".join(set(tsuffix_ll))
    thr = fmod_msk[ref_model].attrs['Description'].\
        split('|')[2].split(':')[1].strip()
    if thr != 'None':
        headtitle = '{} [{}] | Threshold: {}'.format(
            var, units, thr)
        if obs is not None:
            fn = '{}_thr{}{}{}_map_seasonal_cycle_model_{}_{}.png'.format(
                var, thr, tres, tstat, obslbl, tsuffix_fname)
        else:
            fn = '{}_thr{}{}{}_map_seasonal_cycle_model_{}.png'.format(
                var, thr, tres, tstat, tsuffix_fname)
    else:
        headtitle = '{} [{}]'.format(var, units)
        if obs is not None:
            fn = '{}{}{}_map_seasonal_cycle_model_{}_{}.png'.format(
                var, tres, tstat, obslbl, tsuffix_fname)
        else:
            fn = '{}{}{}_map_seasonal_cycle_model_{}.png'.format(
                var, tres, tstat, tsuffix_fname)

    if var == 'pr':
        cmap = [mpl.cm.YlGnBu]*4 + [mpl.cm.BrBG]*ndata*4
    else:
        cmap = [mpl.cm.Spectral_r]*4 + [mpl.cm.RdBu_r]*ndata*4

    clevs_abs = get_clevs(np.array(dlist[0:4]), centered=False)
    clevs_rel = get_clevs(np.array(dlist[4:8]), centered=True)
    clevs = [clevs_abs]*4 + [clevs_rel]*ndata*4

    # TBD: Try to find appropriate formatting for color bar
    # cb_fmt_abs = [str(f)[::-1].find('.') for f in clevs_abs]
    # cb_fmt_rel = [str(f)[::-1].find('.') for f in clevs_rel]

    rpl.figure_init(plottype='map')

    # Create map object and axes grid
    map_proj = rpl.define_map_object(map_projection, **map_config)
    fig, axs_grid = rpl.map_setup(
        map_proj, map_extent, figsize, figshape, grid_lines=map_gridlines,
        **map_axes_conf)

    lts = grid_coords['target grid'][var]['lat'][tgname]
    lns = grid_coords['target grid'][var]['lon'][tgname]

    # Plot the maps
    mp = rpl.make_map_plot(
        dlist, axs_grid, lts, lns, cmap=cmap, clevs=clevs, **map_plot_conf)
    rpl.image_colorbar(mp, axs_grid, labelspacing=2, formatter='{:.1f}')

    # Add contour plot if mslp
    if var == 'psl':
        rpl.make_map_plot(dlist, axs_grid, lts, lns,  clevs=clevs,
                          filled=False, colors='#4f5254', linewidths=1.3)

    # Map settings
    rpl.map_axes_settings(fig, axs_grid, headtitle=headtitle,
                          time_mean='season')

    # Annotate
    [ax.text(-0.08, 0.5, ft.upper(), va='center', ha='center',
             rotation=90, transform=ax.transAxes)
     for ft, ax in zip(ftitles, [axs_grid[i]
                                 for i in [p*4 for p in range(ndata+1)]])]

    plt.savefig(os.path.join(img_dir, fn), bbox_inches='tight')


def map_ann_cycle(fm_list, fo_list, fm_listr, fo_listr, models, nmod,
                  ref_model, obs, var, tres, tstat, units, time_suffix_dd,
                  regions, img_dir, grid_coords, moments_plot_conf, map_domain,
                  map_projection, map_config, map_extent, map_gridlines,
                  map_axes_conf, map_plot_conf, line_grid, line_sets):
    """
    Plotting annual cycle map plot
    """

    ref_mnme = ref_model.upper()
    othr_mod = models.copy()
    othr_mod.remove(ref_model)

    # Obs data list
    obslist = [obs] if not isinstance(obs, list) else obs

    # Map settings
    target_grid_names = list(grid_coords['target grid'][var]['lon'].keys())
    tgname = target_grid_names[0]
    domain_model = map_domain if map_domain else ref_model
    domain = grid_coords['meta data'][var][domain_model]['domain']
    mask = mask_region(
        grid_coords['target grid'][var]['lon'][tgname],
        grid_coords['target grid'][var]['lat'][tgname], domain)

    # Data
    fmod = {m: xa.open_dataset(f) for m, f in zip(models, fm_list)}
    fmod_msk = {m: _mask_data(ds, var, mask) for m, ds in fmod.items()}

    if obslist[0] is not None:
        ref_obs = obslist[0]
        fobs = {o: xa.open_dataset(f) for o, f in zip(obslist, fo_list)}
        fobs_msk = {o: _mask_data(ds, var, mask) for o, ds in fobs.items()}

        dlist = [[fobs_msk[ref_obs][var].values[i, :] for i in range(12)]] +\
                [[fmod_msk[m][var].values[i, :] -
                  fobs_msk[ref_obs][var].values[i, :] for i in range(12)]
                 for m in models]
        if len(obslist) > 1:
            dlist += [[fobs_msk[o][var].values[i, :] -
                       fobs_msk[ref_obs][var].values[i, :]
                      for i in range(12)] for o in obslist[1:]]
            ndata = nmod + len(obslist[1:])
            ftitles = [f"{ref_obs} {time_suffix_dd[ref_obs]}"] +\
                [(f"{m} {time_suffix_dd[m]} -\n "
                  f"{ref_obs} {time_suffix_dd[ref_obs]}")
                 for m in models + obslist[1:]]
            data_names = [ref_obs] + [f"{m}-{ref_obs}"
                                      for m in models + obslist[1:]]
        else:
            ndata = nmod
            ftitles = [f"{ref_obs} {time_suffix_dd[ref_obs]}"] +\
                [(f"{m} {time_suffix_dd[m]} -\n "
                  f"{ref_obs} {time_suffix_dd[ref_obs]}")
                 for m in models]
            data_names = [ref_obs] + [f"{m}-{ref_obs}" for m in models]
    else:
        dlist = [[fmod_msk[ref_model][var].values[i, :] for i in range(12)]] +\
                [[fmod_msk[m][var].values[i, :] -
                 fmod_msk[ref_model][var].values[i, :] for i in range(12)]
                 for m in othr_mod]
        ndata = nmod-1
        ftitles = [f"{ref_mnme} {time_suffix_dd[ref_model]}"] +\
            [(f"{m} {time_suffix_dd[m]} -\n "
              f"{ref_mnme} {time_suffix_dd[ref_model]}")
             for m in othr_mod]
        data_names = [ref_mnme] + [f"{m}-{ref_mnme}" for m in othr_mod]

    tsuffix_ll = [val for key, val in time_suffix_dd.items()]
    tsuffix_fname = "_vs_".join(set(tsuffix_ll))
    thr = fmod_msk[ref_model].attrs['Description'].\
        split('|')[2].split(':')[1].strip()

    # figure settings
    figsize = (16, 18)
    figshape = (4, 3)

    if var == 'pr':
        cmap = [mpl.cm.YlGnBu] + [mpl.cm.BrBG]*ndata
    else:
        cmap = [mpl.cm.Spectral_r] + [mpl.cm.RdBu_r]*ndata

    clevs_abs = get_clevs(np.array(dlist[0]), centered=False)
    clevs_rel = get_clevs(np.array(dlist[1]), centered=True)
    clevs = [clevs_abs] + [clevs_rel]*ndata

    lts = grid_coords['target grid'][var]['lat'][tgname]
    lns = grid_coords['target grid'][var]['lon'][tgname]

    # Loop over data sets
    for p, data_name in zip(range(ndata + 1), data_names):
        if thr != 'None':
            headtitle = '{} | {} [{}] | Threshold: {}'.format(
                ftitles[p], var, units, thr)
            fn = '{}_thr{}{}{}_map_ann_cycle_{}_{}.png'.format(
                var, thr, tres, tstat, data_name, tsuffix_fname)
        else:
            headtitle = '{} | {} [{}]'.format(
                ftitles[p], var, units)
            fn = '{}{}{}_map_ann_cycle_{}_{}.png'.format(
                var, tres, tstat, data_name, tsuffix_fname)

        rpl.figure_init(plottype='map')

        # Create map object and axes grid
        map_proj = rpl.define_map_object(map_projection, **map_config)
        fig, axs_grid = rpl.map_setup(
            map_proj, map_extent, figsize, figshape, grid_lines=map_gridlines,
            **map_axes_conf)

        # Plot the maps
        mp = rpl.make_map_plot(
            dlist[p], axs_grid, lts, lns, cmap=cmap[p], clevs=clevs[p],
            **map_plot_conf)
        rpl.image_colorbar(mp, axs_grid, labelspacing=2, formatter='{:.1f}')

        # Add contour plot if mslp
        if var == 'psl':
            rpl.make_map_plot(
                dlist[p], axs_grid, lts, lns, clevs=clevs[p], filled=False,
                colors='#4f5254', linewidths=1.3)
        # [plt.clabel(mm, fmt='%.1f', colors='k', fontsize=15) for mm in lp]

        # Map settings
        rpl.map_axes_settings(fig, axs_grid, fontsize='large',
                              headtitle=headtitle, time_mean='month')

        plt.savefig(os.path.join(img_dir, fn), bbox_inches='tight')

    # Line plot annual cycle
    if fm_listr is not None:
        line_ann_cycle(fm_listr, fo_listr, models, nmod, ref_model, obs, var,
                       tres, tstat, units, time_suffix_dd, regions, img_dir,
                       line_grid, line_sets)


def line_ann_cycle(fm_list, fo_list, models, nmod, ref_model, obs, var, tres,
                   tstat, units, time_suffix_dd, regions, img_dir, line_grid,
                   line_sets):
    """
    Plotting annual cycle line plot
    """

    ref_mnme = ref_model.upper()
    othr_mod = models.copy()
    othr_mod.remove(ref_model)

    for reg in regions:

        fmod = {m: xa.open_dataset(f) for m, f in zip(models, fm_list[reg])}
        mod_ann = {m: np.nanmean(fmod[m][var].values, axis=(1, 2))
                   for m in models}
        if obs is not None:
            obslist = [obs] if not isinstance(obs, list) else obs
            ref_obs = obslist[0]
            obslbl = "_".join(s for s in obslist)
            fobs = {o: xa.open_dataset(f) for o, f in zip(obslist,
                                                          fo_list[reg])}
            obs_ann = {o: np.nanmean(fobs[o][var].values, axis=(1, 2))
                       for o in obslist}
            dlist = [[obs_ann[ref_obs]] + [mod_ann[m] for m in models],
                     [mod_ann[m] - obs_ann[ref_obs] for m in models]]

            if len(obslist) > 1:
                dlist[0] += [obs_ann[o] for o in obslist[1:]]
                dlist[1] += [obs_ann[o] - obs_ann[ref_obs]
                             for o in obslist[1:]]
                ll_nms = models + obslist[1:]
            else:
                ll_nms = models
            lg_lbls = [[ref_obs] + [m.upper() for m in ll_nms],
                       ['{} - {}'.format(m.upper(), ref_obs) for m in ll_nms]]
        else:
            dlist = [[mod_ann[m] for m in models],
                     [mod_ann[m] - mod_ann[ref_model] for m in othr_mod]]
            lg_lbls = [[m.upper() for m in models], ['{} - {}'.format(
                m.upper(), ref_mnme) for m in othr_mod]]

        tsuffix_ll = [val for key, val in time_suffix_dd.items()]
        tsuffix_fname = "_vs_".join(set(tsuffix_ll))
        thr = fmod[ref_model].attrs['Description'].\
            split('|')[2].split(':')[1].strip()
        regnm = reg.replace(' ', '_')

        if thr != 'None':
            headtitle = '{} | Threshold: {} | {}'.format(var, thr, reg)
            if obs is not None:
                fn = '{}_thr{}{}{}_lnplot_ann_cycle_{}_model_{}_{}.png'.\
                    format(var, thr, tres, tstat, regnm, obslbl,
                           tsuffix_fname)
            else:
                fn = '{}_thr{}{}{}_lnplot_ann_cycle_{}_model_{}.png'.\
                    format(var, thr, tres, tstat, regnm, tsuffix_fname)
        else:
            headtitle = '{} | {}'.format(var, reg)
            if obs is not None:
                fn = '{}{}{}_lnplot_ann_cycle_{}_model_{}_{}.png'.\
                    format(var, tres, tstat, regnm, obslbl, tsuffix_fname)
            else:
                fn = '{}{}{}_lnplot_ann_cycle_{}_model_{}.png'.\
                    format(var, tres, tstat, regnm, tsuffix_fname)

        # figure settings
        figsize = (16, 7)
        figshape = (1, 2)

        ylabel = ['Monthly mean ({})'.format(units),
                  'Difference ({})'.format(units)]
        xlabel = [None]*2
        xlim = [[-.5, 11.5]]*2
        xticks = range(12)
        xtlbls = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
                  'Sep', 'Oct', 'Nov', 'Dec']

        rpl.figure_init(plottype='line')
        fig, lgrid = rpl.fig_grid_setup(fshape=figshape, figsize=figsize,
                                        **line_grid)

        axs = rpl.make_line_plot(lgrid, ydata=dlist, **line_sets)

        [ln.set_color(lc) for ln, lc in zip(axs[0].get_lines(), abs_colors)]
        [ln.set_color(lc) for ln, lc in zip(
            list(axs[1].get_lines())[:len(dlist[1])], rel_colors)]
        # Legend
        legend_elements = [Line2D([0], [0], lw=2, color=c, label=l)
                           for c, l in zip(abs_colors, lg_lbls[0])]
        axs[0].legend(handles=legend_elements, fontsize='large')
        legend_elements = [Line2D([0], [0], lw=2, color=c, label=l)
                           for c, l in zip(rel_colors, lg_lbls[1])]
        axs[1].legend(handles=legend_elements, fontsize='large')

        [rpl.axes_settings(ax, xlabel=xlabel[a], xticks=xticks,
                           ylabel=ylabel[a], xtlabels=xtlbls, xlim=xlim[a])
         for a, ax in enumerate(axs)]

        ttl = fig.suptitle(headtitle, fontsize='xx-large')
        ttl.set_position((.5, 1.08))

        plt.savefig(os.path.join(img_dir, fn), bbox_inches='tight')


def map_pctls(fm_list, fo_list, fm_listr, fo_listr, models, nmod, ref_model,
              obs, var, tres, tstat, units, time_suffix_dd, regions, img_dir,
              grid_coords, moments_plot_conf, map_domain, map_projection,
              map_config, map_extent, map_gridlines, map_axes_conf,
              map_plot_conf, line_grid, line_sets):
    """
    Plotting percentile map plot
    """

    ref_mnme = ref_model.upper()
    othr_mod = models.copy()
    othr_mod.remove(ref_model)

    # Obs data list
    obslist = [obs] if not isinstance(obs, list) else obs

    # Map settings
    target_grid_names = list(grid_coords['target grid'][var]['lon'].keys())
    tgname = target_grid_names[0]
    domain_model = map_domain if map_domain else ref_model
    domain = grid_coords['meta data'][var][domain_model]['domain']
    mask = mask_region(
        grid_coords['target grid'][var]['lon'][tgname],
        grid_coords['target grid'][var]['lat'][tgname], domain)

    # Data
    fmod = {m: xa.open_dataset(f) for m, f in zip(models, fm_list)}
    fmod_msk = {m: _mask_data(ds, var, mask) for m, ds in fmod.items()}

    pctls = fmod[ref_model].percentiles.values
    npctl = pctls.size
    if obslist[0] is not None:
        obslbl = "_".join(s for s in obslist)
        ref_obs = obslist[0]
        fobs = {o: xa.open_dataset(f) for o, f in zip(obslist, fo_list)}
        fobs_msk = {o: _mask_data(ds, var, mask) for o, ds in fobs.items()}

        dlist = [[fobs_msk[ref_obs][var].values[i, :]] +
                 [fmod_msk[m][var].values[i, :] -
                  fobs_msk[ref_obs][var].values[i, :]
                  for m in models] for i in range(npctl)]

        if len(obslist) > 1:
            for i in range(npctl):
                dlist[i] += [fobs_msk[o][var].values[i, :] -
                             fobs_msk[ref_obs][var].values[i, :]
                             for o in obslist[1:]]
            ndata = nmod + len(obslist[1:])
            ftitles = [f"{ref_obs} {time_suffix_dd[ref_obs]}"] +\
                [(f"{m} {time_suffix_dd[m]} -\n "
                  f"{ref_obs} {time_suffix_dd[ref_obs]}")
                 for m in models + obslist[1:]]
        else:
            ndata = nmod
            ftitles = [f"{ref_obs} {time_suffix_dd[ref_obs]}"] +\
                [(f"{m} {time_suffix_dd[m]} -\n "
                  f"{ref_obs} {time_suffix_dd[ref_obs]}")
                 for m in models]
    else:
        dlist = [[fmod_msk[ref_model][var].values[i, :]] +
                 [fmod_msk[m][var].values[i, :] -
                  fmod_msk[ref_model][var].values[i, :]
                  for m in othr_mod] for i in range(npctl)]
        ndata = nmod-1
        ftitles = [f"{ref_mnme} {time_suffix_dd[ref_model]}"] +\
            [(f"{m} {time_suffix_dd[m]} -\n "
              f"{ref_mnme} {time_suffix_dd[ref_model]}")
             for m in othr_mod]

    tsuffix_ll = [val for key, val in time_suffix_dd.items()]
    tsuffix_fname = "_vs_".join(set(tsuffix_ll))
    thr = fmod[ref_model].attrs['Description'].\
        split('|')[2].split(':')[1].strip()
    # season = fmod[ref_model].attrs['Analysed time'].split(' ')[1]
    # seas = season.replace(' ', '')

    # figure settings
    if ndata + 1 < 3:
        figsize = (18, 12)
    else:
        figsize = (20, 10)
    figshape = (1, ndata+1)

    if var == 'pr':
        cmap = [mpl.cm.YlGnBu] + [mpl.cm.BrBG]*ndata
    else:
        cmap = [mpl.cm.Spectral_r] + [mpl.cm.RdBu_r]*ndata

    lts = grid_coords['target grid'][var]['lat'][tgname]
    lns = grid_coords['target grid'][var]['lon'][tgname]

    # Loop over percentiles
    for p in range(npctl):
        if thr != 'None':
            headtitle =\
                    '{} [{}] | p{} | Threshold: {} | {}'.\
                    format(var, units, pctls[p], thr, tsuffix_fname)
            if obs is not None:
                fn = '{}_thr{}{}_map_percentile_p{}_model_{}_{}.png'.\
                    format(var, thr, tres, pctls[p], obslbl,
                           tsuffix_fname)
            else:
                fn = '{}_thr{}{}_map_percentile_p{}_model_{}.png'.\
                    format(var, thr, tres, pctls[p], tsuffix_fname)
        else:
            headtitle = '{} [{}] | p{} | {}'.format(var, units, pctls[p],
                                                    tsuffix_fname)
            if obs is not None:
                fn = '{}{}_map_percentile_p{}_model_{}_{}.png'.format(
                    var, tres, pctls[p], obslbl, tsuffix_fname)
            else:
                fn = '{}{}_map_percentile_p{}_model_{}.png'.format(
                    var, tres, pctls[p], tsuffix_fname)

        rpl.figure_init(plottype='map')

        # Create map object and axes grid
        map_proj = rpl.define_map_object(map_projection, **map_config)
        fig, axs_grid = rpl.map_setup(
            map_proj, map_extent, figsize, figshape, grid_lines=map_gridlines,
            **map_axes_conf)

        clevs_abs = get_clevs(np.array(dlist[p][0]), centered=False)
        clevs_rel = get_clevs(np.array(dlist[p][1]), centered=True)

        fmt_abs = _get_colorbar_label_formatting(clevs_abs[::2])
        fmt_rel = _get_colorbar_label_formatting(clevs_rel[::2])

        clevs = [clevs_abs] + [clevs_rel]*ndata
        fmt = [fmt_abs] + [fmt_rel]*ndata

        # Plot the maps
        mp = rpl.make_map_plot(
            dlist[p], axs_grid, lts, lns, cmap=cmap, clevs=clevs,
            **map_plot_conf)
        rpl.image_colorbar(mp, axs_grid, labelspacing=2, formatter=fmt)

        # Map settings
        rpl.map_axes_settings(fig, axs_grid, headtitle=headtitle)

        [ax.text(0.5, 1.07, ft.upper(), size='x-large',
                 va='center', ha='center', transform=ax.transAxes)
         for ft, ax in zip(ftitles, axs_grid)]

        plt.savefig(os.path.join(img_dir, fn), bbox_inches='tight')


def map_diurnal_cycle(fm_list, fo_list, fm_listr, fo_listr, models, nmod,
                      ref_model, obs, var, tres, tstat, units, time_suffix_dd,
                      regions, img_dir, grid_coords, moments_plot_conf,
                      map_domain, map_projection, map_config, map_extent,
                      map_gridlines, map_axes_conf, map_plot_conf, line_grid,
                      line_sets):
    """
    Plotting diurnal cycle map plot
    """

    ref_mnme = ref_model.upper()
    othr_mod = models.copy()
    othr_mod.remove(ref_model)

    # Obs data list
    obslist = [obs] if not isinstance(obs, list) else obs

    # Map settings
    target_grid_names = list(grid_coords['target grid'][var]['lon'].keys())
    tgname = target_grid_names[0]
    domain_model = map_domain if map_domain else ref_model
    domain = grid_coords['meta data'][var][domain_model]['domain']
    mask = mask_region(
        grid_coords['target grid'][var]['lon'][tgname],
        grid_coords['target grid'][var]['lat'][tgname], domain)

    # Data
    fmod = {m: xa.open_dataset(f) for m, f in zip(models, fm_list)}
    fmod_msk = {m: _mask_data(ds, var, mask) for m, ds in fmod.items()}

    hours = fmod[ref_model].hour.values
    nhour = hours.size
    fshape = ((1, nhour) if nhour <= 6 else (2, int(np.ceil(nhour/2)))
              if 6 < nhour <= 12 else (3, int(np.ceil(nhour/3)))
              if 12 < nhour <= 18 else (4, int(np.ceil(nhour/4))))

    if obslist[0] is not None:
        ref_obs = obslist[0]
        fobs = {o: xa.open_dataset(f) for o, f in zip(obslist, fo_list)}
        fobs_msk = {o: _mask_data(ds, var, mask) for o, ds in fobs.items()}

        dlist = [[fobs_msk[ref_obs][var].values[i, :] for i in range(nhour)]]\
            + [[fmod_msk[m][var].values[i, :] -
                fobs_msk[ref_obs][var].values[i, :]
                for i in range(nhour)] for m in models]

        if len(obslist) > 1:
            dlist += [[fobs_msk[o][var].values[i, :] -
                       fobs_msk[ref_obs][var].values[i, :]
                       for i in range(nhour)] for o in obslist[1:]]
            ndata = nmod + len(obslist[1:])
            ftitles = [f"{ref_obs} {time_suffix_dd[ref_obs]}"] +\
                [(f"{m} {time_suffix_dd[m]} -\n "
                  f"{ref_obs} {time_suffix_dd[ref_obs]}")
                 for m in models + obslist[1:]]
            data_names = [ref_obs] + [f"{m}-{ref_obs}"
                                      for m in models + obslist[1:]]
        else:
            ndata = nmod
            ftitles = [f"{ref_obs} {time_suffix_dd[ref_obs]}"] +\
                [(f"{m} {time_suffix_dd[m]} -\n "
                  f"{ref_obs} {time_suffix_dd[ref_obs]}")
                 for m in models]
            data_names = [ref_obs] + [f"{m}-{ref_obs}" for m in models]
    else:
        dlist = [[fmod_msk[ref_model][var].values[i, :]
                  for i in range(nhour)]] +\
                [[fmod_msk[m][var].values[i, :] -
                  fmod_msk[ref_model][var].values[i, :]
                  for i in range(nhour)] for m in othr_mod]
        ndata = nmod-1
        ftitles = [ref_mnme] + [(f"{m} {time_suffix_dd[m]} -\n "
                                 f"{ref_mnme} {time_suffix_dd[ref_model]}")
                                for m in othr_mod]
        data_names = [ref_mnme] + [f"{m}-{ref_mnme}" for m in othr_mod]

    dcycle_stat = fmod[ref_model].attrs['Description'].\
        split('|')[1].strip()
    if dcycle_stat == 'Amount':
        dc_units = units
        fn_prfx = 'amnt'
    elif dcycle_stat == 'Frequency':
        dc_units = 'frequency'
        fn_prfx = 'freq'
    else:
        print("\n\tUnknown diurnal cycle statistic, exiting...")
        sys.exit()

    tsuffix_ll = [val for key, val in time_suffix_dd.items()]
    tsuffix_fname = "_vs_".join(set(tsuffix_ll))
    thr = fmod[ref_model].attrs['Description'].\
        split('|')[3].split(':')[1].strip()
    # season = fmod[ref_model].attrs['Analysed time'].split(' ')[1]
    # seas = season.replace(' ', '')

    # figure settings
    figsize = (22, 14)
    figshape = fshape

    if var == 'pr':
        cmap = [mpl.cm.YlGnBu] + [mpl.cm.BrBG]*ndata
    else:
        cmap = [mpl.cm.Spectral_r] + [mpl.cm.RdBu_r]*ndata

    clevs_abs = get_clevs(np.array(dlist[0]), centered=False)
    clevs_rel = get_clevs(np.array(dlist[1]), centered=True)
    clevs = [clevs_abs]*1 + [clevs_rel]*ndata

    lts = grid_coords['target grid'][var]['lat'][tgname]
    lns = grid_coords['target grid'][var]['lon'][tgname]

    # Loop over data sets
    for p, data_name in zip(range(ndata + 1), data_names):
        if thr != 'None':
            headtitle = '{} | {} [{}] | Threshold: {} | {}'.format(
                ftitles[p], var, dc_units, thr, tsuffix_fname)
            fn = '{}_thr{}{}{}_map_diurnal_cycle_{}_{}_{}.png'.format(
                var, thr, tres, tstat, fn_prfx, data_name, tsuffix_fname)
        else:
            headtitle = '{} | {} [{}] | {}'.format(
                ftitles[p], var, dc_units, tsuffix_fname)
            fn = '{}{}{}_map_diurnal_cycle_{}_{}_{}.png'.format(
                var, tres, tstat, fn_prfx, data_name, tsuffix_fname)

        rpl.figure_init(plottype='map')

        # Create map object and axes grid
        map_proj = rpl.define_map_object(map_projection, **map_config)
        fig, axs_grid = rpl.map_setup(
            map_proj, map_extent, figsize, figshape, grid_lines=map_gridlines,
            **map_axes_conf)

        # Plot the maps
        mp = rpl.make_map_plot(
            dlist[p], axs_grid, lts, lns, cmap=cmap[p], clevs=clevs[p],
            **map_plot_conf)

        if var == 'pr':
            rpl.image_colorbar(mp, axs_grid, labelspacing=2,
                               formatter='{:.2f}')
        else:
            rpl.image_colorbar(mp, axs_grid, labelspacing=2,
                               formatter='{:.1f}')

        # Map settings
        rpl.map_axes_settings(fig, axs_grid, headtitle=headtitle,
                              time_mean='hour', time_units=hours)

        plt.savefig(os.path.join(img_dir, fn), bbox_inches='tight')

    # Line plot diurnal cycle
    if fm_listr is not None:
        line_diurnal_cycle(fm_listr, fo_listr, models, nmod, ref_model, obs,
                           var, tres, tstat, dc_units, time_suffix_dd, regions,
                           img_dir, line_grid, line_sets)


def line_diurnal_cycle(fm_list, fo_list, models, nmod, ref_model, obs, var,
                       tres, tstat, units, time_suffix_dd, regions, img_dir,
                       line_grid, line_sets):
    """
    Plotting annual cycle line plot
    """

    ref_mnme = ref_model.upper()
    othr_mod = models.copy()
    othr_mod.remove(ref_model)

    for reg in regions:

        fmod = {m: xa.open_dataset(f) for m, f in zip(models, fm_list[reg])}
        mod_ann = {m: np.nanmean(fmod[m][var].values, axis=(1, 2))
                   for m in models}
        hours = fmod[ref_model].hour.values

        if obs is not None:
            obslist = [obs] if not isinstance(obs, list) else obs
            ref_obs = obslist[0]
            obslbl = "_".join(s for s in obslist)
            fobs = {o: xa.open_dataset(f) for o, f in zip(obslist,
                                                          fo_list[reg])}
            obs_ann = {o: np.nanmean(fobs[o][var].values, axis=(1, 2))
                       for o in obslist}
            dlist = [[obs_ann[ref_obs]] + [mod_ann[m] for m in models],
                     [mod_ann[m] - obs_ann[ref_obs] for m in models]]

            if len(obslist) > 1:
                dlist[0] += [obs_ann[o] for o in obslist[1:]]
                dlist[1] += [obs_ann[o] - obs_ann[ref_obs]
                             for o in obslist[1:]]
                ll_nms = models + obslist[1:]
            else:
                ll_nms = models
            lg_lbls = [[ref_obs] + [m.upper() for m in ll_nms],
                       ['{} - {}'.format(m.upper(), ref_obs) for m in ll_nms]]
        else:
            dlist = [[mod_ann[m] for m in models],
                     [mod_ann[m] - mod_ann[ref_model] for m in othr_mod]]
            lg_lbls = [[m.upper() for m in models], ['{} - {}'.format(
                m.upper(), ref_mnme) for m in othr_mod]]

        xdata = [[hours]*len(dlist[0]), [hours]*len(dlist[1])]

        dcycle_stat = fmod[ref_model].attrs['Description'].\
            split('|')[1].strip()
        if dcycle_stat == 'Amount':
            fn_prfx = 'amnt'
        elif dcycle_stat == 'Frequency':
            fn_prfx = 'freq'
        else:
            print("\n\tUnknown diurnal cycle statistic, exiting...")
            sys.exit()

        tsuffix_ll = [val for key, val in time_suffix_dd.items()]
        tsuffix_fname = "_vs_".join(set(tsuffix_ll))
        thr = fmod[ref_model].attrs['Description'].\
            split('|')[3].split(':')[1].strip()
        # season = fmod[ref_model].attrs['Analysed time'].split(' ')[1]
        # seas = season.replace(' ', '')
        regnm = reg.replace(' ', '_')

        if thr != 'None':
            headtitle = '{} ({}) |  Threshold: {}\n{} | {}'.format(
                var, dcycle_stat, thr, reg, tsuffix_fname)
            if obs is not None:
                fn = ("{}_thr{}{}{}_lnplot_diurnal_cycle_{}_{}_model_"
                      "{}_{}.png").format(
                          var, thr, tres, tstat, fn_prfx, regnm, obslbl,
                          tsuffix_fname)
            else:
                fn = ("{}_thr{}{}{}_lnplot_diurnal_cycle_{}_{}_model_"
                      "{}.png").format(var, thr, tres, tstat, fn_prfx,
                                       regnm, tsuffix_fname)
        else:
            headtitle = '{} ({}) |  {} | {}'.format(var, dcycle_stat,
                                                    reg, tsuffix_fname)
            if obs is not None:
                fn = '{}{}{}_lnplot_diurnal_cycle_{}_{}_model_{}_{}.png'.\
                    format(var, tres, tstat, fn_prfx, regnm, obslbl,
                           tsuffix_fname)
            else:
                fn = '{}{}{}_lnplot_diurnal_cycle_{}_{}_model_{}.png'.\
                    format(var, tres, tstat, fn_prfx, regnm, tsuffix_fname)

        # figure settings
        figsize = (14, 12)
        figshape = (2, 1)

        ylabel = ['({})'.format(units), 'Difference ({})'.format(units)]
        xlabel = [None, 'Hour (UTC)']
        xlim = [[-.5, hours[-1]+.5]]*2
        xticks = hours[:]
        xtlbls = ['{:02d}'.format(h) for h in hours]

        rpl.figure_init(plottype='line')
        fig, lgrid = rpl.fig_grid_setup(fshape=figshape, figsize=figsize,
                                        **line_grid)

        # Scatter
        axs = rpl.make_scatter_plot(lgrid, xdata, dlist, s=100,
                                    edgecolors='k', lw=1.5, alpha=1.)
        [pts.set_facecolor(c)
         for pts, c in zip(axs[0].collections, abs_colors)]
        [pts.set_facecolor(c)
         for pts, c in zip(axs[1].collections, rel_colors)]
        # Legend
        legend_elements = [Line2D([0], [0], marker='o', mec='k', ms=10, lw=0,
                                  color=c, label=l)
                           for c, l, in zip(abs_colors, lg_lbls[0])]
        axs[0].legend(handles=legend_elements, ncol=1, fontsize='large')
        legend_elements = [Line2D([0], [0], marker='o', mec='k', ms=10, lw=0,
                                  color=c, label=l)
                           for c, l, in zip(rel_colors, lg_lbls[1])]
        axs[1].legend(handles=legend_elements, ncol=1, fontsize='large')

        [rpl.axes_settings(ax, xlabel=xlabel[a], xticks=xticks,
                           ylabel=ylabel[a], xtlabels=xtlbls, xlim=xlim[a])
         for a, ax in enumerate(axs)]

        [ax.ticklabel_format(useOffset=False, axis='y') for ax in axs]

        ttl = fig.suptitle(headtitle, fontsize='xx-large')
        ttl.set_position((.5, 1.06))

        plt.savefig(os.path.join(img_dir, fn), bbox_inches='tight')


def moments_plot(fm_list, fo_list, fm_listr, fo_listr, models, nmod,
                 ref_model, obs, var, tres, tstat, units, time_suffix_dd,
                 regions, img_dir, grid_coords, moments_plot_conf, map_domain,
                 map_projection, map_config, map_extent, map_gridlines,
                 map_axes_conf, map_plot_conf, line_grid, line_sets):
    """
    Plotting higher-order moment statistics

    Multiple plot types are available; timeseries, box plot and scatter plot.
    The type shall be specified in the main RCAT configuration file (under the
    'plotting' section). Default is timeseries.
    """

    # type_of_plot = moments_plot_conf['plot type']

    ref_mnme = ref_model.upper()
    othr_mod = models.copy()
    othr_mod.remove(ref_model)

    for reg in regions:

        fmod = {m: xa.open_dataset(f) for m, f in zip(models, fm_listr[reg])}
        mod_ann = {m: np.nanmean(fmod[m][var].values, axis=(1, 2))
                   for m in models}

        err_len_msg = ("\n\n\t*** Input data arrays for timeseries plot do not"
                       " all have the same lengths. This is required for these"
                       " plots. ***\n\n")
        ts_len_md = [arr.size for m, arr in mod_ann.items()]
        assert len(set(ts_len_md)) == 1, err_len_msg

        if obs is not None:
            obslist = [obs] if not isinstance(obs, list) else obs
            ref_obs = obslist[0]
            obslbl = "_".join(s for s in obslist)
            fobs = {o: xa.open_dataset(f) for o, f in zip(obslist,
                                                          fo_listr[reg])}
            obs_ann = {o: np.nanmean(fobs[o][var].values, axis=(1, 2))
                       for o in obslist}

            ts_len_all = ts_len_md + [arr.size for o, arr in obs_ann.items()]
            assert len(set(ts_len_all)) == 1, err_len_msg

            dlist = [[obs_ann[ref_obs]] + [mod_ann[m] for m in models],
                     [mod_ann[m] - obs_ann[ref_obs] for m in models]]

            if len(obslist) > 1:
                dlist[0] += [obs_ann[o] for o in obslist[1:]]
                dlist[1] += [obs_ann[o] - obs_ann[ref_obs]
                             for o in obslist[1:]]
                ll_nms = models + obslist[1:]
            else:
                ll_nms = models
            lg_lbls = [[ref_obs] + [m.upper() for m in ll_nms],
                       ['{} - {}'.format(m.upper(), ref_obs) for m in ll_nms]]
        else:
            dlist = [[mod_ann[m] for m in models],
                     [mod_ann[m] - mod_ann[ref_model] for m in othr_mod]]
            lg_lbls = [[m.upper() for m in models], ['{} - {}'.format(
                m.upper(), ref_mnme) for m in othr_mod]]

        tsuffix_ll = [val for key, val in time_suffix_dd.items()]
        tsuffix_fname = "_vs_".join(set(tsuffix_ll))
        thr = fmod[ref_model].attrs['Description'].\
            split('|')[1].split(':')[1].strip()
        # season = fmod[ref_model].attrs['Analysed time'].split('|')[1]
        # seas = season.replace(' ', '')
        regnm = reg.replace(' ', '_')

        if thr != 'None':
            headtitle = '{} |  Threshold: {}\n{} | {}'.format(
                var, thr, reg, tsuffix_fname)
            if obs is not None:
                fn = 'timeseries_{}_thr{}{}_moment_{}_model_{}_{}.png'.\
                        format(var, thr, tres, regnm, obslbl, tsuffix_fname)
            else:
                fn = 'timeseries_{}_thr{}{}_moment_{}_model_{}.png'.format(
                    var, thr, tres, regnm, tsuffix_fname)
        else:
            headtitle = '{} |  {} | {}'.format(var, reg, tsuffix_fname)
            if obs is not None:
                fn = 'timeseries_{}{}_moment_{}_model_{}_{}.png'.format(
                    var, tres, regnm, obslbl, tsuffix_fname)
            else:
                fn = 'timeseries_{}{}_moment_{}_model_{}.png'.format(
                    var, tres, regnm, tsuffix_fname)

        # figure settings
        figsize = (16, 10)
        figshape = (2, 1)

        ylabel = [f'{units}', 'Difference']
        ylim = [None]*2
        xlabel = ['']*2
        xlim = [None]*2
        xticks = None
        xtlbls = None

        rpl.figure_init(plottype='line')
        fig, lgrid = rpl.fig_grid_setup(fshape=figshape, figsize=figsize,
                                        **line_grid)

        axs = rpl.make_line_plot(lgrid, ydata=dlist, **line_sets)
        [ln.set_color(lc) for ln, lc in zip(axs[0].get_lines(), abs_colors)]
        [ln.set_color(lc) for ln, lc in zip(list(axs[1].get_lines())[:-1],
                                            rel_colors)]

        # Trendlines
        if moments_plot_conf['trendline']:
            for ydata, lc in zip(dlist[0], abs_colors):
                z = np.polyfit(np.arange(len(ydata)), ydata, 1)
                p = np.poly1d(z)
                rpl.make_line_plot([lgrid[0]], ydata=p(np.arange(len(ydata))),
                                   color='k', lw=2, alpha=.6)
                rpl.make_line_plot([lgrid[0]], ydata=p(np.arange(len(ydata))),
                                   lw=0, marker='o', markersize=4.5, mec=lc,
                                   mfc=lc, markevery=2, alpha=1)
        # Running mean
        if moments_plot_conf['running mean']:
            window = moments_plot_conf['running mean']
            for ydata, lc in zip(dlist[0], abs_colors):
                rmn = run_mean(ydata, window, 'same')
                rpl.make_line_plot(
                    [lgrid[0]], ydata=rmn, color='k', lw=2, alpha=.6)
                rpl.make_line_plot([lgrid[0]], ydata=rmn, lw=0, marker='o',
                                   markersize=4, mec=lc, mfc=lc, alpha=1)

        # Legend
        legend_elements = [Line2D([0], [0], lw=3, color=c, label=l)
                           for c, l in zip(abs_colors, lg_lbls[0])]
        if moments_plot_conf['trendline']:
            legend_elements = legend_elements + [
                Line2D([0], [0], lw=3, color='k', marker='o', mfc=c, mec=c,
                       markersize=8, alpha=.6, label='lin. trend')
                for c, _ in zip(abs_colors, lg_lbls[0])]
        if moments_plot_conf['running mean']:
            legend_elements = legend_elements + [
                Line2D([0], [0], lw=3, color='k', marker='o', mfc=c, mec=c,
                       markersize=8, alpha=.6,
                       label=f'run. avg (window: {window})')
                for c, _ in zip(abs_colors, lg_lbls[0])]

        axs[0].legend(handles=legend_elements, ncol=2, fontsize='large')
        legend_elements = [Line2D([0], [0], lw=3, color=c, label=l)
                           for c, l in zip(rel_colors, lg_lbls[1])]
        axs[1].legend(handles=legend_elements, fontsize='large')

        [rpl.axes_settings(ax, xlabel=xlabel[a], xticks=xticks,
                           ylabel=ylabel[a], xtlabels=xtlbls,
                           xlim=xlim[a], ylim=ylim[a])
         for a, ax in enumerate(axs)]

        ttl = fig.suptitle(headtitle, fontsize='xx-large')
        ttl.set_position((.5, 1.03))

        plt.savefig(os.path.join(img_dir, fn), bbox_inches='tight')


def pdf_plot(fm_list, fo_list, fm_listr, fo_listr, models, nmod, ref_model,
             obs, var, tres, tstat, units, time_suffix_dd, regions, img_dir,
             grid_coords, moments_plot_conf, map_domain, map_projection,
             map_config, map_extent, map_gridlines, map_axes_conf,
             map_plot_conf, line_grid, line_sets):
    """
    Plotting frequency-intensity-distribution plot
    """

    ref_mnme = ref_model.upper()
    othr_mod = models.copy()
    othr_mod.remove(ref_model)

    for reg in regions:

        fmod = {m: xa.open_dataset(f) for m, f in zip(models, fm_listr[reg])}
        mod_ann = {m: np.nanmean(fmod[m][var].values*100, axis=(1, 2))
                   for m in models}
        bins = fmod[ref_model].bin_edges.values[1:]
        nbins = bins.size

        if obs is not None:
            obslist = [obs] if not isinstance(obs, list) else obs
            ref_obs = obslist[0]
            obslbl = "_".join(s for s in obslist)
            fobs = {o: xa.open_dataset(f) for o, f in zip(obslist,
                                                          fo_listr[reg])}
            obs_ann = {o: np.nanmean(fobs[o][var].values*100, axis=(1, 2))
                       for o in obslist}
            dlist = [[obs_ann[ref_obs]] + [mod_ann[m] for m in models],
                     [mod_ann[m] - obs_ann[ref_obs] for m in models]]

            if len(obslist) > 1:
                dlist[0] += [obs_ann[o] for o in obslist[1:]]
                dlist[1] += [obs_ann[o] - obs_ann[ref_obs]
                             for o in obslist[1:]]
                ll_nms = models + obslist[1:]
            else:
                ll_nms = models
            lg_lbls = [[ref_obs] + [m.upper() for m in ll_nms],
                       ['{} - {}'.format(m.upper(), ref_obs) for m in ll_nms]]
        else:
            dlist = [[mod_ann[m] for m in models],
                     [mod_ann[m] - mod_ann[ref_model] for m in othr_mod]]
            lg_lbls = [[m.upper() for m in models], ['{} - {}'.format(
                m.upper(), ref_mnme) for m in othr_mod]]

        tsuffix_ll = [val for key, val in time_suffix_dd.items()]
        tsuffix_fname = "_vs_".join(set(tsuffix_ll))
        thr = fmod[ref_model].attrs['Description'].\
            split('|')[1].split(':')[1].strip()
        # season = fmod[ref_model].attrs['Analysed time'].split(' ')[1]
        # seas = season.replace(' ', '')
        regnm = reg.replace(' ', '_')

        if thr != 'None':
            headtitle = '{} |  Threshold: {}\n{} | {}'.format(
                var, thr, reg, tsuffix_fname)
            if obs is not None:
                fn = '{}_thr{}{}_pdf_{}_model_{}_{}.png'.format(
                    var, thr, tres, regnm, obslbl, tsuffix_fname)
            else:
                fn = '{}_thr{}{}_pdf_{}_model_{}.png'.format(
                    var, thr, tres, regnm, tsuffix_fname)
        else:
            headtitle = '{} |  {} | {}'.format(var, reg, tsuffix_fname)
            if obs is not None:
                fn = '{}{}_pdf_{}_model_{}_{}.png'.format(
                    var, tres, regnm, obslbl, tsuffix_fname)
            else:
                fn = '{}{}_pdf_{}_model_{}.png'.format(
                    var, tres, regnm, tsuffix_fname)

        # figure settings
        figsize = (23, 7)
        figshape = (1, 2)

        ylabel = ['Frequency (%)',
                  'Difference']
        ylim = [None]*2
        xlabel = ['({})'.format(units)]*2
        xlim = [[-.5, nbins-.5]]*2
        xticks = range(nbins)[::6]
        xtlbls = bins[::6]

        rpl.figure_init(plottype='line')
        fig, lgrid = rpl.fig_grid_setup(fshape=figshape, figsize=figsize,
                                        **line_grid)

        axs = rpl.make_line_plot(lgrid, ydata=dlist, **line_sets)
        if var == 'pr':
            axs[0].set_yscale('log')

        [ln.set_color(lc) for ln, lc in zip(axs[0].get_lines(), abs_colors)]
        [ln.set_color(lc) for ln, lc in zip(list(axs[1].get_lines())[:-1],
                                            rel_colors)]
        # Legend
        legend_elements = [Line2D([0], [0], lw=2, color=c, label=l)
                           for c, l in zip(abs_colors, lg_lbls[0])]
        axs[0].legend(handles=legend_elements, fontsize='large')
        legend_elements = [Line2D([0], [0], lw=2, color=c, label=l)
                           for c, l in zip(rel_colors, lg_lbls[1])]
        axs[1].legend(handles=legend_elements, fontsize='large')

        [rpl.axes_settings(ax, xlabel=xlabel[a], xticks=xticks,
                           ylabel=ylabel[a], xtlabels=xtlbls,
                           xlim=xlim[a], ylim=ylim[a])
         for a, ax in enumerate(axs)]

        ttl = fig.suptitle(headtitle, fontsize='xx-large')
        ttl.set_position((.5, 1.06))

        plt.savefig(os.path.join(img_dir, fn), bbox_inches='tight')


def map_asop(fm_list, fo_list, fm_listr, fo_listr, models, nmod, ref_model,
             obs, var, tres, tstat, units, time_suffix_dd, regions, img_dir,
             grid_coords, moments_plot_conf, map_domain, map_projection,
             map_config, map_extent, map_gridlines, map_axes_conf,
             map_plot_conf, line_grid, line_sets):
    """
    Plotting ASoP FC factor map plot
    """

    ref_mnme = ref_model.upper()
    othr_mod = models.copy()
    othr_mod.remove(ref_model)

    # Obs data list
    obslist = [obs] if not isinstance(obs, list) else obs

    # Map settings
    target_grid_names = list(grid_coords['target grid'][var]['lon'].keys())
    tgname = target_grid_names[0]

    # Data
    fmod = {m: xa.open_dataset(f) for m, f in zip(models, fm_list)}

    if obslist[0] is not None:
        obslbl = "_".join(s for s in obslist)
        ref_obs = obslist[0]
        fobs = {o: xa.open_dataset(f) for o, f in zip(obslist, fo_list)}

        FCI = {m: np.nansum(
            np.fabs(fmod[m][var].values[1, :] -
                    fobs[ref_obs][var].values[1, :]), axis=0)
               for m in models}
        dlist = [FCI[m] for m in models]
        if len(obslist) > 1:
            dlist += [np.nansum(
                np.fabs(fobs[o][var].values[1, :] -
                        fobs[ref_obs][var].values[1, :]), axis=0)
                      for o in obslist[1:]]
            ndata = nmod + len(obslist[1:])
            ftitles = [(f"{m} {time_suffix_dd[m]} -\n "
                        f"{ref_obs} {time_suffix_dd[ref_obs]}")
                       for m in models + obslist[1:]]
        else:
            ndata = nmod
            ftitles = [(f"{m} {time_suffix_dd[m]} -\n "
                        f"{ref_obs} {time_suffix_dd[ref_obs]}")
                       for m in models]
    else:
        FCI = {m: np.nansum(
            np.fabs(fmod[m][var].values[1, :] -
                    fmod[ref_model][var].values[1, :]), axis=0)
               for m in othr_mod}
        dlist = [FCI[m] for m in othr_mod]
        ndata = nmod-1
        ftitles = [(f"{m} {time_suffix_dd[m]} -\n "
                    f"{ref_mnme} {time_suffix_dd[ref_model]}")
                   for m in othr_mod]

    tsuffix_ll = [val for key, val in time_suffix_dd.items()]
    tsuffix_fname = "_vs_".join(set(tsuffix_ll))
    thr = fmod[ref_model].attrs['Description'].\
        split('|')[1].split(':')[1].strip()
    # season = fmod[ref_model].attrs['Analysed time'].split(' ')[1]
    # seas = season.replace(' ', '')

    # figure settings
    figsize = (20, 8)
    figshape = (1, ndata)

    if thr != 'None':
        headtitle = 'ASoP FC Index | Thr: {} | {}'.format(
            thr, tsuffix_fname)
        if obs is not None:
            fn = 'asop_FC_thr{}{}_map_model_{}_{}.png'.\
                format(thr, tres, obslbl, tsuffix_fname)
        else:
            fn = 'asop_FC_thr{}{}_map_model_{}.png'.format(
                thr, tres, tsuffix_fname)
    else:
        headtitle = 'ASoP FC Index | {}'.format(tsuffix_fname)
        if obs is not None:
            fn = 'asop_FC{}_map_model_{}_{}.png'.\
                format(tres, obslbl, tsuffix_fname)
        else:
            fn = 'asop_FC{}_map_model_{}.png'.format(tres, tsuffix_fname)

    rpl.figure_init(plottype='map')

    # Create map object and axes grid
    map_proj = rpl.define_map_object(map_projection, **map_config)
    fig, axs_grid = rpl.map_setup(
        map_proj, map_extent, figsize, figshape, grid_lines=map_gridlines,
        **map_axes_conf)

    lts = grid_coords['target grid'][var]['lat'][tgname]
    lns = grid_coords['target grid'][var]['lon'][tgname]

    cmap = [mpl.cm.PuRd]*ndata
    clevs = [np.linspace(0, 2, 21)]*ndata

    # Plot the maps
    mp = rpl.make_map_plot(
        dlist, axs_grid, lts, lns, cmap=cmap, clevs=clevs, **map_plot_conf)
    rpl.image_colorbar(mp, axs_grid, labelspacing=2, formatter='{:.1f}')

    # Map settings
    rpl.map_axes_settings(fig, axs_grid, headtitle=headtitle)

    [ax.text(0.5, 1.05, ft.upper(), size=21, va='center', ha='center',
             transform=ax.transAxes) for ft, ax in zip(ftitles, axs_grid)]

    plt.savefig(os.path.join(img_dir, fn), bbox_inches='tight')

    # Line plot asop
    if fm_listr is not None:
        line_asop(fm_listr, fo_listr, models, nmod, ref_model, obs, var, tres,
                  tstat, units, time_suffix_dd, regions, img_dir, line_grid,
                  line_sets)


def line_asop(fm_listr, fo_listr, models, nmod, ref_model, obs, var, tres,
              tstat, units, time_suffix_dd, regions, img_dir, line_grid,
              line_sets):
    """
    Plotting ASoP line plot of C and FC factors
    """

    ref_mnme = ref_model.upper()
    othr_mod = models.copy()
    othr_mod.remove(ref_model)

    for reg in regions:

        fmod = {m: xa.open_dataset(f) for m, f in zip(models, fm_listr[reg])}
        mod_ann = {m: np.nanmean(fmod[m][var].values, axis=(2, 3))
                   for m in models}

        bins = fmod[ref_model].bin_edges.values
        factors = fmod[ref_model].factors.values

        if obs is not None:
            obslist = [obs] if not isinstance(obs, list) else obs
            ref_obs = obslist[0]
            obslbl = "_".join(s for s in obslist)
            fobs = {o: xa.open_dataset(f) for o, f in zip(obslist,
                                                          fo_listr[reg])}
            obs_ann = {o: np.nanmean(fobs[o][var].values, axis=(2, 3))
                       for o in obslist}
            dlist = [[obs_ann[ref_obs]] + [mod_ann[m] for m in models],
                     [mod_ann[m] - obs_ann[ref_obs] for m in models]]

            if len(obslist) > 1:
                dlist[0] += [obs_ann[o] for o in obslist[1:]]
                dlist[1] += [obs_ann[o] - obs_ann[ref_obs]
                             for o in obslist[1:]]
                ll_nms = models + obslist[1:]
            else:
                ll_nms = models
            lg_lbls = [[ref_obs] + [m.upper() for m in ll_nms],
                       ['{} - {}'.format(m.upper(), ref_obs) for m in ll_nms]]
        else:
            dlist = [[mod_ann[m] for m in models],
                     [mod_ann[m] - mod_ann[ref_model] for m in othr_mod]]
            lg_lbls = [[m.upper() for m in models], ['{} - {}'.format(
                m.upper(), ref_mnme) for m in othr_mod]]

        ylabels = [units, '%']
        tsuffix_ll = [val for key, val in time_suffix_dd.items()]
        tsuffix_fname = "_vs_".join(set(tsuffix_ll))
        for ff, fctr in enumerate(factors):
            sc = 100 if fctr == 'FC' else 1
            dlist_ff = [[arr[ff, :]*sc for arr in dlist[0]],
                        [arr[ff, :]*sc for arr in dlist[1]]]
            xdata = [[bins[:-1]]*len(dlist[0]), [bins[:-1]]*len(dlist[1])]

            thr = fmod[ref_model].attrs['Description'].\
                split('|')[1].split(':')[1].strip()
            # season = fmod[ref_model].attrs['Analysed time'].split(' ')[1]
            # seas = season.replace(' ', '')
            regnm = reg.replace(' ', '_')

            if thr != 'None':
                headtitle = 'ASoP ({}) |  Thr: {}\n{} | {}'.format(
                    fctr, thr, reg, tsuffix_fname)
                if obs is not None:
                    fn = '{}_thr{}{}_asop_{}_{}_model_{}_{}.png'.format(
                        var, thr, tres, fctr, regnm, obslbl, tsuffix_fname)
                else:
                    fn = '{}_thr{}{}_asop_{}_{}_model_{}.png'.format(
                        var, thr, tres, fctr, regnm, tsuffix_fname)
            else:
                headtitle = 'ASoP ({}) |  {} | {}'.format(
                    fctr, reg, tsuffix_fname)
                if obs is not None:
                    fn = '{}{}_asop_{}_{}_model_{}_{}.png'.format(
                        var, tres, fctr, regnm, obslbl, tsuffix_fname)
                else:
                    fn = '{}{}_asop_{}_{}_model_{}.png'.format(
                        var, tres, fctr, regnm, tsuffix_fname)

            # figure settings
            figsize = (13, 10)
            figshape = (2, 1)

            ylabel = [ylabels[ff]]*2
            ylim = [None]*2
            xlabel = [None, 'Intensity ({})'.format(units)]
            xlim = [[1e-2, 1e2]]*2

            rpl.figure_init(plottype='line')
            ln_grid = deepcopy(line_grid)
            if 'axes_pad' in ln_grid:
                ln_grid.pop('axes_pad')
            if 'sharex' not in ln_grid:
                ln_grid.update({'sharex': True})
            fig, lgrid = rpl.fig_grid_setup(fshape=figshape, figsize=figsize,
                                            **ln_grid)

            axs = rpl.make_line_plot(lgrid, xdata=xdata, ydata=dlist_ff,
                                     axis_type='logx', **line_sets)

            [ln.set_color(lc) for ln, lc in zip(axs[0].get_lines(),
                                                abs_colors)]
            [ln.set_color(lc) for ln, lc in zip(list(axs[1].get_lines())[:-1],
                                                rel_colors)]
            # Legend
            legend_elements = [Line2D([0], [0], lw=2, color=c, label=l)
                               for c, l in zip(abs_colors, lg_lbls[0])]
            axs[0].legend(handles=legend_elements, fontsize='large')
            legend_elements = [Line2D([0], [0], lw=2, color=c, label=l)
                               for c, l in zip(rel_colors, lg_lbls[1])]
            axs[1].legend(handles=legend_elements, fontsize='large')

            [rpl.axes_settings(ax, xlabel=xlabel[a],  # xticks=xticks,
                               ylabel=ylabel[a],  # xtlabels=xtlbls,
                               xlim=xlim[a], ylim=ylim[a])
             for a, ax in enumerate(axs)]

            ttl = fig.suptitle(headtitle, fontsize='xx-large')
            ttl.set_position((.5, 1.06))

            plt.savefig(os.path.join(img_dir, fn), bbox_inches='tight')
