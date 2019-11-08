"""
Module script for plotting
"""
import os
import xarray as xa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
import plots as rpl
import geosfuncs as gfunc
import colors
from copy import deepcopy

# Colors
set1 = colors.getcolormap('set1', custom=True)
set1_m = [[int(255*x) for x in triplet] for triplet in set1]
rel_colors = ['#{:02x}{:02x}{:02x}'.format(s[0], s[1], s[2]) for s in set1_m]
abs_colors = rel_colors[:]
abs_colors.insert(0, 'k')


def plot_main(pltdict, statistic):
    """
    Main function that controls the plotting procedure.
    """
    pdict = deepcopy(pltdict)
    models = pdict['models']
    nmod = len(models)
    obs = pdict['observation']
    var = pdict['variable']
    tres = pdict['time res']
    tstat = pdict['time stat']
    units = pdict['units']
    years = pdict['years']
    regions = pdict['regions']
    img_dir = pdict['img dir']

    grid_coords = pdict['grid coords']
    map_conf = pdict['map configure']
    map_grid = pdict['map grid setup']
    map_sets = pdict['map kwargs']
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

    if isinstance(years, dict):
        ytitle = "{}-{} vs {}-{}".format(years['T1'][0], years['T1'][1],
                                         years['T2'][0], years['T2'][1])
    else:
        ytitle = "{}-{}".format(years[0], years[1])

    _plots(statistic)(fm_list, fo_list, fm_listr, fo_listr, models, nmod, obs,
                      var, tres, tstat, units, ytitle, regions, img_dir,
                      grid_coords, map_conf, map_grid, map_sets, line_grid,
                      line_sets)


def _plots(stat):
    p = {
        'seasonal cycle': map_season,
        'annual cycle': map_ann_cycle,
        'percentile': map_pctls,
        'diurnal cycle': map_diurnal_cycle,
        'pdf': pdf_plot,
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

    nsteps = 15
    diff = round_max - round_min
    if diff < 1:
        nz = np.floor(np.abs(np.log10(np.abs(diff))))
        step = _round_down(diff/nsteps, nz+2)
    elif 1 <= diff < 10:
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


def map_season(fm_list, fo_list, fm_listr, fo_listr, models, nmod, obs, var,
               tres, tstat, units, ytitle, regions, img_dir, grid_coords,
               map_conf, map_grid, map_sets, line_grid, line_sets):
    """
    Plotting seasonal mean map plot
    """

    # Reference model (first in the list in the configuration file (under
    # MODELS)), and the other model(s)
    ref_model = models[0]
    ref_mnme = ref_model.upper()
    othr_mod = models.copy()
    othr_mod.remove(ref_model)

    # Obs data list
    obslist = [obs] if not isinstance(obs, list) else obs

    # Map configurations
    if (map_conf['zoom'] == 'crnrs' and
            'crnr_vals' not in map_conf):
        map_conf.update({
            'crnr_vals': grid_coords['meta data'][var][ref_model]['crnrs']
        })
    if 'lon_0' not in map_conf:
        map_conf.update({
            'lon_0': grid_coords['meta data'][var][ref_model]['lon_0']
        })
    if map_conf['proj'] == 'lcc' and 'lat_0' not in map_conf:
        map_conf.update({
            'lat_0': grid_coords['meta data'][var][ref_model]['lat_0']
        })
    target_grid_names = list(grid_coords['target grid'][var]['lon'].keys())
    # Native grids are not possible for this plot, unless model(s)/obs have
    # same native grid configurations.
    tgname = target_grid_names[0]
    domain = grid_coords['meta data'][var][ref_model]['domain']
    mask = gfunc.reg_mask(
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
            ftitles = [ref_obs] + ["{} - {}".format(m, ref_obs)
                                   for m in models + obslist[1:]]
        else:
            ndata = nmod
            ftitles = [ref_obs] + ["{} - {}".format(m, ref_obs)
                                   for m in models]
    else:
        dlist = [fmod_msk[ref_model][var].values[i, :] for i in range(4)] +\
                [fmod_msk[m][var].values[i, :] -
                 fmod_msk[ref_model][var].values[i, :]
                 for m in othr_mod for i in range(4)]
        ndata = nmod-1
        ftitles = [ref_mnme] + ["{} - {}".format(m, ref_mnme)
                                for m in othr_mod]

    # figure settings
    figsize = (22, 14)
    figshape = (ndata + 1, 4)
    thr = fmod_msk[ref_model].attrs['Description'].split('|')[2].split(':')[1]
    thr = thr.replace(' ', '')
    thrnm = thr.replace('.', '')
    if thr != 'None':
        headtitle = '{} [{}] | Threshold: {}\n{}'.format(
            var, units, thr, ytitle)
        if obs is not None:
            fn = '{}_thr{}_{}{}_map_seasonal_cycle_model_{}_{}.png'.format(
                var, thrnm, tres, tstat, obslbl, ytitle.replace(' ', '_'))
        else:
            fn = '{}_thr{}_{}{}_map_seasonal_cycle_model_{}.png'.format(
                var, thrnm, tres, tstat, ytitle.replace(' ', '_'))
    else:
        headtitle = '{} [{}] | {}'.format(var, units, ytitle)
        if obs is not None:
            fn = '{}_{}{}_map_seasonal_cycle_model_{}_{}.png'.format(
                var, tres, tstat, obslbl, ytitle.replace(' ', '_'))
        else:
            fn = '{}_{}{}_map_seasonal_cycle_model_{}.png'.format(
                var, tres, tstat, ytitle.replace(' ', '_'))

    if var == 'pr':
        cmap = [colors.getcolormap('YlGnBu', custom=False)]*4 +\
               [colors.getcolormap('prec_diff_wzero', custom=True)]*ndata*4
    else:
        cmap = [colors.getcolormap('Spectral_r', custom=False)]*4 +\
               [colors.getcolormap('RdBu_r', custom=False)]*ndata*4

    clevs_abs = get_clevs(np.array(dlist[0:4]), centered=False)
    clevs_rel = get_clevs(np.array(dlist[4:8]), centered=True)
    clevs = [clevs_abs]*4 + [clevs_rel]*ndata*4

    rpl.figure_init(plottype='map')

    if var == 'psl':
        map_kw = map_grid.copy()
        if (('cbar_mode' in map_kw and
             map_kw['cbar_mode'] is not None)):
            map_kw.update({'cbar_mode': None})
        else:
            map_kw.update({'cbar_mode': None})
        fig, grid = rpl.image_grid_setup(figsize=figsize, fshape=figshape,
                                         **map_kw)
    else:
        fig, grid = rpl.image_grid_setup(figsize=figsize, fshape=figshape,
                                         **map_grid)

    lts = grid_coords['target grid'][var]['lat'][tgname]
    lns = grid_coords['target grid'][var]['lon'][tgname]

    # Create map object
    m, coords = rpl.map_setup(grid, lts, lns, **map_conf)

    mp = rpl.make_map_plot(dlist, grid, m, coords,  cmap=cmap,
                           clevs=clevs, **map_sets)
    rpl.image_colorbar(mp, grid, labelspacing=2, formatter='{:.1f}')

    # Add contour plot if mslp
    if var == 'psl':
        lp = rpl.make_map_plot(dlist, grid, m, coords,  clevs=clevs,
                               filled=False, colors='#4f5254', linewidths=2.3)
        [plt.clabel(mm, fmt='%.1f', colors='k', fontsize=15) for mm in lp]

    # Map settings
    rpl.map_axes_settings(fig, grid, fontsize='large', headtitle=headtitle,
                          time_mean='season')

    # Annotate
    [ax.text(-0.05, 0.5, ft.upper(), size='x-large', va='center', ha='center',
             rotation=90, transform=ax.transAxes)
     for ft, ax in zip(ftitles, [grid[i]
                                 for i in [p*4 for p in range(ndata+1)]])]

    plt.savefig(os.path.join(img_dir, fn), bbox_inches='tight')


def map_ann_cycle(fm_list, fo_list, fm_listr, fo_listr, models, nmod, obs, var,
                  tres, tstat, units, ytitle, regions, img_dir, grid_coords,
                  map_conf, map_grid, map_sets, line_grid, line_sets):
    """
    Plotting annual cycle map plot
    """

    # Reference model (first in the list in the configuration file (under
    # MODELS))
    ref_model = models[0]
    ref_mnme = ref_model.upper()
    othr_mod = models.copy()
    othr_mod.remove(ref_model)

    # Obs data list
    obslist = [obs] if not isinstance(obs, list) else obs

    # Map configurations
    if (map_conf['zoom'] == 'crnrs' and
            'crnr_vals' not in map_conf):
        map_conf.update({
            'crnr_vals': grid_coords['meta data'][var][ref_model]['crnrs']
        })
    if 'lon_0' not in map_conf:
        map_conf.update({
            'lon_0': grid_coords['meta data'][var][ref_model]['lon_0']
        })
    if map_conf['proj'] == 'lcc' and 'lat_0' not in map_conf:
        map_conf.update({
            'lat_0': grid_coords['meta data'][var][ref_model]['lat_0']
        })
    target_grid_names = list(grid_coords['target grid'][var]['lon'].keys())
    # Native grids are not possible for this plot, unless model(s)/obs have
    # same native grid configurations.
    tgname = target_grid_names[0]
    domain = grid_coords['meta data'][var][ref_model]['domain']
    mask = gfunc.reg_mask(
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
            ftitles = [ref_obs] + ['{} - {}'.format(m.upper(), ref_obs)
                                   for m in models + obslist[1:]]
        else:
            ndata = nmod
            ftitles = [ref_obs] + ['{} - {}'.format(m.upper(), ref_obs)
                                   for m in models]
    else:
        dlist = [[fmod_msk[ref_model][var].values[i, :] for i in range(12)]] +\
                [[fmod_msk[m][var].values[i, :] -
                 fmod_msk[ref_model][var].values[i, :] for i in range(12)]
                 for m in othr_mod]
        ndata = nmod-1
        ftitles = [ref_mnme] + ['{} - {}'.format(m.upper(), ref_mnme)
                                for m in othr_mod]

    thr = fmod_msk[ref_model].attrs['Description'].split('|')[2].split(':')[1]
    thr = thr.replace(' ', '')
    thrnm = thr.replace('.', '')

    # figure settings
    figsize = (16, 18)
    figshape = (4, 3)
    if var == 'pr':
        cmap = [colors.getcolormap('YlGnBu', custom=False)] +\
               [colors.getcolormap('prec_diff_wzero', custom=True)]*ndata
    else:
        cmap = [colors.getcolormap('Spectral_r', custom=False)] +\
               [colors.getcolormap('RdBu_r', custom=False)]*ndata

    clevs_abs = get_clevs(np.array(dlist[0]), centered=False)
    clevs_rel = get_clevs(np.array(dlist[1]), centered=True)
    clevs = [clevs_abs]*1 + [clevs_rel]*ndata

    lts = grid_coords['target grid'][var]['lat'][tgname]
    lns = grid_coords['target grid'][var]['lon'][tgname]

    # Loop over data sets
    for p in range(ndata + 1):
        data_name = ftitles[p].replace(' ', '')
        if thr != 'None':
            headtitle = '{} | {} [{}] | Threshold: {}\n{}'.format(
                ftitles[p], var, units, thr, ytitle)
            fn = '{}_thr{}_{}{}_map_ann_cycle_{}_{}.png'.format(
                var, thrnm, tres, tstat, data_name, ytitle.replace(' ', '_'))
        else:
            headtitle = '{} | {} [{}] | {}'.format(
                ftitles[p], var, units, ytitle)
            fn = '{}_{}{}_map_ann_cycle_{}_{}.png'.format(
                var, tres, tstat, data_name, ytitle.replace(' ', '_'))

        rpl.figure_init(plottype='map')

        if var == 'psl':
            map_kw = map_grid.copy()
            if (('cbar_mode' in map_kw and
                 map_kw['cbar_mode'] is not None)):
                map_kw.update({'cbar_mode': None})
            else:
                map_kw.update({'cbar_mode': None})
            fig, grid = rpl.image_grid_setup(figsize=figsize, fshape=figshape,
                                             **map_kw)
        else:
            fig, grid = rpl.image_grid_setup(figsize=figsize, fshape=figshape,
                                             **map_grid)

        # Create map object
        m, coords = rpl.map_setup(grid, lts, lns, **map_conf)

        mp = rpl.make_map_plot(dlist[p], grid, m, coords, cmap=cmap[p],
                               clevs=clevs[p], **map_sets)
        rpl.image_colorbar(mp, grid, labelspacing=2, formatter='{:.1f}')

        # Add contour plot if mslp
        if var == 'psl':
            lp = rpl.make_map_plot(dlist[p], grid, m, coords, clevs=clevs[p],
                                   filled=False, colors='#4f5254',
                                   linewidths=2.3)
            [plt.clabel(mm, fmt='%.1f', colors='k', fontsize=15) for mm in lp]

        # Map settings
        rpl.map_axes_settings(fig, grid, fontsize='large', headtitle=headtitle,
                              time_mean='month')

        plt.savefig(os.path.join(img_dir, fn), bbox_inches='tight')

    # Line plot annual cycle
    if fm_listr is not None:
        line_ann_cycle(fm_listr, fo_listr, models, nmod, obs, var, tres,
                       tstat, units, ytitle, regions, img_dir, line_grid,
                       line_sets)


def line_ann_cycle(fm_list, fo_list, models, nmod, obs, var, tres, tstat,
                   units, ytitle, regions, img_dir, line_grid, line_sets):
    """
    Plotting annual cycle line plot
    """
    # Reference model (first in the list in the configuration file (under
    # MODELS))
    ref_model = models[0]
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

        thr = fmod[ref_model].attrs['Description'].split('|')[2].split(':')[1]
        thr = thr.replace(' ', '')
        thrnm = thr.replace('.', '')
        regnm = reg.replace(' ', '_')

        if thr != 'None':
            headtitle = '{} | Threshold: {}\n{} {}'.format(
                var, thr, reg, ytitle)
            if obs is not None:
                fn = '{}_thr{}_{}{}_lnplot_ann_cycle_{}_model_{}_{}.png'.\
                    format(var, thrnm, tres, tstat, regnm, obslbl,
                           ytitle.replace(' ', '_'))
            else:
                fn = '{}_thr{}_{}{}_lnplot_ann_cycle_{}_model_{}.png'.\
                    format(var, thrnm, tres, tstat, regnm,
                           ytitle.replace(' ', '_'))
        else:
            headtitle = '{} | {} {}'.format(var, reg, ytitle)
            if obs is not None:
                fn = '{}_{}{}_lnplot_ann_cycle_{}_model_{}_{}.png'.\
                    format(var, tres, tstat, regnm, obslbl,
                           ytitle.replace(' ', '_'))
            else:
                fn = '{}_{}{}_lnplot_ann_cycle_{}_model_{}.png'.\
                    format(var, tres, tstat, regnm, ytitle.replace(' ', '_'))

        # figure settings
        figsize = (16, 7)
        figshape = (1, 2)

        ylabel = ['Monthly mean ({})'.format(units),
                  'Bias ({})'.format(units)]
        xlabel = [None]*2
        xlim = [[-.5, 11.5]]*2
        xticks = range(12)
        xtlbls = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
                  'Sep', 'Oct', 'Nov', 'Dec']

        rpl.figure_init(plottype='line')
        fig, lgrid = rpl.fig_grid_setup(fshape=figshape, figsize=figsize,
                                        **line_grid)

        axs = rpl.make_line_plot(lgrid, ydata=dlist, **line_sets)

        [l.set_color(lc) for l, lc in zip(axs[0].get_lines(), abs_colors)]
        [l.set_color(lc) for l, lc in zip(
            list(axs[1].get_lines())[:len(dlist[1])], rel_colors)]
        # Legend
        legend_elements = [Line2D([0], [0], lw=2, color=c, label=l)
                           for c, l in zip(abs_colors, lg_lbls[0])]
        axs[0].legend(handles=legend_elements, fontsize=16)
        legend_elements = [Line2D([0], [0], lw=2, color=c, label=l)
                           for c, l in zip(rel_colors, lg_lbls[1])]
        axs[1].legend(handles=legend_elements, fontsize=16)

        [rpl.axes_settings(ax, xlabel=xlabel[a], xticks=xticks,
                           ylabel=ylabel[a], xtlabels=xtlbls, xlim=xlim[a])
         for a, ax in enumerate(axs)]

        ttl = fig.suptitle(headtitle, fontsize=22)
        ttl.set_position((.5, 1.08))

        plt.savefig(os.path.join(img_dir, fn), bbox_inches='tight')


def map_pctls(fm_list, fo_list, fm_listr, fo_listr, models, nmod, obs, var,
              tres, tstat, units, ytitle, regions, img_dir, grid_coords,
              map_conf, map_grid, map_sets, line_grid, line_sets):
    """
    Plotting percentile map plot
    """

    # Reference model (first in the list in the configuration file (under
    # MODELS))
    ref_model = models[0]
    ref_mnme = ref_model.upper()
    othr_mod = models.copy()
    othr_mod.remove(ref_model)

    # Obs data list
    obslist = [obs] if not isinstance(obs, list) else obs

    # Map configurations
    if (map_conf['zoom'] == 'crnrs' and
            'crnr_vals' not in map_conf):
        map_conf.update({
            'crnr_vals': grid_coords['meta data'][var][ref_model]['crnrs']
        })
    if 'lon_0' not in map_conf:
        map_conf.update({
            'lon_0': grid_coords['meta data'][var][ref_model]['lon_0']
        })
    if map_conf['proj'] == 'lcc' and 'lat_0' not in map_conf:
        map_conf.update({
            'lat_0': grid_coords['meta data'][var][ref_model]['lat_0']
        })
    target_grid_names = list(grid_coords['target grid'][var]['lon'].keys())
    # Native grids are not possible for this plot, unless model(s)/obs have
    # same native grid configurations.
    tgname = target_grid_names[0]
    domain = grid_coords['meta data'][var][ref_model]['domain']
    mask = gfunc.reg_mask(
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
            ftitles = [ref_obs] + ['{} - {}'.format(m.upper(), ref_obs)
                                   for m in models + obslist[1:]]
        else:
            ndata = nmod
            ftitles = [ref_obs] + ['{} - {}'.format(m.upper(), ref_obs)
                                   for m in models]
    else:
        dlist = [[fmod_msk[ref_model][var].values[i, :]] +
                 [fmod_msk[m][var].values[i, :] -
                  fmod_msk[ref_model][var].values[i, :]
                  for m in othr_mod] for i in range(npctl)]
        ndata = nmod-1
        ftitles = [ref_mnme] + ['{} - {}'.format(m.upper(), ref_mnme)
                                for m in othr_mod]
    thr = fmod[ref_model].attrs['Description'].split('|')[2].split(':')[1]
    thr = thr.replace(' ', '')
    thrnm = thr.replace('.', '')
    season = fmod[ref_model].attrs['Analysed time'].split('|')[1]
    seas = season.replace(' ', '')

    # figure settings
    figsize = (20, 10)
    figshape = (1, ndata+1)
    if var == 'pr':
        cmap = [colors.getcolormap('YlGnBu', custom=False)] +\
               [colors.getcolormap('prec_diff_wzero', custom=True)]*ndata
    else:
        cmap = [colors.getcolormap('Spectral_r', custom=False)] +\
               [colors.getcolormap('RdBu_r', custom=False)]*ndata

    lts = grid_coords['target grid'][var]['lat'][tgname]
    lns = grid_coords['target grid'][var]['lon'][tgname]

    # Loop over percentiles
    for p in range(npctl):
        if thr != 'None':
            headtitle =\
                    '{} [{}] | p{} | Threshold: {} | {} {}'.\
                    format(var, units, pctls[p], thr, seas, ytitle)
            if obs is not None:
                fn = '{}_thr{}_{}_map_percentile_p{}_model_{}_{}_{}.png'.\
                    format(var, thrnm, tres, pctls[p], obslbl,
                           ytitle.replace(' ', '_'), seas)
            else:
                fn = '{}_thr{}_{}_map_percentile_p{}_model_{}_{}.png'.\
                    format(var, thrnm, tres, pctls[p],
                           ytitle.replace(' ', '_'), seas)
        else:
            headtitle = '{} [{}] | p{} | {} {}'.format(
                var, units, pctls[p], seas, ytitle)
            if obs is not None:
                fn = '{}_{}_map_percentile_p{}_model_{}_{}_{}.png'.format(
                    var, tres, pctls[p], obslbl, ytitle.replace(' ', '_'),
                    seas)
            else:
                fn = '{}_{}_map_percentile_p{}_model_{}_{}.png'.format(
                    var, tres, pctls[p], ytitle.replace(' ', '_'), seas)

        rpl.figure_init(plottype='map')

        # Setup figure grid with colorbar settings
        fig, grid = rpl.image_grid_setup(figsize=figsize, fshape=figshape,
                                         **map_grid)

        # Create map object
        m, coords = rpl.map_setup(grid, lts, lns, **map_conf)

        clevs_abs = get_clevs(np.array(dlist[p][0]), centered=False)
        clevs_rel = get_clevs(np.array(dlist[p][1]), centered=True)
        clevs = [clevs_abs] + [clevs_rel]*ndata

        mp = rpl.make_map_plot(dlist[p], grid, m, coords,  cmap=cmap,
                               clevs=clevs, **map_sets)
        rpl.image_colorbar(mp, grid, labelspacing=2, formatter='{:.1f}')

        # Map settings
        rpl.map_axes_settings(fig, grid, headtitle=headtitle)

        [ax.text(0.5, 1.04, ft.upper(), size=21, va='center', ha='center',
                 transform=ax.transAxes) for ft, ax in zip(ftitles, grid)]

        plt.savefig(os.path.join(img_dir, fn), bbox_inches='tight')


def map_diurnal_cycle(fm_list, fo_list, fm_listr, fo_listr, models, nmod, obs,
                      var, tres, tstat, units, years, regions, img_dir,
                      grid_coords, map_conf, map_grid, map_sets, line_grid,
                      line_sets):
    """
    Plotting diurnal cycle map plot
    """

    # Reference model (first in the list in the configuration file (under
    # MODELS))
    ref_model = models[0]
    ref_mnme = ref_model.upper()
    othr_mod = models.copy()
    othr_mod.remove(ref_model)

    # Obs data list
    obslist = [obs] if not isinstance(obs, list) else obs

    # Map configurations
    if (map_conf['zoom'] == 'crnrs' and
            'crnr_vals' not in map_conf):
        map_conf.update({
            'crnr_vals': grid_coords['meta data'][var][ref_model]['crnrs']
        })
    if 'lon_0' not in map_conf:
        map_conf.update({
            'lon_0': grid_coords['meta data'][var][ref_model]['lon_0']
        })
    if map_conf['proj'] == 'lcc' and 'lat_0' not in map_conf:
        map_conf.update({
            'lat_0': grid_coords['meta data'][var][ref_model]['lat_0']
        })
    target_grid_names = list(grid_coords['target grid'][var]['lon'].keys())
    # Native grids are not possible for this plot, unless model(s)/obs have
    # same native grid configurations.
    tgname = target_grid_names[0]
    domain = grid_coords['meta data'][var][ref_model]['domain']
    mask = gfunc.reg_mask(
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
            ftitles = [ref_obs] + ['{} - {}'.format(m.upper(), ref_obs)
                                   for m in models + obslist[1:]]
        else:
            ndata = nmod
            ftitles = [ref_obs] + ['{} - {}'.format(m.upper(), ref_obs)
                                   for m in models]
    else:
        dlist = [[fmod_msk[ref_model][var].values[i, :]
                  for i in range(nhour)]] +\
                [[fmod_msk[m][var].values[i, :] -
                  fmod_msk[ref_model][var].values[i, :]
                  for i in range(nhour)] for m in othr_mod]
        ndata = nmod-1
        ftitles = [ref_mnme] + ['{} - {}'.format(m.upper(), ref_mnme)
                                for m in othr_mod]

    thr = fmod[ref_model].attrs['Description'].split('|')[2].split(':')[1]
    thr = thr.replace(' ', '')
    thrnm = thr.replace('.', '')
    season = fmod[ref_model].attrs['Analysed time'].split('|')[1]
    seas = season.replace(' ', '')

    # figure settings
    figsize = (22, 14)
    figshape = fshape
    if var == 'pr':
        cmap = [colors.getcolormap('YlGnBu', custom=False)] +\
               [colors.getcolormap('prec_diff_wzero', custom=True)]*ndata
    else:
        cmap = [colors.getcolormap('Spectral_r', custom=False)] +\
               [colors.getcolormap('RdBu_r', custom=False)]*ndata

    clevs_abs = get_clevs(np.array(dlist[0]), centered=False)
    clevs_rel = get_clevs(np.array(dlist[1]), centered=True)
    clevs = [clevs_abs]*1 + [clevs_rel]*ndata

    lts = grid_coords['target grid'][var]['lat'][tgname]
    lns = grid_coords['target grid'][var]['lon'][tgname]

    # Loop over data sets
    for p in range(ndata + 1):
        data_name = ftitles[p].replace(' ', '')
        if thr != 'None':
            headtitle = '{} | {} [{}] | Threshold: {}\n{}-{} | {}'.format(
                ftitles[p], var, units, thr, years[0], years[1], seas)
            fn = '{}_thr{}_{}{}_map_diurnal_cycle_{}_{}-{}_{}.png'.format(
                var, thrnm, tres, tstat, data_name, years[0], years[1], seas)
        else:
            headtitle = '{} | {} [{}] | {}-{} | {}'.format(
                ftitles[p], var, units, years[0], years[1], seas)
            fn = '{}_{}{}_map_diurnal_cycle_{}_{}-{}_{}.png'.format(
                var, tres, tstat, data_name, years[0], years[1], seas)

        rpl.figure_init(plottype='map')

        # Setup figure grid with colorbar settings
        fig, grid = rpl.image_grid_setup(figsize=figsize, fshape=figshape,
                                         **map_grid)

        # Create map object
        m, coords = rpl.map_setup(grid, lts, lns, **map_conf)

        mp = rpl.make_map_plot(dlist[p], grid, m, coords,  cmap=cmap[p],
                               clevs=clevs[p], **map_sets)
        if var == 'pr':
            rpl.image_colorbar(mp, grid, labelspacing=2, formatter='{:.2f}')
        else:
            rpl.image_colorbar(mp, grid, labelspacing=2, formatter='{:.1f}')

        # Map settings
        rpl.map_axes_settings(fig, grid, headtitle=headtitle,
                              time_mean='hour', time_units=hours)

        plt.savefig(os.path.join(img_dir, fn), bbox_inches='tight')

    # Line plot diurnal cycle
    if fm_listr is not None:
        line_diurnal_cycle(fm_listr, fo_listr, models, nmod, obs, var,
                           tres, tstat, units, years, regions, img_dir,
                           line_grid, line_sets)


def line_diurnal_cycle(fm_list, fo_list, models, nmod, obs, var, tres, tstat,
                       units, years, regions, img_dir, line_grid, line_sets):
    """
    Plotting annual cycle line plot
    """
    # Reference model (first in the list in the configuration file (under
    # MODELS))
    ref_model = models[0]
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

        thr = fmod[ref_model].attrs['Description'].split('|')[3].split(':')[1]
        thr = thr.replace(' ', '')
        thrnm = thr.replace('.', '')
        season = fmod[ref_model].attrs['Analysed time'].split('|')[1]
        seas = season.replace(' ', '')
        regnm = reg.replace(' ', '_')

        if thr != 'None':
            headtitle = '{} |  Threshold: {}\n{} | {}-{} | {}'.format(
                var, thr, reg, years[0], years[1], seas)
            if obs is not None:
                fn = """{}_thr{}_{}{}_lnplot_diurnal_cycle_{}_model_"""
                """{}_{}-{}_{}.png""".format(var, thrnm, tres, tstat, regnm,
                                             obslbl, years[0], years[1], seas)
            else:
                fn = """{}_thr{}_{}{}_lnplot_diurnal_cycle_{}_model_"""
                """{}-{}_{}.png""".format(var, thrnm, tres, tstat, regnm,
                                          years[0], years[1], seas)
        else:
            headtitle = '{} |  {} | {}-{} | {}'.format(var, reg, years[0],
                                                       years[1], seas)
            if obs is not None:
                fn = '{}_{}{}_lnplot_diurnal_cycle_{}_model_{}_{}-{}_{}.png'.\
                    format(var, tres, tstat, regnm, obslbl,
                           years[0], years[1], seas)
            else:
                fn = '{}_{}{}_lnplot_diurnal_cycle_{}_model_{}-{}_{}.png'.\
                    format(var, tres, tstat, regnm, years[0], years[1], seas)

        # figure settings
        figsize = (14, 12)
        figshape = (2, 1)

        ylabel = ['({})'.format(units),
                  'Bias ({})'.format(units)]
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
        axs[0].legend(handles=legend_elements, ncol=1, fontsize=17)
        legend_elements = [Line2D([0], [0], marker='o', mec='k', ms=10, lw=0,
                                  color=c, label=l)
                           for c, l, in zip(rel_colors, lg_lbls[1])]
        axs[1].legend(handles=legend_elements, ncol=1, fontsize=17)

        [rpl.axes_settings(ax, xlabel=xlabel[a], xticks=xticks,
                           ylabel=ylabel[a], xtlabels=xtlbls, xlim=xlim[a])
         for a, ax in enumerate(axs)]

        [ax.ticklabel_format(useOffset=False, axis='y') for ax in axs]

        ttl = fig.suptitle(headtitle, fontsize=22)
        ttl.set_position((.5, 1.06))

        plt.savefig(os.path.join(img_dir, fn), bbox_inches='tight')


def pdf_plot(fm_list, fo_list, fm_listr, fo_listr, models, nmod, obs,
             var, tres, tstat, units, years, regions, img_dir,
             grid_coords, map_conf, map_grid, map_sets, line_grid,
             line_sets):
    """
    Plotting frequency-intensity-distribution plot
    """
    # Reference model (first in the list in the configuration file (under
    # MODELS))
    ref_model = models[0]
    ref_mnme = ref_model.upper()
    othr_mod = models.copy()
    othr_mod.remove(ref_model)

    for reg in regions:

        fmod = {m: xa.open_dataset(f) for m, f in zip(models, fm_listr[reg])}
        mod_ann = {m: np.nanmean(fmod[m][var].values*100, axis=(1, 2))[1:]
                   for m in models}
        bins = fmod[ref_model].bin_edges.values[1:]
        nbins = bins.size

        if obs is not None:
            obslist = [obs] if not isinstance(obs, list) else obs
            ref_obs = obslist[0]
            obslbl = "_".join(s for s in obslist)
            fobs = {o: xa.open_dataset(f) for o, f in zip(obslist,
                                                          fo_listr[reg])}
            obs_ann = {o: np.nanmean(fobs[o][var].values*100, axis=(1, 2))[1:]
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

        thr = fmod[ref_model].attrs['Description'].split('|')[1].split(':')[1]
        thr = thr.replace(' ', '')
        thrnm = thr.replace('.', '')
        season = fmod[ref_model].attrs['Analysed time'].split('|')[1]
        seas = season.replace(' ', '')
        regnm = reg.replace(' ', '_')

        if thr != 'None':
            headtitle = '{} |  Threshold: {}\n{} | {}-{} | {}'.format(
                var, thr, reg, years[0], years[1], seas)
            if obs is not None:
                fn = '{}_thr{}_{}_pdf_{}_model_{}_{}-{}_{}.png'.format(
                    var, thrnm, tres, regnm, obslbl, years[0], years[1], seas)
            else:
                fn = '{}_thr{}_{}_pdf_{}_model_{}-{}_{}.png'.format(
                    var, thrnm, tres, regnm, years[0], years[1], seas)
        else:
            headtitle = '{} |  {} | {}-{} | {}'.format(var, reg, years[0],
                                                       years[1], seas)
            if obs is not None:
                fn = '{}_{}_pdf_{}_model_{}_{}-{}_{}.png'.format(
                    var, tres, regnm, obslbl, years[0], years[1], seas)
            else:
                fn = '{}_{}_pdf_{}_model_{}-{}_{}.png'.format(
                    var, tres, regnm, years[0], years[1], seas)

        # figure settings
        figsize = (19, 7)
        figshape = (1, 2)

        ylabel = ['Frequency (%)',
                  'Frequency (%)']
        ylim = [None]*2
        xlabel = ['({})'.format(units)]*2
        xlim = [[-.5, nbins-.5]]*2
        xticks = range(nbins-1)[::4]
        xtlbls = bins[:-1][::4]

        rpl.figure_init(plottype='line')
        fig, lgrid = rpl.fig_grid_setup(fshape=figshape, figsize=figsize,
                                        **line_grid)

        axs = rpl.make_line_plot(lgrid, ydata=dlist, **line_sets)
        if var == 'pr':
            axs[0].set_yscale('log')

        [l.set_color(lc) for l, lc in zip(axs[0].get_lines(), abs_colors)]
        [l.set_color(lc) for l, lc in zip(list(axs[1].get_lines())[:-1],
                                          rel_colors)]
        # Legend
        legend_elements = [Line2D([0], [0], lw=2, color=c, label=l)
                           for c, l in zip(abs_colors, lg_lbls[0])]
        axs[0].legend(handles=legend_elements, fontsize=17)
        legend_elements = [Line2D([0], [0], lw=2, color=c, label=l)
                           for c, l in zip(rel_colors, lg_lbls[1])]
        axs[1].legend(handles=legend_elements, fontsize=17)

        [rpl.axes_settings(ax, xlabel=xlabel[a], xticks=xticks,
                           ylabel=ylabel[a], xtlabels=xtlbls,
                           xlim=xlim[a], ylim=ylim[a])
         for a, ax in enumerate(axs)]

        ttl = fig.suptitle(headtitle, fontsize=22)
        ttl.set_position((.5, 1.06))

        plt.savefig(os.path.join(img_dir, fn), bbox_inches='tight')


def map_asop(fm_list, fo_list, fm_listr, fo_listr, models, nmod, obs, var,
             tres, tstat, units, ytitle, regions, img_dir, grid_coords,
             map_conf, map_grid, map_sets, line_grid, line_sets):
    """
    Plotting ASoP FC factor map plot
    """

    # Reference model (first in the list in the configuration file (under
    # MODELS))
    ref_model = models[0]
    ref_mnme = ref_model.upper()
    othr_mod = models.copy()
    othr_mod.remove(ref_model)

    # Obs data list
    obslist = [obs] if not isinstance(obs, list) else obs

    # Map configurations
    if (map_conf['zoom'] == 'crnrs' and
            'crnr_vals' not in map_conf):
        map_conf.update({
            'crnr_vals': grid_coords['meta data'][var][ref_model]['crnrs']
        })
    if 'lon_0' not in map_conf:
        map_conf.update({
            'lon_0': grid_coords['meta data'][var][ref_model]['lon_0']
        })
    if map_conf['proj'] == 'lcc' and 'lat_0' not in map_conf:
        map_conf.update({
            'lat_0': grid_coords['meta data'][var][ref_model]['lat_0']
        })
    target_grid_names = list(grid_coords['target grid'][var]['lon'].keys())
    # Native grids are not possible for this plot, unless model(s)/obs have
    # same native grid configurations.
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
            ftitles = ['{} - {}'.format(m.upper(), ref_obs)
                       for m in models + obslist[1:]]
        else:
            ndata = nmod
            ftitles = ['{} - {}'.format(m.upper(), ref_obs)
                       for m in models]
    else:
        FCI = {m: np.nansum(
            np.fabs(fmod[m][var].values[1, :] -
                    fmod[ref_model][var].values[1, :]), axis=0)
               for m in othr_mod}
        dlist = [FCI[m] for m in othr_mod]
        ndata = nmod-1
        ftitles = ['{} - {}'.format(m.upper(), ref_mnme) for m in othr_mod]

    thr = fmod[ref_model].attrs['Description'].split('|')[1].split(':')[1]
    thr = thr.replace(' ', '')
    thrnm = thr.replace('.', '')
    season = fmod[ref_model].attrs['Analysed time'].split('|')[1]
    seas = season.replace(' ', '')

    # figure settings
    figsize = (20, 8)
    figshape = (1, ndata)

    if thr != 'None':
        headtitle = 'ASoP FC Index | Thr: {}\n{} | {}'.format(
            thr, ytitle, seas)
        if obs is not None:
            fn = 'asop_FC_thr{}_{}_map_model_{}_{}_{}.png'.\
                format(thrnm, tres, obslbl, ytitle.replace(' ', '_'), seas)
        else:
            fn = 'asop_FC_thr{}_{}_map_model_{}_{}.png'.\
                format(thrnm, tres, ytitle.replace(' ', '_'), seas)
    else:
        headtitle = 'ASoP FC Index | {} | {}'.format(ytitle, seas)
        if obs is not None:
            fn = 'asop_FC_{}_map_model_{}_{}_{}.png'.\
                format(tres, obslbl, ytitle.replace(' ', '_'), seas)
        else:
            fn = 'asop_FC_{}_map_model_{}_{}.png'.\
                format(tres, ytitle.replace(' ', '_'), seas)

    rpl.figure_init(plottype='map')

    # Setup figure grid with colorbar settings
    fig, grid = rpl.image_grid_setup(figsize=figsize, fshape=figshape,
                                     **map_grid)

    lts = grid_coords['target grid'][var]['lat'][tgname]
    lns = grid_coords['target grid'][var]['lon'][tgname]

    # Create map object
    m, coords = rpl.map_setup(grid, lts, lns, **map_conf)

    cmap = [colors.getcolormap('PuRd', custom=False)]*ndata
    clevs = [np.linspace(0, 2, 21)]*ndata

    mp = rpl.make_map_plot(dlist, grid, m, coords,  cmap=cmap,
                           clevs=clevs, **map_sets)
    rpl.image_colorbar(mp, grid, labelspacing=2, formatter='{:.1f}')

    # Map settings
    rpl.map_axes_settings(fig, grid, headtitle=headtitle)

    [ax.text(0.5, 1.02, ft.upper(), size=21, va='center', ha='center',
             transform=ax.transAxes) for ft, ax in zip(ftitles, grid)]

    plt.savefig(os.path.join(img_dir, fn), bbox_inches='tight')

    # Line plot asop
    if fm_listr is not None:
        line_asop(fm_listr, fo_listr, models, nmod, obs, var, tres, tstat,
                  units, ytitle, regions, img_dir, line_grid, line_sets)


def line_asop(fm_listr, fo_listr, models, nmod, obs, var, tres, tstat,
              units, ytitle, regions, img_dir, line_grid, line_sets):
    """
    Plotting ASoP line plot of C and FC factors
    """
    # Reference model (first in the list in the configuration file (under
    # MODELS))
    ref_model = models[0]
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
        for ff, fctr in enumerate(factors):
            # obs_mn = np.nanmean(fobs[var].values[ff, :], axis=(1, 2))
            # mod_mn = [np.nanmean(fmod[m][var].values[ff, :], axis=(1, 2))
            #           for m in models]
            #     obs_mn *= 100
            #     mod_mn = [arr*100 for arr in mod_mn]

            sc = 100 if fctr == 'FC' else 1
            dlist_ff = [[arr[ff, :]*sc for arr in dlist[0]],
                        [arr[ff, :]*sc for arr in dlist[1]]]
            xdata = [[bins[:-1]]*len(dlist[0]), [bins[:-1]]*len(dlist[1])]

            thr = fmod[ref_model].attrs['Description'].\
                split('|')[1].split(':')[1]
            thr = thr.replace(' ', '')
            thrnm = thr.replace('.', '')
            season = fmod[ref_model].attrs['Analysed time'].split('|')[1]
            seas = season.replace(' ', '')
            regnm = reg.replace(' ', '_')

            if thr != 'None':
                headtitle = 'ASoP ({}) |  Thr: {}\n{} | {} | {}'.format(
                    fctr, thr, reg, ytitle, seas)
                if obs is not None:
                    fn = '{}_thr{}_{}_asop_{}_{}_model_{}_{}_{}.png'.format(
                        var, thrnm, tres, fctr, regnm, obslbl,
                        ytitle.replace(' ', '_'), seas)
                else:
                    fn = '{}_thr{}_{}_asop_{}_{}_model_{}_{}.png'.format(
                        var, thrnm, tres, fctr, regnm,
                        ytitle.replace(' ', '_'), seas)
            else:
                headtitle = 'ASoP ({}) |  {} | {} | {}'.format(
                    fctr, reg, ytitle, seas)
                if obs is not None:
                    fn = '{}_{}_asop_{}_{}_model_{}_{}_{}.png'.format(
                        var, tres, fctr, regnm, obslbl,
                        ytitle.replace(' ', '_'), seas)
                else:
                    fn = '{}_{}_asop_{}_{}_model_{}_{}.png'.format(
                        var, tres, fctr, regnm, ytitle.replace(' ', '_'), seas)

            # figure settings
            figsize = (13, 10)
            figshape = (2, 1)

            ylabel = [ylabels[ff]]*2
            ylim = [None]*2
            xlabel = [None, 'Intensity ({})'.format(units)]
            xlim = [[1e-4, 1e2]]*2

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

            [l.set_color(lc) for l, lc in zip(axs[0].get_lines(), abs_colors)]
            [l.set_color(lc) for l, lc in zip(list(axs[1].get_lines())[:-1],
                                              rel_colors)]
            # Legend
            legend_elements = [Line2D([0], [0], lw=2, color=c, label=l)
                               for c, l in zip(abs_colors, lg_lbls[0])]
            axs[0].legend(handles=legend_elements, fontsize=17)
            legend_elements = [Line2D([0], [0], lw=2, color=c, label=l)
                               for c, l in zip(rel_colors, lg_lbls[1])]
            axs[1].legend(handles=legend_elements, fontsize=17)

            [rpl.axes_settings(ax, xlabel=xlabel[a],  # xticks=xticks,
                               ylabel=ylabel[a],  # xtlabels=xtlbls,
                               xlim=xlim[a], ylim=ylim[a])
             for a, ax in enumerate(axs)]

            ttl = fig.suptitle(headtitle, fontsize=22)
            ttl.set_position((.5, 1.06))

            plt.savefig(os.path.join(img_dir, fn), bbox_inches='tight')
