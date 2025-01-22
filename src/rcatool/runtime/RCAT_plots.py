""" Module script for plotting """

import os
import sys
import xarray as xa
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import math
import rcatool.plot.plots as rpl
from rcatool.utils.polygons import mask_region
from rcatool.stats.arithmetics import run_mean
from copy import deepcopy

# Colors
set1 = mpl.cm.Set1.colors
set1_m = [[int(255*x) for x in triplet] for triplet in set1]
colors = ['#{:02x}{:02x}{:02x}'.format(s[0], s[1], s[2]) for s in set1_m]


class PlotConfiguration(object):
    """
    The plotting configuration and applications.
    """

    def __init__(self, plotdict, statistic):

        pdict = deepcopy(plotdict)
        self.statistic = statistic

        # Reference model and the rest
        self.models = pdict['models']
        self.nmod = len(self.models)
        self.ref_model = self. models[0]
        self.othr_mod = self.models.copy()
        self.othr_mod.remove(self.ref_model)

        # Obs data list
        obs = pdict['observation']
        self.obslist = [obs] if not isinstance(obs, list) else obs
        self.ref_obs = self.obslist[0]
        self.obslbl = "_".join(s for s in self.obslist)\
            if self.ref_obs is not None else None

        self.var = pdict['variable']
        self.tres = pdict['time res']
        self.tstat = pdict['stat method']
        self.units = pdict['units']

        self.time_suffix_dd = pdict['time suffix dict']
        tsuffix_ll = [val for key, val in self.time_suffix_dd.items()]
        tsuffix_ll.sort()
        self.tsuffix_fname = "_vs_".join(set(tsuffix_ll))
        self.tsuffix_title = self.tsuffix_fname.replace('_', ' ')

        self.regions = pdict['regions']
        self.img_dir = pdict['img dir']

        # Map settings
        self.map_projection = pdict['map projection']
        self.map_config = pdict['map config']
        self.map_extent = pdict['map extent']
        self.map_gridlines = pdict['map gridlines']
        self.map_axes_conf = self._map_grid_setup(
            pdict['map grid setup'])
        self.map_plot_conf = pdict['map plot kwargs']

        grid_coords = pdict['grid coords']
        target_grid_names = list(
            grid_coords['target grid'][self.var]['lon'].keys())
        tgname = target_grid_names[0]
        map_domain = pdict['map domain']
        domain_model = map_domain if map_domain else self.ref_model
        domain = grid_coords['meta data'][self.var][
            domain_model]['domain']
        self.mask = mask_region(
            grid_coords['target grid'][self.var]['lon'][tgname],
            grid_coords['target grid'][self.var]['lat'][tgname], domain)
        self.lts = grid_coords['target grid'][self.var]['lat'][tgname]
        self.lns = grid_coords['target grid'][self.var]['lon'][tgname]

        self.moments_plot_conf = pdict['moments plot config']

        # Line plot settings
        self.line_grid = pdict['line grid setup']
        self.line_sets = pdict['line kwargs']
        self.abs_colors = colors[:]
        if self.ref_obs is not None:
            self.abs_colors.insert(0, 'k')
        self.rel_colors = self.abs_colors[1:]

        # File lists
        self.fm_list = pdict['mod files'][self.statistic]
        self.fo_list = pdict['obs files'][self.statistic]
        if self.regions is not None:
            self.fm_listr = pdict['mod reg files'][self.statistic]
            self.fo_listr = pdict['obs reg files'][self.statistic]
        else:
            self.fm_listr = None
            self.fo_listr = None

    def run_plotting_application(self):
        plot_funcs = {
            'seasonal cycle': self.map_season,
            'annual cycle': self.map_ann_cycle,
            'percentile': self.map_pctls,
            'diurnal cycle': self.map_diurnal_cycle,
            'pdf': self.pdf_plot,
            'moments': self.moments_plot,
            'asop': self.map_asop,
        }

        # types_of_diff = ['absolute', 'relative']
        self.include_relative_change = True
        self.plot_mulc = 2 if self.include_relative_change else 1

        # for diff_type in types_of_diff:
        # Call plot function
        plot_funcs[self.statistic]()

    def _map_grid_setup(self, map_grid_set):
        """
        Potentially modify map grid settings for map plots
        """
        if self.statistic in ('annual cycle', 'diurnal cycle', 'asop'):
            if 'cbar_mode' not in map_grid_set:
                map_grid_set.update({'cbar_mode': 'single'})
                map_grid_set.update({'cbar_location': 'right'})
                map_grid_set.update({'cbar_size': '2%'})
                map_grid_set.update({'cbar_pad': 0.06})
            if 'axes_pad' not in map_grid_set:
                map_grid_set.update({'axes_pad': 0.1})
        if self.statistic in ('seasonal cycle'):
            if 'cbar_mode' not in map_grid_set:
                map_grid_set.update({'cbar_mode': 'edge'})
                map_grid_set.update({'cbar_location': 'right'})
                map_grid_set.update({'cbar_size': '5%'})
                map_grid_set.update({'cbar_pad': 0.04})
            if 'axes_pad' not in map_grid_set:
                map_grid_set.update({'axes_pad': 0.1})
        if self.statistic in ('percentile'):
            if 'cbar_mode' not in map_grid_set:
                map_grid_set.update({'cbar_mode': 'single'})
                map_grid_set.update({'cbar_location': 'right'})
                map_grid_set.update({'cbar_size': '5%'})
                map_grid_set.update({'cbar_pad': 0.04})
            if 'axes_pad' not in map_grid_set:
                map_grid_set.update({'axes_pad': 0.4})

        return map_grid_set

    def _round_down(self, n, decimals=0):
        multiplier = 10 ** decimals
        return math.floor(n * multiplier) / multiplier

    def _round_up(self, n, decimals=0):
        multiplier = 10 ** decimals
        return math.ceil(n * multiplier) / multiplier

    def _mask_data(self, ds_in):
        mdata = ds_in[self.var].where(self.mask)
        ds_out = mdata.to_dataset()
        ds_out.attrs = ds_in.attrs
        return ds_out

    def round_to_sign_digits(self, x, sig=2):
        out = round(x, sig-int(np.floor(np.log10(abs(x))))-1)\
                if not x == 0 else x
        return out

    def get_clevs(self, data, centered=False):
        from scipy.stats import skew
        from decimal import Decimal

        def _calc_clevels(int_min, int_max):
            if 1 <= abs(int_min) < 10:
                factor = 10
            elif 0.1 <= abs(int_min) < 1:
                factor = 1e2
            elif abs(int_min) < 0.1:
                factor = 1e3
            else:
                factor = 1
            _min = int(int_min * factor)
            _max = int(int_max * factor)
            diff = _max - _min

            N_samples = np.arange(10, 21)
            idx, = np.where([diff % val == 0
                             for val in N_samples])
            sample_nlvs = N_samples[idx]
            list_nsamples = np.array(list(sample_nlvs) +
                                     list(sample_nlvs+1))
            avg_dec = []
            for n in list_nsamples:
                lvls = np.linspace(_min, _max, n)
                dec = np.mean([(str(x))[::-1].find('.')
                               for x in lvls])
                avg_dec.append(dec)
            indices = [i for i, v in enumerate(avg_dec)
                       if v == min(avg_dec)]
            if len(indices) > 1:
                sign_num = []
                for ix in indices:
                    n = list_nsamples[ix]
                    cl = np.linspace(_min, _max, n)
                    sign_num.append(np.mean([_count_sigfigs(str(x))
                                             for x in cl]))
                sel_idx = np.argmin(sign_num)
                sel_n = list_nsamples[indices[sel_idx]]
            else:
                sel_n = list_nsamples[indices[0]]

            sel_clevs = np.linspace(_min, _max, sel_n)
            sel_clevs = sel_clevs / factor
            return sel_clevs

        def _count_sigfigs(numstr):
            return len(Decimal(numstr).normalize().as_tuple().digits)

        if centered:
            abs_max = np.nanpercentile(data, 98)
            abs_min = np.nanpercentile(data, 2)
        else:
            if skew(data.ravel()[~np.isnan(data.ravel())]) > 2:
                abs_max = np.nanpercentile(data, 99)
                abs_min = np.nanmin(data)
            else:
                abs_max = np.nanpercentile(data, 97)
                abs_min = np.nanmin(data)
        if centered:
            abs_max = np.maximum(abs(abs_max), abs(abs_min))
            abs_min = -abs_max
        if ((abs(abs_max) < 1.0) & (abs(abs_min) < 1.0)):
            nz = np.floor(abs(np.log10(abs(abs_max))))
            round_max = self._round_up(abs_max, nz+1)
            if ((abs(abs_min) <= 0.2) and not centered):
                round_min = 0.0
            else:
                nz = np.floor(abs(np.log10(abs(abs_min))))
                round_min = self._round_down(abs_min, nz+1)
        elif ((abs_max < 10) & (abs_min < 10)):
            round_max = self._round_up(abs_max, 0)
            round_min = self._round_down(abs_min, 0)
        else:
            round_max = self._round_up(abs_max, -1)
            round_min = self._round_down(abs_min, -1)

        clevs = _calc_clevels(round_min, round_max)

        return clevs

    def _space_dim(self, ds):
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

    def _get_colorbar_label_formatting(self, clevs):
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

    def define_figure_titles(self):
        if self.obslist[0] is not None:
            ref_name = self.ref_obs
            if len(self.obslist) > 1:
                data_name_list = self.models + self.obslist[1:]
            else:
                data_name_list = self.models
        else:
            ref_name = self.ref_model
            data_name_list = self.othr_mod

        if self.statistic == 'asop':
            ftitles = [(f"{m.upper()} {self.time_suffix_dd[m]} -\n "
                        f"{ref_name.upper()} {self.time_suffix_dd[ref_name]}")
                       for m in data_name_list]
        else:
            ftitles = [f"{ref_name.upper()} {self.time_suffix_dd[ref_name]}"]\
                + [(f"{m.upper()} {self.time_suffix_dd[m]} -\n "
                    f"{ref_name.upper()} {self.time_suffix_dd[ref_name]}")
                    for m in data_name_list] * self.plot_mulc
        ftitles = [ft.replace('_', ' ') for ft in ftitles]

        return ftitles

    def define_file_names(self, thr, plot_type, stat_name=None,
                          data_name=None, region=None):
        if stat_name is None:
            stat_str = f'{plot_type}_{self.statistic.replace(" ", "_")}'
        else:
            stat_str = f'{plot_type}_{stat_name}'
        stat_str = f'{stat_str}_{region}' if region is not None else stat_str
        thr_str = '' if thr == 'None' else f'_thr{thr}'
        obs_name = f'{self.obslbl}_' if self.obslist[0] is not None else ''
        dname_str = f'model_{obs_name.lower()}' if data_name is None else\
            f'{data_name.lower()}_'

        fn = (f"{self.var}{thr_str}{self.tres}{self.tstat}_"
              f"{stat_str}_{dname_str}{self.tsuffix_fname}.png")

        return fn

    def map_season(self):
        """
        Plotting seasonal mean map plot
        """

        # Data
        fmod = {m: xa.open_dataset(f)
                for m, f in zip(self.models, self.fm_list)}
        fmod_msk = {m: self._mask_data(ds)
                    for m, ds in fmod.items()}

        if self.ref_obs is not None:
            fobs = {o: xa.open_dataset(f)
                    for o, f in zip(self.obslist, self.fo_list)}
            fobs_msk = {o: self._mask_data(ds)
                        for o, ds in fobs.items()}
            dlist = [fobs_msk[self.ref_obs][self.var].values[i, :]
                     for i in range(4)] +\
                    [fmod_msk[m][self.var].values[i, :] -
                     fobs_msk[self.ref_obs][self.var].values[i, :]
                     for m in self.models for i in range(4)]
            if self.include_relative_change:
                dlist_rel = [
                    fobs_msk[self.ref_obs][self.var].values[i, :]
                    for i in range(4)] +\
                    [(fmod_msk[m][self.var].values[i, :] /
                     fobs_msk[self.ref_obs][self.var].values[i, :]-1)*100
                     for m in self.models for i in range(4)]
            if len(self.obslist) > 1:
                dlist += [fobs_msk[o][self.var].values[i, :] -
                          fobs_msk[self.ref_obs][self.var].values[i, :]
                          for o in self.obslist[1:] for i in range(4)]
                if self.include_relative_change:
                    dlist_rel += [
                        (fobs_msk[o][self.var].values[i, :] /
                         fobs_msk[self.ref_obs][self.var].values[i, :]-1)*100
                        for o in self.obslist[1:] for i in range(4)]
                ndata = self.nmod + len(self.obslist[1:])
            else:
                ndata = self.nmod
        else:
            dlist = [fmod_msk[self.ref_model][self.var].values[i, :]
                     for i in range(4)] +\
                    [fmod_msk[m][self.var].values[i, :] -
                     fmod_msk[self.ref_model][self.var].values[i, :]
                     for m in self.othr_mod for i in range(4)]
            if self.include_relative_change:
                dlist_rel = [
                    fmod_msk[self.ref_model][self.var].values[i, :]
                    for i in range(4)] +\
                        [(fmod_msk[m][self.var].values[i, :] /
                         fmod_msk[self.ref_model][self.var].values[i, :]
                         - 1)*100 for m in self.othr_mod for i in range(4)]
            ndata = self.nmod-1

        data_list = [dlist, dlist_rel] if\
            self.include_relative_change else [dlist]

        # figure settings
        figshape = (ndata + 1, 4)
        if np.prod(figshape) > 8:
            figsize = (22, 14)
        else:
            figsize = (20, 9)

        # labels
        thr = fmod_msk[self.ref_model].attrs['Description'].\
            split('|')[2].split(':')[1].strip()

        ftitles = self.define_figure_titles()
        units = [self.units, f'{self.units}, diff: (%)'] if\
            self.include_relative_change else [self.units]
        stat_names = [f"{self.statistic.replace(' ', '_')}_{x}"
                      for x in ['abs_diff', 'rel_diff']]

        for dd, st_nme, uts in zip(data_list, stat_names, units):
            headtitle = f'{self.var} [{uts}]' if thr == 'None' else\
                f'{self.var} [{uts}] | Threshold: {thr}'

            fn = self.define_file_names(thr, 'map', stat_name=st_nme)

            # color maps
            if self.var == 'pr':
                cmap = [mpl.cm.YlGnBu]*4 + [mpl.cm.BrBG]*ndata*4
            else:
                cmap = [mpl.cm.Spectral_r]*4 + [mpl.cm.RdBu_r]*ndata*4

            clevs_abs = self.get_clevs(np.array(dd[0:4]), centered=False)
            clevs_dif = self.get_clevs(np.array(dd[4:8]), centered=True)
            fmt_abs = self._get_colorbar_label_formatting(clevs_abs[::2])
            fmt_dif = self._get_colorbar_label_formatting(clevs_dif[::2])

            clevs = [clevs_abs]*4 + [clevs_dif]*ndata*4
            fmt = [fmt_abs]*4 + [fmt_dif]*ndata*4

            rpl.figure_init(plottype='map')

            # Create map object and axes grid
            map_proj = rpl.define_map_object(
                self.map_projection, **self.map_config)
            fig, axs_grid = rpl.map_setup(
                map_proj, self.map_extent, figsize, figshape,
                grid_lines=self.map_gridlines, **self.map_axes_conf)

            # Plot the maps
            mp = rpl.make_map_plot(
                dd, axs_grid, self.lts, self.lns, cmap=cmap, clevs=clevs,
                **self.map_plot_conf)
            rpl.image_colorbar(mp, axs_grid, labelspacing=2, formatter=fmt)

            # Add contour plot if mslp
            if self.var == 'psl':
                rpl.make_map_plot(
                    dd, axs_grid, self.lts, self.lns, clevs=clevs,
                    filled=False, colors='#4f5254', linewidths=1.3)

            # Map settings
            rpl.map_axes_settings(fig, axs_grid, headtitle=headtitle,
                                  time_mean='season')

            # Annotate
            [ax.text(-0.08, 0.5, ft.upper(), va='center', ha='center',
                     rotation=90, transform=ax.transAxes)
             for ft, ax in zip(ftitles, [
                 axs_grid[i] for i in [p*4 for p in range(ndata+1)]])]

            plt.savefig(os.path.join(self.img_dir, fn), bbox_inches='tight')

        # Box plot seasonal cycle
        if self.fm_listr is not None:
            self.box_seas_cycle()

    def box_seas_cycle(self):
        """
        Plotting seasonal cycle box plot
        """

        def _flatten(arr):
            arr_1d = arr.ravel()
            arr_out = arr_1d[~np.isnan(arr_1d)]
            return arr_out

        seasons = {'DJF': 0, 'MAM': 1, 'JJA': 2, 'SON': 3}
        for reg in self.regions:
            fmod = {m: xa.open_dataset(f)
                    for m, f in zip(self.models, self.fm_listr[reg])}
            mdata = {s: {m: arr[self.var].values[i, :]
                         for m, arr in fmod.items()}
                     for s, i in seasons.items()}

            if self.ref_obs is not None:
                fobs = {s: {o: xa.open_dataset(f)[self.var].values[i, :]
                            for o, f in zip(self.obslist, self.fo_listr[reg])}
                        for s, i in seasons.items()}
                if len(self.obslist) > 1:
                    fdiff_m = {s: {m: mdata[s][m] - fobs[s][self.ref_obs]
                                   for m in self.models} for s in seasons}
                    fdiff_o = {s: {o: fobs[s][o] - fobs[s][self.ref_obs]
                                   for o in self.obslist[1:]} for s in seasons}
                    fdiff = [fdiff_m | fdiff_o]
                    if self.include_relative_change:
                        fdiff_mr = {s: {m: (mdata[s][m] / fobs[s][self.ref_obs]
                                            - 1)*100
                                        for m in self.models} for s in seasons}
                        fdiff_or = {s: {o: (fobs[s][o] / fobs[s][self.ref_obs]
                                            - 1)*100
                                        for o in self.obslist[1:]}
                                    for s in seasons}
                        fdiff += [fdiff_mr | fdiff_or]
                    dlist = [{s: [_flatten(fobs[s][self.ref_obs])] +
                              [_flatten(mdata[s][m])
                               for m in self.models + self.obslist[1:]]
                              for s, i in seasons.items()}] +\
                            [{s: [_flatten(fd[s][m])
                                  for m in self.models + self.obslist[1:]]
                              for s, i in seasons.items()} for fd in fdiff]
                    ll_nms = self.models + self.obslist[1:]
                else:
                    fdiff = [{s: {m: mdata[s][m] - fobs[s][self.ref_obs]
                                  for m in self.models} for s in seasons}]
                    if self.include_relative_change:
                        fdiff_r = [{s: {m: (mdata[s][m] / fobs[s][self.ref_obs]
                                            - 1)*100 for m in self.models}
                                    for s in seasons}]
                        fdiff += fdiff_r

                    dlist = [{s: [_flatten(fobs[s][self.ref_obs])] +
                              [_flatten(mdata[s][m]) for m in self.models]
                              for s, i in seasons.items()}] +\
                            [{s: [_flatten(fd[s][m]) for m in self.models]
                              for s, i in seasons.items()} for fd in fdiff]
                    ll_nms = self.models

                lg_lbls = [[self.ref_obs] + [m.upper() for m in ll_nms]] +\
                    [[f'{m.upper()} - {self.ref_obs}' for m in ll_nms]] *\
                    self.plot_mulc
            else:
                fdiff = [{s: {m: mdata[s][m] - mdata[s][self.ref_model]
                              for m in self.othr_mod} for s in seasons}]
                if self.include_relative_change:
                    fdiff_r = [{s: {m: (mdata[s][m] / mdata[s][self.ref_model]
                                        - 1)*100
                                    for m in self.othr_mod} for s in seasons}]
                    fdiff += fdiff_r

                dlist = [{s: [_flatten(mdata[s][m]) for m in self.models]
                          for s, i in seasons.items()}] +\
                        [{s: [_flatten(fd[s][m]) for m in self.othr_mod]
                          for s, i in seasons.items()} for fd in fdiff]
                lg_lbls = [[m.upper() for m in self.models]] +\
                    [[f'{m.upper()} - {self.ref_model.upper()}'
                      for m in self.othr_mod]] * self.plot_mulc

            thr = fmod[self.ref_model].attrs['Description'].\
                split('|')[2].split(':')[1].strip()
            regnm = reg.replace(' ', '_')

            fn = self.define_file_names(thr, 'bxplot', region=regnm)
            headtitle = f'{self.var} | {reg} | {self.tsuffix_title}' if\
                thr == 'None' else\
                f'{self.var} | Threshold: {thr} | {reg} | {self.tsuffix_title}'

            # figure settings
            if self.include_relative_change:
                figsize = (20, 8)
                figshape = (1, 3)
                ylabel = [f'{self.units}',
                          f'Difference ({self.units})',
                          'Difference (%)']
            else:
                figsize = (18, 8)
                figshape = (1, 2)
                ylabel = [f'{self.units}',
                          f'Difference ({self.units})']

            ylim = [None]*(self.plot_mulc + 1)
            xlabel = [None]*(self.plot_mulc + 1)
            xlim = [None]*(self.plot_mulc + 1)
            xticks = [None]*(self.plot_mulc + 1)
            xtlbls = [None]*(self.plot_mulc + 1)

            rpl.figure_init(plottype='box')
            fig, lgrid = rpl.fig_grid_setup(fshape=figshape, figsize=figsize,
                                            **self.line_grid)

            lbls = [list(seasons.keys())]*(self.plot_mulc + 1)
            axs, bps = rpl.make_box_plot(
                lgrid, data=dlist, labels=lbls, leg_labels=None,
                grouped=True, whis=[5, 95], showfliers=False)

            # breakpoint()
            _ = [rpl._decorate_box(axs[0], bps[0][i], self.abs_colors)
                 for s, i in seasons.items()]
            _ = [[rpl._decorate_box(ax, bp[i], self.rel_colors)
                  for s, i in seasons.items()]
                 for ax, bp in zip(axs[1:], bps[1:])]

            # Legend
            leg_elements = [Patch(color=c, label=l)
                            for c, l in zip(
                                self.abs_colors, lg_lbls[0])]

            axs[0].legend(
                handles=leg_elements, fontsize='medium', framealpha=.5)
            leg_elements = [Patch(color=c, label=l)
                            for c, l in zip(
                                self.rel_colors, lg_lbls[1])]
            [ax.legend(handles=leg_elements, fontsize='medium', framealpha=.5)
             for ax in axs[1:]]

            [rpl.axes_settings(
                ax, xlabel=xlabel[a], xticks=xticks[a], xtlabels=xtlbls[a],
                xlim=xlim[a], ylabel=ylabel[a], ylim=ylim[a],
                fontsize='large', fontsize_lbls='large')
             for a, ax in enumerate(axs)]

            ttl = fig.suptitle(headtitle, fontsize='x-large')
            ttl.set_position((.5, 1.04))

            plt.savefig(os.path.join(self.img_dir, fn), bbox_inches='tight')

    def map_ann_cycle(self):
        """
        Plotting annual cycle map plot
        """

        # Data
        fmod = {m: xa.open_dataset(f)
                for m, f in zip(self.models, self.fm_list)}
        fmod_msk = {m: self._mask_data(ds) for m, ds in fmod.items()}

        if self.ref_obs is not None:
            fobs = {o: xa.open_dataset(f)
                    for o, f in zip(self.obslist, self.fo_list)}
            fobs_msk = {o: self._mask_data(ds) for o, ds in fobs.items()}

            ll_abs = [[fobs_msk[self.ref_obs][self.var].values[i, :]
                       for i in range(12)]]
            ll_diff = [[fmod_msk[m][self.var].values[i, :] -
                        fobs_msk[self.ref_obs][self.var].values[i, :]
                        for i in range(12)] for m in self.models]
            if len(self.obslist) > 1:
                ll_diff += [[fobs_msk[o][self.var].values[i, :] -
                             fobs_msk[self.ref_obs][self.var].values[i, :]
                             for i in range(12)] for o in self.obslist[1:]]
            if self.include_relative_change:
                ll_diff_rel = [
                    [(fmod_msk[m][self.var].values[i, :] /
                      fobs_msk[self.ref_obs][self.var].values[i, :] - 1)*100
                     for i in range(12)] for m in self.models]
                if len(self.obslist) > 1:
                    ll_diff_rel += [
                        [(fobs_msk[o][self.var].values[i, :] /
                          fobs_msk[self.ref_obs][self.var].values[i, :]
                          - 1)*100 for i in range(12)]
                        for o in self.obslist[1:]]
            if len(self.obslist) > 1:
                ndata = (self.nmod + len(self.obslist[1:]))
                data_names = [self.ref_obs] + [
                    f"{m}-{self.ref_obs}"
                    for m in self.models + self.obslist[1:]] * self.plot_mulc
            else:
                ndata = self.nmod
                data_names = [self.ref_obs] +\
                    [f"{m}-{self.ref_obs}" for m in self.models] *\
                    self.plot_mulc
        else:
            ll_abs = [[fmod_msk[self.ref_model][self.var].values[i, :]
                       for i in range(12)]]

            ll_diff = [[fmod_msk[m][self.var].values[i, :] -
                        fmod_msk[self.ref_model][self.var].values[i, :]
                        for i in range(12)] for m in self.othr_mod]
            if self.include_relative_change:
                ll_diff_rel = [
                    [(fmod_msk[m][self.var].values[i, :] /
                      fmod_msk[self.ref_model][self.var].values[i, :] - 1)*100
                     for i in range(12)] for m in self.othr_mod]
            ndata = (self.nmod - 1)
            data_names = [self.ref_model.upper()] +\
                [f"{m}-{self.ref_model.upper()}"
                 for m in self.othr_mod] * self.plot_mulc

        dlist = ll_abs + ll_diff + ll_diff_rel if\
            self.include_relative_change else ll_abs + ll_diff

        # figure settings
        figsize = (18, 14)
        figshape = (3, 4)

        # color maps
        if self.var == 'pr':
            cmap = [mpl.cm.YlGnBu] + [mpl.cm.BrBG]*ndata*self.plot_mulc
        else:
            cmap = [mpl.cm.Spectral_r] + [mpl.cm.RdBu_r]*ndata*self.plot_mulc

        clevs_abs = self.get_clevs(np.array(dlist[0]), centered=False)
        clevs_dif = self.get_clevs(np.array(dlist[1]), centered=True)
        fmt_abs = self._get_colorbar_label_formatting(clevs_abs[::2])
        fmt_dif = self._get_colorbar_label_formatting(clevs_dif[::2])

        if self.include_relative_change:
            clevs_rel = self.get_clevs(np.array(dlist[1+ndata]), centered=True)
            fmt_rel = self._get_colorbar_label_formatting(clevs_rel[::2])
            clevs = [clevs_abs] + [clevs_dif]*ndata + [clevs_rel]*ndata
            fmt = [fmt_abs] + [fmt_dif]*ndata + [fmt_rel]*ndata
        else:
            clevs = [clevs_abs] + [clevs_dif]*ndata
            fmt = [fmt_abs] + [fmt_dif]*ndata

        thr = fmod_msk[self.ref_model].attrs['Description'].\
            split('|')[2].split(':')[1].strip()
        units = [self.units] * (ndata + 1)
        fn_stat_names = [None] + [(f"{self.statistic.replace(' ', '_')}"
                                   f"_abs_diff")] * ndata
        ftitles = self.define_figure_titles()

        if self.include_relative_change:
            fn_stat_names = fn_stat_names + [
                f'{self.statistic.replace(' ', '_')}_rel_diff'] * ndata
            units = units + ["%"] * ndata

        # Loop over data sets
        for p, ft, data_name, st_nme, uts in zip(range(ndata*self.plot_mulc+1),
                                                 ftitles, data_names,
                                                 fn_stat_names, units):
            headtitle = f'{ft} | {self.var} [{uts}]'\
                    if thr == 'None' else\
                    f'{ft} | {self.var} [{uts}] | Threshold: {thr}'

            fn = self.define_file_names(
                thr, 'map', data_name=data_name, stat_name=st_nme)

            rpl.figure_init(plottype='map')

            # Create map object and axes grid
            map_proj = rpl.define_map_object(
                self.map_projection, **self.map_config)
            fig, axs_grid = rpl.map_setup(
                map_proj, self.map_extent, figsize, figshape,
                grid_lines=self.map_gridlines, **self.map_axes_conf)

            # Plot the maps
            mp = rpl.make_map_plot(
                dlist[p], axs_grid, self.lts, self.lns, cmap=cmap[p],
                clevs=clevs[p], **self.map_plot_conf)
            rpl.image_colorbar(mp, axs_grid, labelspacing=2, formatter=fmt[p])

            # Add contour plot if mslp
            if self.var == 'psl':
                rpl.make_map_plot(
                    dlist[p], axs_grid, self.lts, self.lns, clevs=clevs[p],
                    filled=False, colors='#4f5254', linewidths=1.3)
            # [plt.clabel(c, fmt='%.1f', colors='k', fontsize=15) for c in lp]

            # Map settings
            rpl.map_axes_settings(fig, axs_grid, fontsize='large',
                                  headtitle=headtitle, time_mean='month')

            plt.savefig(os.path.join(self.img_dir, fn), bbox_inches='tight')

        # Line plot annual cycle
        if self.fm_listr is not None:
            self.line_ann_cycle()

    def line_ann_cycle(self):
        """
        Plotting annual cycle line plot
        """

        for reg in self.regions:

            fmod = {m: xa.open_dataset(f)
                    for m, f in zip(self.models, self.fm_listr[reg])}
            mod_data = {m: np.nanmean(fmod[m][self.var].values, axis=(1, 2))
                        for m in self.models}
            if self.ref_obs is not None:
                fobs = {o: xa.open_dataset(f)
                        for o, f in zip(self.obslist, self.fo_listr[reg])}
                obs_data = {o: np.nanmean(
                    fobs[o][self.var].values, axis=(1, 2))
                    for o in self.obslist}
                ll_abs = [[obs_data[self.ref_obs]] +
                          [mod_data[m] for m in self.models]]
                ll_diff = [[mod_data[m] - obs_data[self.ref_obs]
                            for m in self.models]]
                ll_diff_rel = [[(mod_data[m] / obs_data[self.ref_obs] - 1)*100
                                for m in self.models]]

                if len(self.obslist) > 1:
                    ll_abs[0] += [obs_data[o] for o in self.obslist[1:]]
                    ll_diff[0] += [obs_data[o] - obs_data[self.ref_obs]
                                   for o in self.obslist[1:]]
                    ll_diff_rel[0] += [(obs_data[o] / obs_data[self.ref_obs]
                                       - 1)*100 for o in self.obslist[1:]]
                    ll_nms = self.models + self.obslist[1:]
                else:
                    ll_nms = self.models
                lg_lbls = [[self.ref_obs] + [m.upper() for m in ll_nms]] +\
                    [[f'{m.upper()} - {self.ref_obs}' for m in ll_nms]] *\
                    self.plot_mulc
            else:
                ll_abs = [[mod_data[m] for m in self.models]]
                ll_diff = [[mod_data[m] - mod_data[self.ref_model]
                            for m in self.othr_mod]]
                ll_diff_rel = [[(mod_data[m] / mod_data[self.ref_model] - 1) *
                                100 for m in self.othr_mod]]
                lg_lbls = [[m.upper() for m in self.models]] +\
                    [[f'{m.upper()} - {self.ref_model.upper()}'
                      for m in self.othr_mod]] * self.plot_mulc

            dlist = ll_abs + ll_diff + ll_diff_rel if\
                self.include_relative_change else ll_abs + ll_diff

            thr = fmod[self.ref_model].attrs['Description'].\
                split('|')[2].split(':')[1].strip()
            regnm = reg.replace(' ', '_')

            fn = self.define_file_names(thr, 'lnplot', region=regnm)
            headtitle = f'{self.var} | {reg} | {self.tsuffix_title}' if\
                thr == 'None' else\
                f'{self.var} | Threshold: {thr} | {reg} | {self.tsuffix_title}'

            # figure settings
            if self.include_relative_change:
                figsize = (15, 11)
                figshape = (3, 1)
                ylabel = [f'Monthly mean ({self.units})',
                          f'Difference ({self.units})',
                          'Difference (%)']
            else:
                figsize = (14, 9)
                figshape = (2, 1)
                ylabel = [f'Monthly mean ({self.units})',
                          f'Difference ({self.units})']

            xlabel = None
            xlim = [-.5, 11.5]
            xticks = range(12)
            xtlbls = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            rpl.figure_init(plottype='line')
            fig, lgrid = rpl.fig_grid_setup(fshape=figshape, figsize=figsize,
                                            sharex=True, **self.line_grid)

            axs = rpl.make_line_plot(lgrid, ydata=dlist, **self.line_sets)

            [ln.set_color(lc) for ln, lc in zip(
                list(axs[0].get_lines())[:len(dlist[0])], self.abs_colors)]
            [[ln.set_color(lc) for ln, lc in zip(
                list(ax.get_lines())[:len(ll)], self.rel_colors)]
                for ax, ll in zip(axs[1:], dlist[1:])]

            # Legend
            legend_elements = [Line2D([0], [0], lw=2, color=c, label=l)
                               for c, l in zip(self.abs_colors, lg_lbls[0])]
            axs[0].legend(handles=legend_elements, fontsize='x-large',
                          framealpha=.5)
            legend_elements = [Line2D([0], [0], lw=2, color=c, label=l)
                               for c, l in zip(self.rel_colors, lg_lbls[1])]
            [ax.legend(handles=legend_elements, fontsize='x-large',
                       framealpha=.5) for ax in axs[1:]]

            [rpl.axes_settings(ax, xlabel=xlabel, xticks=xticks,
                               ylabel=ylabel[a], xtlabels=xtlbls, xlim=xlim)
             for a, ax in enumerate(axs)]

            ttl = fig.suptitle(headtitle, fontsize='xx-large')
            ttl.set_position((.5, 1.04))

            plt.savefig(os.path.join(self.img_dir, fn), bbox_inches='tight')

    def map_pctls(self):
        """
        Plotting percentile map plot
        """

        # Data
        fmod = {m: xa.open_dataset(f)
                for m, f in zip(self.models, self.fm_list)}
        fmod_msk = {m: self._mask_data(ds) for m, ds in fmod.items()}

        pctls = fmod[self.ref_model].percentiles.values
        npctl = pctls.size
        if self.ref_obs is not None:
            fobs = {o: xa.open_dataset(f)
                    for o, f in zip(self.obslist, self.fo_list)}
            fobs_msk = {o: self._mask_data(ds) for o, ds in fobs.items()}

            dlist = [[fobs_msk[self.ref_obs][self.var].values[i, :]] +
                     [fmod_msk[m][self.var].values[i, :] -
                      fobs_msk[self.ref_obs][self.var].values[i, :]
                      for m in self.models] for i in range(npctl)]
            if self.include_relative_change:
                dlist_rel = [
                    [fobs_msk[self.ref_obs][self.var].values[i, :]] +
                    [(fmod_msk[m][self.var].values[i, :] /
                     fobs_msk[self.ref_obs][self.var].values[i, :] - 1)*100
                     for m in self.models] for i in range(npctl)]

            if len(self.obslist) > 1:
                for i in range(npctl):
                    dlist[i] += [fobs_msk[o][self.var].values[i, :] -
                                 fobs_msk[self.ref_obs][self.var].values[i, :]
                                 for o in self.obslist[1:]]
                if self.include_relative_change:
                    for i in range(npctl):
                        dlist_rel[i] += [
                            (fobs_msk[o][self.var].values[i, :] /
                             fobs_msk[self.ref_obs][self.var].values[i, :]
                             - 1)*100 for o in self.obslist[1:]]
                ndata = self.nmod + len(self.obslist[1:])
            else:
                ndata = self.nmod
        else:
            dlist = [[fmod_msk[self.ref_model][self.var].values[i, :]] +
                     [fmod_msk[m][self.var].values[i, :] -
                      fmod_msk[self.ref_model][self.var].values[i, :]
                      for m in self.othr_mod] for i in range(npctl)]
            if self.include_relative_change:
                dlist_rel = [
                    [fmod_msk[self.ref_model][self.var].values[i, :]] +
                    [(fmod_msk[m][self.var].values[i, :] /
                      fmod_msk[self.ref_model][self.var].values[i, :] - 1)*100
                     for m in self.othr_mod] for i in range(npctl)]
            ndata = self.nmod-1

        data_list = [dlist, dlist_rel] if\
            self.include_relative_change else [dlist]

        # figure settings
        if ndata + 1 < 3:
            figsize = (18, 12)
        else:
            figsize = (20, 8)
        figshape = (1, ndata+1)

        # color maps
        if self.var == 'pr':
            cmap = [mpl.cm.YlGnBu] + [mpl.cm.BrBG]*ndata
        else:
            cmap = [mpl.cm.Spectral_r] + [mpl.cm.RdBu_r]*ndata

        # Labels
        ftitles = self.define_figure_titles()

        thr = fmod[self.ref_model].attrs['Description'].\
            split('|')[2].split(':')[1].strip()
        units = [self.units, f'{self.units}, diff: (%)'] if\
            self.include_relative_change else [self.units]

        st_nm_appnd = ['abs_diff', 'rel_diff']

        for dd, append, uts in zip(data_list, st_nm_appnd, units):

            # Loop over percentiles
            for p in range(npctl):
                headtitle = (f'{self.var} [{uts}] | p{pctls[p]} | '
                             f'{self.tsuffix_title}') if thr == 'None' else\
                        (f'{self.var} [{uts}] | p{pctls[p]} | '
                         f'Threshold: {thr} | {self.tsuffix_title}')

                fn = self.define_file_names(
                    thr, 'map', stat_name=f'percentile_p{pctls[p]}_{append}')

                rpl.figure_init(plottype='map')

                # Create map object and axes grid
                map_proj = rpl.define_map_object(
                    self.map_projection, **self.map_config)
                fig, axs_grid = rpl.map_setup(
                    map_proj, self.map_extent, figsize, figshape,
                    grid_lines=self.map_gridlines, **self.map_axes_conf)

                clevs_abs = self.get_clevs(np.array(dd[p][0]), centered=False)
                clevs_dif = self.get_clevs(np.array(dd[p][1]), centered=True)

                fmt_abs = self._get_colorbar_label_formatting(clevs_abs[::2])
                fmt_dif = self._get_colorbar_label_formatting(clevs_dif[::2])

                clevs = [clevs_abs] + [clevs_dif]*ndata
                fmt = [fmt_abs] + [fmt_dif]*ndata

                # Plot the maps
                mp = rpl.make_map_plot(
                    dd[p], axs_grid, self.lts, self.lns,
                    cmap=cmap, clevs=clevs, **self.map_plot_conf)
                rpl.image_colorbar(mp, axs_grid, labelspacing=2, formatter=fmt)

                # Map settings
                rpl.map_axes_settings(fig, axs_grid, headtitle=headtitle)

                [ax.text(0.5, 1.03, ft.upper(), size='large',
                         va='center', ha='center', transform=ax.transAxes)
                 for ft, ax in zip(ftitles, axs_grid)]

                plt.savefig(os.path.join(self.img_dir, fn),
                            bbox_inches='tight')

    def map_diurnal_cycle(self):
        """
        Plotting diurnal cycle map plot
        """

        # Data
        fmod = {m: xa.open_dataset(f)
                for m, f in zip(self.models, self.fm_list)}
        fmod_msk = {m: self._mask_data(ds) for m, ds in fmod.items()}

        hours = fmod[self.ref_model].hour.values
        nhour = hours.size
        fshape = ((1, nhour) if nhour <= 6 else (2, int(np.ceil(nhour/2)))
                  if 6 < nhour <= 12 else (3, int(np.ceil(nhour/3)))
                  if 12 < nhour <= 18 else (4, int(np.ceil(nhour/4))))

        if self.ref_obs is not None:
            fobs = {o: xa.open_dataset(f)
                    for o, f in zip(self.obslist, self.fo_list)}
            fobs_msk = {o: self._mask_data(ds) for o, ds in fobs.items()}

            dlist = [[fobs_msk[self.ref_obs][self.var].values[i, :]
                      for i in range(nhour)]]\
                + [[fmod_msk[m][self.var].values[i, :] -
                    fobs_msk[self.ref_obs][self.var].values[i, :]
                    for i in range(nhour)] for m in self.models]

            if len(self.obslist) > 1:
                dlist += [[fobs_msk[o][self.var].values[i, :] -
                           fobs_msk[self.ref_obs][self.var].values[i, :]
                           for i in range(nhour)] for o in self.obslist[1:]]
                ndata = self.nmod + len(self.obslist[1:])
                data_names = [self.ref_obs] +\
                    [f"{m}-{self.ref_obs}"
                     for m in self.models + self.obslist[1:]]
            else:
                ndata = self.nmod
                data_names = [self.ref_obs] + [f"{m}-{self.ref_obs}"
                                               for m in self.models]
        else:
            dlist = [[fmod_msk[self.ref_model][self.var].values[i, :]
                      for i in range(nhour)]] +\
                    [[fmod_msk[m][self.var].values[i, :] -
                      fmod_msk[self.ref_model][self.var].values[i, :]
                      for i in range(nhour)] for m in self.othr_mod]
            ndata = self.nmod-1
            data_names = [self.ref_model] + [f"{m}-{self.ref_model}"
                                             for m in self.othr_mod]

        dcycle_stat = fmod[self.ref_model].attrs['Description'].\
            split('|')[1].strip()
        if dcycle_stat == 'Amount':
            dc_units = self.units
            fn_prfx = 'amnt'
        elif dcycle_stat == 'Frequency':
            dc_units = 'frequency'
            fn_prfx = 'freq'
        else:
            print("\n\tUnknown diurnal cycle statistic, exiting...")
            sys.exit()

        thr = fmod[self.ref_model].attrs['Description'].\
            split('|')[3].split(':')[1].strip()

        # figure settings
        figsize = (22, 14)
        figshape = fshape

        # color maps and levels
        if self.var == 'pr':
            cmap = [mpl.cm.YlGnBu] + [mpl.cm.BrBG]*ndata
        else:
            cmap = [mpl.cm.Spectral_r] + [mpl.cm.RdBu_r]*ndata

        clevs_abs = self.get_clevs(np.array(dlist[0]), centered=False)
        clevs_dif = self.get_clevs(np.array(dlist[1]), centered=True)
        fmt_abs = self._get_colorbar_label_formatting(clevs_abs[::2])
        fmt_dif = self._get_colorbar_label_formatting(clevs_dif[::2])

        clevs = [clevs_abs] + [clevs_dif]*ndata
        fmt = [fmt_abs] + [fmt_dif]*ndata

        ftitles = self.define_figure_titles()

        # Loop over data sets
        for p, ft, data_name in zip(range(ndata + 1), ftitles, data_names):
            headtitle = f'{ft} | {self.var} [{dc_units}]'\
                    if thr == 'None' else\
                    f'{ft} | {self.var} [{dc_units}] | Threshold: {thr}'

            fn = self.define_file_names(thr, 'map', data_name=data_name,
                                        stat_name=f'diurnal_cycle_{fn_prfx}')

            rpl.figure_init(plottype='map')

            # Create map object and axes grid
            map_proj = rpl.define_map_object(
                self.map_projection, **self.map_config)
            fig, axs_grid = rpl.map_setup(
                map_proj, self.map_extent, figsize, figshape,
                grid_lines=self.map_gridlines, **self.map_axes_conf)

            # Plot the maps
            mp = rpl.make_map_plot(
                dlist[p], axs_grid, self.lts, self.lns, cmap=cmap[p],
                clevs=clevs[p], **self.map_plot_conf)
            rpl.image_colorbar(mp, axs_grid, labelspacing=2, formatter=fmt[p])

            # Map settings
            rpl.map_axes_settings(fig, axs_grid, headtitle=headtitle,
                                  time_mean='hour', time_units=hours)

            plt.savefig(os.path.join(self.img_dir, fn), bbox_inches='tight')

        # Line plot diurnal cycle
        if self.fm_listr is not None:
            self.line_diurnal_cycle()

    def line_diurnal_cycle(self):
        """
        Plotting diurnal cycle line plot
        """

        for reg in self.regions:

            fmod = {m: xa.open_dataset(f)
                    for m, f in zip(self.models, self.fm_listr[reg])}
            mod_data = {m: np.nanmean(fmod[m][self.var].values, axis=(1, 2))
                        for m in self.models}
            hours = fmod[self.ref_model].hour.values

            if self.ref_obs is not None:
                fobs = {o: xa.open_dataset(f)
                        for o, f in zip(self.obslist, self.fo_listr[reg])}
                obs_data = {o: np.nanmean(
                    fobs[o][self.var].values, axis=(1, 2))
                    for o in self.obslist}
                dlist = [[obs_data[self.ref_obs]] + [mod_data[m]
                                                     for m in self.models],
                         [mod_data[m] - obs_data[self.ref_obs]
                          for m in self.models]]

                if len(self.obslist) > 1:
                    dlist[0] += [obs_data[o] for o in self.obslist[1:]]
                    dlist[1] += [obs_data[o] - obs_data[self.ref_obs]
                                 for o in self.obslist[1:]]
                    ll_nms = self.models + self.obslist[1:]
                else:
                    ll_nms = self.models
                lg_lbls = [[self.ref_obs] + [m.upper() for m in ll_nms],
                           [f'{m.upper()} - {self.ref_obs}' for m in ll_nms]]
            else:
                dlist = [[mod_data[m] for m in self.models],
                         [mod_data[m] - mod_data[self.ref_model]
                          for m in self.othr_mod]]
                lg_lbls = [[m.upper() for m in self.models],
                           [f'{m.upper()} - {self.ref_model.upper()}'
                            for m in self.othr_mod]]

            xdata = [[hours]*len(dlist[0]), [hours]*len(dlist[1])]

            dcycle_stat = fmod[self.ref_model].attrs['Description'].\
                split('|')[1].strip()
            if dcycle_stat == 'Amount':
                fn_prfx = 'amnt'
            elif dcycle_stat == 'Frequency':
                fn_prfx = 'freq'
            else:
                print("\n\tUnknown diurnal cycle statistic, exiting...")
                sys.exit()

            thr = fmod[self.ref_model].attrs['Description'].\
                split('|')[3].split(':')[1].strip()
            regnm = reg.replace(' ', '_')

            fn = self.define_file_names(thr, 'lnplot', region=regnm,
                                        stat_name=f'diurnal_cycle_{fn_prfx}')
            headtitle = (f'{self.var} ({dcycle_stat}) | {reg} | '
                         f'{self.tsuffix_title}') if thr == 'None' else\
                (f'{self.var} ({dcycle_stat}) | Threshold: {thr} | {reg} | '
                 f'{self.tsuffix_title}')

            # figure settings
            figsize = (14, 12)
            figshape = (2, 1)

            ylabel = ['({})'.format(self.units),
                      'Difference ({})'.format(self.units)]
            xlabel = [None, 'Hour (UTC)']
            xlim = [[-.5, hours[-1]+.5]]*2
            xticks = hours[:]
            xtlbls = ['{:02d}'.format(h) for h in hours]

            rpl.figure_init(plottype='line')
            fig, lgrid = rpl.fig_grid_setup(fshape=figshape, figsize=figsize,
                                            **self.line_grid)

            # Scatter
            axs = rpl.make_scatter_plot(lgrid, xdata, dlist, s=100,
                                        edgecolors='#303030', lw=1.5, alpha=1.)
            [pts.set_facecolor(c)
             for pts, c in zip(axs[0].collections, self.abs_colors)]
            [pts.set_facecolor(c)
             for pts, c in zip(axs[1].collections, self.rel_colors)]

            # Legend
            legend_elements = [Line2D([0], [0], marker='o', mec='k', ms=10,
                                      lw=0, color=c, label=l)
                               for c, l, in zip(self.abs_colors, lg_lbls[0])]
            axs[0].legend(handles=legend_elements, ncol=1,
                          fontsize='large', framealpha=.5)
            legend_elements = [Line2D([0], [0], marker='o', mec='k', ms=10,
                                      lw=0, color=c, label=l)
                               for c, l, in zip(self.rel_colors, lg_lbls[1])]
            axs[1].legend(handles=legend_elements, ncol=1,
                          fontsize='large', framealpha=.5)

            [rpl.axes_settings(ax, xlabel=xlabel[a], xticks=xticks,
                               ylabel=ylabel[a], xtlabels=xtlbls, xlim=xlim[a])
             for a, ax in enumerate(axs)]

            [ax.ticklabel_format(useOffset=False, axis='y') for ax in axs]

            ttl = fig.suptitle(headtitle, fontsize='x-large')
            ttl.set_position((.5, 1.04))

            plt.savefig(os.path.join(self.img_dir, fn), bbox_inches='tight')

    def moments_plot(self):
        """
        Plotting higher-order moment statistics

        Multiple plot types are available, e.g. timeseries and map plot.
        The type shall be specified in the main RCAT configuration file
        (under the 'plotting' section). Default is timeseries.
        """

        type_of_plot = self.moments_plot_conf['plot type']

        if type_of_plot == 'timeseries':

            for reg in self.regions:

                fmod = {m: xa.open_dataset(f)
                        for m, f in zip(self.models, self.fm_listr[reg])}
                mod_data = {m: np.nanmean(
                    fmod[m][self.var].values, axis=(1, 2))
                            for m in self.models}

                err_len_msg = (
                    "\n\n\t*** Input data arrays for timeseries plot do not"
                    " all have the same lengths. This is required for these"
                    " plots. ***\n\n")
                ts_len_md = [arr.size for m, arr in mod_data.items()]
                assert len(set(ts_len_md)) == 1, err_len_msg

                if self.ref_obs is not None:
                    fobs = {o: xa.open_dataset(f)
                            for o, f in zip(self.obslist, self.fo_listr[reg])}
                    obs_data = {o: np.nanmean(
                        fobs[o][self.var].values, axis=(1, 2))
                                for o in self.obslist}

                    ts_len_all = ts_len_md +\
                        [v.size for o, v in obs_data.items()]
                    assert len(set(ts_len_all)) == 1, err_len_msg

                    dlist = [[obs_data[self.ref_obs]] +
                             [mod_data[m] for m in self.models],
                             [mod_data[m] - obs_data[self.ref_obs]
                              for m in self.models]]

                    if len(self.obslist) > 1:
                        dlist[0] += [obs_data[o] for o in self.obslist[1:]]
                        dlist[1] += [obs_data[o] - obs_data[self.ref_obs]
                                     for o in self.obslist[1:]]
                        ll_nms = self.models + self.obslist[1:]
                    else:
                        ll_nms = self.models
                    lg_lbls = [[self.ref_obs] + [m.upper() for m in ll_nms],
                               [f'{m.upper()} - {self.ref_obs}'
                                for m in ll_nms]]
                else:
                    dlist = [[mod_data[m] for m in self.models],
                             [mod_data[m] - mod_data[self.ref_model]
                              for m in self.othr_mod]]
                    lg_lbls = [[m.upper() for m in self.models],
                               [f'{m.upper()} - {self.ref_model.upper()}'
                                for m in self.othr_mod]]

                thr = fmod[self.ref_model].attrs['Description'].\
                    split('|')[1].split(':')[1].strip()
                moment_stat = fmod[self.ref_model].attrs['Description'].\
                    split('|')[0].split(':')[1].replace(' ', '').lower()
                regnm = reg.replace(' ', '_')

                headtitle = (
                    f'{self.var} | Threshold: {thr} | '
                    f'Stat: {moment_stat}\n{reg} | '
                    f'{self.tsuffix_title}') if thr != 'None' else\
                    (f'{self.var} | Stat: {moment_stat}\n{reg} | '
                     f'{self.tsuffix_title}')

                fn = self.define_file_names(thr, 'timeseries', region=regnm,
                                            stat_name=f'stat_{moment_stat}')
                # figure settings
                figsize = (16, 10)
                figshape = (2, 1)

                ylabel = [f'{self.units}', 'Difference']
                ylim = [None]*2
                xlabel = ['']*2
                xlim = [None]*2
                xticks = None
                xtlbls = None

                rpl.figure_init(plottype='line')
                fig, lgrid = rpl.fig_grid_setup(
                    fshape=figshape, figsize=figsize, **self.line_grid)

                axs = rpl.make_line_plot(lgrid, ydata=dlist, **self.line_sets)
                [ln.set_color(lc) for ln, lc in zip(
                    list(axs[0].get_lines())[:len(dlist[0])], self.abs_colors)]
                [ln.set_color(lc) for ln, lc in zip(
                    list(axs[1].get_lines())[:len(dlist[1])], self.rel_colors)]

                # Trendlines
                if self.moments_plot_conf['trendline']:
                    for ydata, lc in zip(dlist[0], self.abs_colors):
                        z = np.polyfit(np.arange(len(ydata)), ydata, 1)
                        p = np.poly1d(z)
                        rpl.make_line_plot(
                            [lgrid[0]], ydata=p(np.arange(len(ydata))),
                            color='k', lw=2, alpha=.6)
                        rpl.make_line_plot(
                            [lgrid[0]], ydata=p(np.arange(len(ydata))),
                            lw=0, marker='o', markersize=4.5, mec=lc, mfc=lc,
                            markevery=2, alpha=1)
                # Running mean
                if self.moments_plot_conf['running mean']:
                    window = self.moments_plot_conf['running mean']
                    for ydata, lc in zip(dlist[0], self.abs_colors):
                        rmn = run_mean(ydata, window, 'same')
                        rpl.make_line_plot(
                            [lgrid[0]], ydata=rmn, color='k', lw=2, alpha=.6)
                        rpl.make_line_plot(
                            [lgrid[0]], ydata=rmn, lw=0, marker='o',
                            markersize=4, mec=lc, mfc=lc, alpha=1)

                # Legend
                leg_elements = [Line2D([0], [0], lw=3, color=c, label=l)
                                for c, l in zip(self.abs_colors, lg_lbls[0])]
                if self.moments_plot_conf['trendline']:
                    legend_elements = leg_elements + [
                        Line2D([0], [0], lw=3, color='k', marker='o', mfc=c,
                               mec=c, markersize=8, alpha=.6,
                               label='lin. trend')
                        for c, _ in zip(self.abs_colors, lg_lbls[0])]
                if self.moments_plot_conf['running mean']:
                    legend_elements = leg_elements + [
                        Line2D([0], [0], lw=3, color='k', marker='o', mfc=c,
                               mec=c, markersize=8, alpha=.6,
                               label=f'run. avg (window: {window})')
                        for c, _ in zip(self.abs_colors, lg_lbls[0])]

                axs[0].legend(handles=legend_elements, ncol=2,
                              fontsize='x-large', framealpha=.5)
                leg_elements = [Line2D([0], [0], lw=3, color=c, label=l)
                                for c, l in zip(self.rel_colors, lg_lbls[1])]
                axs[1].legend(handles=leg_elements,
                              fontsize='x-large', framealpha=.5)

                [rpl.axes_settings(ax, xlabel=xlabel[a], xticks=xticks,
                                   ylabel=ylabel[a], xtlabels=xtlbls,
                                   xlim=xlim[a], ylim=ylim[a])
                 for a, ax in enumerate(axs)]

                ttl = fig.suptitle(headtitle, fontsize='xx-large')
                ttl.set_position((.5, 1.03))

                plt.savefig(
                    os.path.join(self.img_dir, fn), bbox_inches='tight')

        elif type_of_plot == 'boxplot':

            # Dimension(s) to average over
            _dim_avg = self.moments_plot_conf['boxplot averaging dimension']
            dim_avg = ('x', 'y') if _dim_avg == 'space' else _dim_avg

            grouped = self.moments_plot_conf['grouped boxplot']

            for reg in self.regions:

                mod_data = {}
                for m, f in zip(self.models, self.fm_listr[reg]):
                    with xa.open_dataset(f) as fmod:
                        dim_avg = self._space_dim(fmod) if\
                                _dim_avg == 'space' else _dim_avg
                        mod_mean = fmod[self.var].mean(dim_avg).values.ravel()
                        mod_data[m] = mod_mean[~np.isnan(mod_mean)]
                        if m == self.ref_model:
                            thr = fmod.attrs['Description'].\
                                split('|')[1].split(':')[1].strip()
                            moment_stat = fmod.attrs['Description'].\
                                split('|')[0].split(':')[1].replace(
                                    ' ', '').lower()

                if self.ref_obs is not None:
                    obs_data = {}
                    for o, f in zip(self.obslist, self.fo_listr[reg]):
                        with xa.open_dataset(f) as fobs:
                            dim_avg = self._space_dim(fobs) if\
                                    _dim_avg == 'space' else _dim_avg
                            obs_mean = fobs[self.var].mean(
                                dim_avg).values.ravel()
                            obs_data[o] = obs_mean[~np.isnan(obs_mean)]

                    dlist = [obs_data[self.ref_obs]] +\
                            [mod_data[m] for m in self.models]

                    if len(self.obslist) > 1:
                        dlist[0] += [obs_data[o] for o in self.obslist[1:]]
                        ll_nms = self.models + self.obslist[1:]
                    else:
                        ll_nms = self.models
                    lg_lbls = [self.ref_obs] + [m.upper() for m in ll_nms]
                else:
                    dlist = [mod_data[m] for m in self.models]
                    lg_lbls = [m.upper() for m in self.models]

                regnm = reg.replace(' ', '_')

                headtitle = (f'{self.var} | Stat: {moment_stat} | '
                             f'{reg} | {self.tsuffix_title}') if thr == 'None'\
                    else (f'{self.var} | Threshold: {thr} | Stat: '
                          f'{moment_stat}\n{reg} | {self.tsuffix_title}')

                fn = self.define_file_names(thr, 'boxplot', region=regnm,
                                            stat_name=f'stat_{moment_stat}')

                # figure settings
                figsize = (12, 8)
                figshape = (1, 1)

                ylabel = [f'{self.units}']
                ylim = [None]
                xlabel = ['']
                xlim = [None]
                xticks = None
                xtlbls = None

                rpl.figure_init(plottype='box')
                fig, lgrid = rpl.fig_grid_setup(
                    fshape=figshape, figsize=figsize, **self.line_grid)

                lbls = None if grouped else lg_lbls
                # bx_colors = [abs_colors, rel_colors]
                axs, bps = rpl.make_box_plot(
                    lgrid, data=dlist, labels=lbls, leg_labels=None,
                    grouped=grouped, box_colors=self.abs_colors, whis=[5, 95],
                    showfliers=False)

                if grouped:
                    # Legend
                    leg_elements = [Patch(color=c, label=l)
                                    for c, l in zip(
                                        self.abs_colors, lg_lbls[0])]

                    axs[0].legend(handles=leg_elements, fontsize='large',
                                  framealpha=.5)
                    leg_elements = [Patch(color=c, label=l)
                                    for c, l in zip(
                                        self.rel_colors, lg_lbls[1])]
                    axs[1].legend(handles=leg_elements, fontsize='large',
                                  framealpha=.5)

                [rpl.axes_settings(ax, xlabel=xlabel[a], xticks=xticks,
                                   ylabel=ylabel[a], xtlabels=xtlbls,
                                   xlim=xlim[a], ylim=ylim[a],
                                   fontsize='x-large', fontsize_lbls='x-large')
                 for a, ax in enumerate(axs)]

                ttl = fig.suptitle(headtitle, fontsize='x-large')
                ttl.set_position((.5, 1.03))

                plt.savefig(os.path.join(
                    self.img_dir, fn), bbox_inches='tight')

        elif type_of_plot == 'map':

            # Data
            fmod = {m: xa.open_dataset(f)
                    for m, f in zip(self.models, self.fm_list)}
            fmod_msk = {m: self._mask_data(ds) for m, ds in fmod.items()}
            if 'time' in fmod_msk[self.ref_model].dims:
                fmod_msk = {m: ds.mean('time') for m, ds in fmod_msk.items()}

            if self.ref_obs is not None:
                fobs = {o: xa.open_dataset(f)
                        for o, f in zip(self.obslist, self.fo_list)}
                fobs_msk = {o: self._mask_data(ds) for o, ds in fobs.items()}
                if 'time' in fobs_msk[self.ref_obs].dims:
                    fobs_msk = {o: ds.mean('time')
                                for o, ds in fobs_msk.items()}

                dlist = [fobs_msk[self.ref_obs][self.var].values] +\
                        [fmod_msk[m][self.var].values -
                         fobs_msk[self.ref_obs][self.var].values
                         for m in self.models]

                if len(self.obslist) > 1:
                    dlist += [fobs_msk[o][self.var].values -
                              fobs_msk[self.ref_obs][self.var].values
                              for o in self.obslist[1:]]
                    ndata = self.nmod + len(self.obslist[1:])
                else:
                    ndata = self.nmod
            else:
                dlist = [fmod_msk[self.ref_model][self.var].values] +\
                        [fmod_msk[m][self.var].values -
                         fmod_msk[self.ref_model][self.var].values
                         for m in self.othr_mod]
                ndata = self.nmod-1

            ftitles = self.define_figure_titles()

            thr = fmod[self.ref_model].attrs['Description'].\
                split('|')[1].split(':')[1].strip()
            moment_stat = fmod[self.ref_model].attrs['Description'].\
                split('|')[0].split(':')[1].replace(' ', '').lower()

            # figure settings
            if ndata + 1 < 3:
                figsize = (18, 12)
            else:
                figsize = (20, 10)
            figshape = (1, ndata+1)

            if self.var == 'pr':
                cmap = [mpl.cm.YlGnBu] + [mpl.cm.BrBG]*ndata
            else:
                cmap = [mpl.cm.Spectral_r] + [mpl.cm.RdBu_r]*ndata

            headtitle = (f'{self.var} [{self.units}] | Stat: '
                         f'{moment_stat} | {self.tsuffix_title}') if\
                thr == 'None' else\
                (f'{self.var} [{self.units}] | Stat: {moment_stat} | '
                 f'Threshold: {thr} | {self.tsuffix_title}')

            fn = self.define_file_names(thr, 'map',
                                        stat_name=f'stat_{moment_stat}')

            rpl.figure_init(plottype='map')

            # Create map object and axes grid
            map_proj = rpl.define_map_object(
                self.map_projection, **self.map_config)
            fig, axs_grid = rpl.map_setup(
                map_proj, self.map_extent, figsize, figshape,
                grid_lines=self.map_gridlines, **self.map_axes_conf)

            clevs_abs = self.get_clevs(np.array(dlist[0]), centered=False)
            clevs_dif = self.get_clevs(np.array(dlist[1]), centered=True)

            fmt_abs = self._get_colorbar_label_formatting(clevs_abs[::2])
            fmt_dif = self._get_colorbar_label_formatting(clevs_dif[::2])

            clevs = [clevs_abs] + [clevs_dif]*ndata
            fmt = [fmt_abs] + [fmt_dif]*ndata

            # Plot the maps
            mp = rpl.make_map_plot(
                dlist, axs_grid, self.lts, self.lns, cmap=cmap, clevs=clevs,
                **self.map_plot_conf)
            rpl.image_colorbar(mp, axs_grid, labelspacing=2, formatter=fmt)

            # Map settings
            rpl.map_axes_settings(fig, axs_grid, headtitle=headtitle)

            [ax.text(0.5, 1.03, ft.upper(), size='large',
                     va='center', ha='center', transform=ax.transAxes)
             for ft, ax in zip(ftitles, axs_grid)]

            plt.savefig(os.path.join(self.img_dir, fn), bbox_inches='tight')

    def pdf_plot(self):
        """
        Plotting frequency-intensity-distribution plot
        """

        for reg in self.regions:

            fmod = {m: xa.open_dataset(f)
                    for m, f in zip(self.models, self.fm_listr[reg])}
            mod_data = {m: np.nanmean(
                fmod[m][self.var].values*100, axis=(1, 2))
                        for m in self.models}
            bins = fmod[self.ref_model].bin_edges.values[1:]
            nbins = bins.size

            if self.ref_obs is not None:
                fobs = {o: xa.open_dataset(f)
                        for o, f in zip(self.obslist, self.fo_listr[reg])}
                obs_data = {o: np.nanmean(
                    fobs[o][self.var].values*100, axis=(1, 2))
                            for o in self.obslist}
                dlist = [[obs_data[self.ref_obs]] +
                         [mod_data[m] for m in self.models],
                         [mod_data[m] - obs_data[self.ref_obs]
                          for m in self.models]]

                if len(self.obslist) > 1:
                    dlist[0] += [obs_data[o] for o in self.obslist[1:]]
                    dlist[1] += [obs_data[o] - obs_data[self.ref_obs]
                                 for o in self.obslist[1:]]
                    ll_nms = self.models + self.obslist[1:]
                else:
                    ll_nms = self.models
                lg_lbls = [[self.ref_obs] + [m.upper() for m in ll_nms],
                           [f'{m.upper()} - {self.ref_obs}' for m in ll_nms]]
            else:
                dlist = [[mod_data[m] for m in self.models],
                         [mod_data[m] - mod_data[self.ref_model]
                          for m in self.othr_mod]]
                lg_lbls = [[m.upper() for m in self.models],
                           [f'{m.upper()} - {self.ref_model.upper()}'
                            for m in self.othr_mod]]

            thr = fmod[self.ref_model].attrs['Description'].\
                split('|')[1].split(':')[1].strip()
            regnm = reg.replace(' ', '_')

            headtitle = f'{self.var} |  {reg} | {self.tsuffix_title}'\
                if thr != 'None' else\
                f'{self.var} |  Threshold: {thr}\n{reg} | {self.tsuffix_title}'

            fn = self.define_file_names(thr, 'lnplot', region=regnm)

            # figure settings
            figsize = (19, 8)
            figshape = (1, 2)

            ylabel = ['Frequency (%)', 'Difference']
            ylim = [None]*2
            xlabel = ['({})'.format(self.units)]*2
            xlim = [[-.5, nbins-.5]]*2
            xticks = range(nbins)[::6]
            xtlbls = bins[::6]

            rpl.figure_init(plottype='scatter')
            fig, lgrid = rpl.fig_grid_setup(fshape=figshape, figsize=figsize,
                                            **self.line_grid)

            axs = rpl.make_line_plot(lgrid, ydata=dlist, **self.line_sets)
            if self.var == 'pr':
                axs[0].set_yscale('log')

            [ln.set_color(lc) for ln, lc in zip(
                list(axs[0].get_lines())[:len(dlist[0])], self.abs_colors)]
            [ln.set_color(lc) for ln, lc in zip(
                list(axs[1].get_lines())[:len(dlist[1])], self.rel_colors)]

            # Legend
            legend_elements = [Line2D([0], [0], lw=2, color=c, label=l)
                               for c, l in zip(self.abs_colors, lg_lbls[0])]
            axs[0].legend(handles=legend_elements,
                          fontsize='x-large', framealpha=.5)
            legend_elements = [Line2D([0], [0], lw=2, color=c, label=l)
                               for c, l in zip(self.rel_colors, lg_lbls[1])]
            axs[1].legend(handles=legend_elements,
                          fontsize='x-large', framealpha=.5)

            [rpl.axes_settings(ax, xlabel=xlabel[a], xticks=xticks,
                               ylabel=ylabel[a], xtlabels=xtlbls,
                               xlim=xlim[a], ylim=ylim[a])
             for a, ax in enumerate(axs)]

            ttl = fig.suptitle(headtitle, fontsize='xx-large')
            ttl.set_position((.5, 1.05))

            plt.savefig(os.path.join(self.img_dir, fn), bbox_inches='tight')

    def map_asop(self):
        """
        Plotting ASoP FC factor map plot
        """

        # Data
        fmod = {m: xa.open_dataset(f)
                for m, f in zip(self.models, self.fm_list)}

        if self.ref_obs is not None:
            fobs = {o: xa.open_dataset(f)
                    for o, f in zip(self.obslist, self.fo_list)}

            FCI = {m: np.nansum(
                np.fabs(fmod[m][self.var].values[1, :] -
                        fobs[self.ref_obs][self.var].values[1, :]), axis=0)
                   for m in self.models}
            dlist = [FCI[m] for m in self.models]
            if len(self.obslist) > 1:
                dlist += [np.nansum(
                    np.fabs(fobs[o][self.var].values[1, :] -
                            fobs[self.ref_obs][self.var].values[1, :]), axis=0)
                          for o in self.obslist[1:]]
                ndata = self.nmod + len(self.obslist[1:])
            else:
                ndata = self.nmod
        else:
            FCI = {m: np.nansum(
                np.fabs(fmod[m][self.var].values[1, :] -
                        fmod[self.ref_model][self.var].values[1, :]), axis=0)
                   for m in self.othr_mod}
            dlist = [FCI[m] for m in self.othr_mod]
            ndata = self.nmod-1

        thr = fmod[self.ref_model].attrs['Description'].\
            split('|')[2].split(':')[1].strip()

        # figure settings
        figshape = (1, ndata)
        if ndata < 3:
            figsize = (16, 10)
        else:
            figsize = (20, 9)

        fn = self.define_file_names(thr, 'map', stat_name='asop_FC')
        ftitles = self.define_figure_titles()
        headtitle = 'ASoP FC Index' if thr == 'None' else\
            f'ASoP FC Index | Threshold: {thr}'

        rpl.figure_init(plottype='map')

        # Create map object and axes grid
        map_proj = rpl.define_map_object(
            self.map_projection, **self.map_config)
        fig, axs_grid = rpl.map_setup(
            map_proj, self.map_extent, figsize, figshape,
            grid_lines=self.map_gridlines, **self.map_axes_conf)

        cmap = [mpl.cm.Spectral_r]*ndata
        clevs = [np.linspace(0, 2, 21)]*ndata

        # Plot the maps
        mp = rpl.make_map_plot(dlist, axs_grid, self.lts, self.lns, cmap=cmap,
                               clevs=clevs, **self.map_plot_conf)
        rpl.image_colorbar(mp, axs_grid, labelspacing=2, formatter='{:.1f}')

        # Map settings
        rpl.map_axes_settings(fig, axs_grid, headtitle=headtitle)

        [ax.text(0.5, 1.03, ft.upper(), size='large', va='center', ha='center',
                 transform=ax.transAxes) for ft, ax in zip(ftitles, axs_grid)]

        plt.savefig(os.path.join(self.img_dir, fn), bbox_inches='tight')

        # Line plot asop
        if self.fm_listr is not None:
            self.line_asop()

    def line_asop(self):
        """
        Plotting ASoP line plot of C and FC factors
        """

        for reg in self.regions:

            fmod = {m: xa.open_dataset(f)
                    for m, f in zip(self.models, self.fm_listr[reg])}
            mod_data = {m: np.nanmean(fmod[m][self.var].values, axis=(2, 3))
                        for m in self.models}

            bins = fmod[self.ref_model].bin_edges.values
            factors = fmod[self.ref_model].factors.values

            if self.ref_obs is not None:
                fobs = {o: xa.open_dataset(f)
                        for o, f in zip(self.obslist, self.fo_listr[reg])}
                obs_data = {o: np.nanmean(
                    fobs[o][self.var].values, axis=(2, 3))
                            for o in self.obslist}
                dlist = [[obs_data[self.ref_obs]] +
                         [mod_data[m] for m in self.models],
                         [mod_data[m] - obs_data[self.ref_obs]
                          for m in self.models]]

                if len(self.obslist) > 1:
                    dlist[0] += [obs_data[o] for o in self.obslist[1:]]
                    dlist[1] += [obs_data[o] - obs_data[self.ref_obs]
                                 for o in self.obslist[1:]]
                    ll_nms = self.models + self.obslist[1:]
                else:
                    ll_nms = self.models
                lg_lbls = [[self.ref_obs] + [m.upper() for m in ll_nms],
                           [f'{m.upper()} - {self.ref_obs}' for m in ll_nms]]
            else:
                dlist = [[mod_data[m] for m in self.models],
                         [mod_data[m] - mod_data[self.ref_model]
                          for m in self.othr_mod]]
                lg_lbls = [[m.upper() for m in self.models],
                           ['{m.upper()} - {self.ref_model.upper()}'
                            for m in self.othr_mod]]

            ylabels = [self.units, '%']
            for ff, fctr in enumerate(factors):
                sc = 100 if fctr == 'FC' else 1
                dlist_ff = [[arr[ff, :]*sc for arr in dlist[0]],
                            [arr[ff, :]*sc for arr in dlist[1]]]
                xdata = [[bins[:-1]]*len(dlist[0]), [bins[:-1]]*len(dlist[1])]

                thr = fmod[self.ref_model].attrs['Description'].\
                    split('|')[2].split(':')[1].strip()
                regnm = reg.replace(' ', '_')

                headtitle = f'ASoP ({fctr}) | {reg}\n{self.tsuffix_title}'\
                    if thr == 'None' else\
                    f'ASoP ({fctr}) | Thr: {thr}\n{reg} | {self.tsuffix_title}'

                # figure settings
                figsize = (13, 10)
                figshape = (2, 1)

                fn = self.define_file_names(
                    thr, 'lnplot', stat_name=f'asop_{fctr}', region=regnm)

                ylabel = [ylabels[ff]]*2
                ylim = [None]*2
                xlabel = [None, 'Intensity ({})'.format(self.units)]
                xlim = [[1e-2, 1e2]]*2

                rpl.figure_init(plottype='line')
                ln_grid = deepcopy(self.line_grid)
                if 'axes_pad' in ln_grid:
                    ln_grid.pop('axes_pad')
                if 'sharex' not in ln_grid:
                    ln_grid.update({'sharex': True})
                fig, lgrid = rpl.fig_grid_setup(
                    fshape=figshape, figsize=figsize, **ln_grid)

                axs = rpl.make_line_plot(lgrid, xdata=xdata, ydata=dlist_ff,
                                         axis_type='logx', **self.line_sets)

                [ln.set_color(lc) for ln, lc in zip(
                    list(axs[0].get_lines())[:len(dlist[0])], self.abs_colors)]
                [ln.set_color(lc) for ln, lc in zip(
                    list(axs[1].get_lines())[:len(dlist[1])], self.rel_colors)]

                # Legend
                legend_elements = [Line2D([0], [0], lw=2, color=c, label=l)
                                   for c, l in zip(
                                       self.abs_colors, lg_lbls[0])]
                axs[0].legend(handles=legend_elements,
                              fontsize='x-large', framealpha=.5)
                legend_elements = [Line2D([0], [0], lw=2, color=c, label=l)
                                   for c, l in zip(
                                       self.rel_colors, lg_lbls[1])]
                axs[1].legend(handles=legend_elements,
                              fontsize='x-large', framealpha=.5)

                [rpl.axes_settings(ax, xlabel=xlabel[a], ylabel=ylabel[a],
                                   xlim=xlim[a], ylim=ylim[a])
                 for a, ax in enumerate(axs)]

                ttl = fig.suptitle(headtitle, fontsize='x-large')
                ttl.set_position((.5, 1.06))

                plt.savefig(
                    os.path.join(self.img_dir, fn), bbox_inches='tight')
