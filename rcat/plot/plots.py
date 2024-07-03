"""
Graphics module
---------------
Functions for different plots such as scatterplots and mapplots,
initiation and configuration of figure objects etc

Created: Autumn 2016
Authors: Petter Lind & David Lindstedt
"""

# Global modules
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# Functions
def figure_init(plottype='line', printtypes=False):
    """
    Setting up a figure object

    Parameters
    ----------
    plottype: string
        Type of plot to make
    printtypes: boolean
        If available plottypes should be printed on screen
    """

    pltypes = {
            'map': 'classic',
            'line': 'ggplot',
            'scatter': 'fivethirtyeight',
            'raster': 'seaborn-white',
            'box': 'fivethirtyeight',
            }

    if printtypes:
        print("Available plot types: \n", pltypes.keys())
    else:
        plt.style.use(pltypes[plottype])


def image_grid_setup(figsize=(12, 12), fshape=(1, 1), **grid_kwargs):
    """
    Set up the plot axes using mpl_toolkits.axes_grid1.ImageGrid
    Used primarily when plotting maps or for image analysis
    For more information on available settings:
    https://doc.ebichu.cc/matplotlib/mpl_toolkits/axes_grid1/overview.html

    Parameters
    ----------
        figsize: tuple
            Size of figure in inches; (width, height)
        fshape: tuple
            setting the shape of figure (nrow, ncol)
        **grid_kwargs: Additional keyword arguments

    Returns
    -------
        fig: Figure object
        grid: AxesGrid object

    Examples of ``**kwargs`` with default values:
        direction="row",
        axes_pad=0.02,
        add_all=True,
        share_all=False,
        label_mode="L",
        aspect=True,
        cbar_mode=None,
        cbar_location="right",
        cbar_size="5%",
        cbar_pad=None,
    """

    from mpl_toolkits.axes_grid1 import ImageGrid

    fig = plt.figure(figsize=figsize)

    grid = ImageGrid(fig,
                     111,
                     nrows_ncols=fshape,
                     **grid_kwargs
                     )
    return fig, grid


def get_nrow_ncol(npanels):
    """
    Return number of rows and columns from a given total no
    of panels for a grid
    """

    from math import sqrt, floor, ceil
    nrow, ncol = (floor(sqrt(npanels)), ceil(npanels/(floor(sqrt(npanels)))))
    return (nrow, ncol)


def fig_grid_setup(figsize=(12, 12), fshape=(1, 1), direction='row',
                   axes_pad=(None, None), **grid_kwargs):
    """
    Set up the plot axes using pyplot.subplots

    Parameters
    ----------
        figsize: tuple
            Size of figure in inches; (width, height)
        fshape: tuple
            setting the shape of figure (nrow, ncol)
        direction: string
            'row' or 'col'; rowwise or columnwise order of axes instances
        axes_pad: tuple
            padding (height,width) between edges of adjacent subplots
        **grid_kwargs: Additional keyword arguments

    Returns
    -------
        fig: Figure object
        grid: List with axes instances
    """

    fig, grid = plt.subplots(
                                fshape[0],
                                fshape[1],
                                squeeze=False,
                                **grid_kwargs
                            )

    fig.set_size_inches(figsize)
    fig.tight_layout(h_pad=axes_pad[0], w_pad=axes_pad[1])

    if direction == 'row':
        return fig, grid.flatten()
    else:
        return fig, grid.flatten(order='F')


def axes_settings(ax, figtitle=None, xlabel=None, ylabel=None, xtlabels=None,
                  ytlabels=None, xticks=None, yticks=None, xlim=None,
                  ylim=None, color='k', fontsize='xx-large',
                  fontsize_lbls='xx-large', fontsize_title='xx-large',
                  ftitle_location='center'):
    """
    Configuration of axes; titles and labels

    Parameters
    ----------
        ax: axes object
            Axes object assoicated with figure
        figtitle: string
            The headtitle
        xlabel, ylabel: strings
            Labels of the axes
        xticks, yticks: lists or arrays
            If set, location of ticks
        xtlabels, ytlabels: lists
            Contains tick labels (corresponding to ticks location)
        xlim, ylim: lists
            Limits of the axes
        color: str
            Color for the xtick labels
        fontsize/fontsize_lbls/fontsize_title: string or int
            Size of font for axes labels, ticklabels, and figure title
            respectively
        ftitle_location: str
            Horizontal alignment of figure title; center (default),
            right or left.
    """

    if figtitle is not None:
        ax.set_title(figtitle, fontsize=fontsize_title, color=color,
                     loc=ftitle_location)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize, color=color)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize, color=color)

    ((xticks is not None) and ax.set_xticks(xticks))
    ((xtlabels is not None) and ax.set_xticklabels(xtlabels, color=color))
    ((yticks is not None) and ax.set_yticks(yticks))
    ((ytlabels is not None) and ax.set_yticklabels(ytlabels, color=color))

    ((xlim is not None) and ax.set_xlim(xlim))
    ((ylim is not None) and ax.set_ylim(ylim))

    [t.set_fontsize(fontsize_lbls) for t in ax.get_xticklabels()]
    [t.set_fontsize(fontsize_lbls) for t in ax.get_yticklabels()]


def map_axes_settings(fig, axs, figtitle=None, headtitle=None,
                      time_mean=None, time_units=None, fontsize='x-large',
                      fontsize_htitle='xx-large'):
    """
    Settings for map plot axes

    Parameters
    ----------
        fig: Object handle
            Figure handle
        axs: Axes objects
            Single object or list with axes from map plot
        figtitle: Strings
            Single title or list of titles for each plot in figure
        headtitle: String
            Head title of figure
        time_mean: string
            If maps should be labeled according to time averages;
            'season'/'month'/'hour'
        time_units: list/array
            Optional. If time_mean is set, time_units is a list of seasons,
            months or hours to be used in the labeling.
        fontsize: string or int
            Size of font for figure title
        fontsize_htitle: string or int
            Size of font for suptitle
    """

    from itertools import cycle

    def map_label(axs, labels):
        [ax.text(0.05, 0.92, label, horizontalalignment='left',
                 verticalalignment='center', fontsize='large',
                 bbox=dict(facecolor='white', edgecolor='black',
                           boxstyle='round'), transform=ax.transAxes)
         for ax, label in zip(axs, cycle(labels))]

    laxs = axs if isinstance(axs, list) else list(axs)
    if figtitle is not None:
        ftitles = figtitle if isinstance(figtitle, list) else [figtitle]
        [ax.set_title(title, fontsize=fontsize) for title, (i, ax)
         in zip(cycle(ftitles), enumerate(laxs))]
    if headtitle is not None:
        fig.suptitle(headtitle, fontsize=fontsize_htitle)

    if time_mean == 'season':
        assert len(laxs) % 4 == 0,\
                "\n{}\n".format("For seasonal data, a plot (axis instance) "
                                "for each of the four seasons "
                                "must be provided")
        map_label(laxs, ["DJF", "MAM", "JJA", "SON"])
    elif time_mean == 'month':
        assert len(laxs) % 12 == 0,\
                "\n{}\n".format("For monthly data, a plot (axis instance) "
                                "for each of the twelve months must "
                                "be provided")
        map_label(laxs, ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL",
                         "AUG", "SEP", "OCT", "NOV", "DEC"])
    elif time_mean == 'hour':
        map_label(laxs, ["{:02d} Z".format(h) for h in time_units])


def make_scatter_plot(grid, xdata, ydata, sdata=None, fcolors=None,
                      ecolors=None, lbl_fontsize='large',
                      axis_type='linear', labels=None, **sc_kwargs):
    """
    Create a scatter plot

    Parameters
    ----------
        grid: AxesGrid object
            returned from the 'fig_grid_setup' function
        xdata/ydata: Array/list
            1D array or list of 1D arrays with data for x/y axis
        fcolors: array/list
            List of colors to be used for each data set in input data. This is
            separate from 'color'/'c' option available from matplotlib.scatter
            call (and set in sc_kwargs) where all individual input data sets
            will have that specific color/s. If set, then
            ecolors need also be supplied.
        ecolors: array/list
            List of edge colors to be used for each data set in input data.
            This is separate from 'edgecolors'/'ec' option available from
            matplotlib.scatter call (and set in sc_kwargs) where all individual
            input data sets will have that specific color/s. If set, then
            fcolors need also be supplied.
        lbl_fontsize: string/float
            Fontsize for legend labels
        axis_type: str
            Linear or log axes: 'linear' (defualt), 'logx'/'logy'/'logxy' (log
            x, y or both axes).
        labels: String/List
            String or list of strings with legend labels
        **sc_kwargs: keyword arguments
            arguments (key=value) that can be used in pyplot.scatter
            See matplotlib.org for more information

    Returns
    -------
        axs: Axes objects
            The axes objects created for each plot
    """

    # Check that necessary color settings are supplied
    if ecolors is not None:
        errmsg = "If edgecolors (ecolors) is set, so must also fcolors!"
        assert fcolors is not None, errmsg
    if fcolors is not None:
        errmsg = "If facecolors (fcolors) is set, so must also ecolors!"
        assert ecolors is not None, errmsg

    if labels is not None:
        dlabels = [[labels]] if not isinstance(labels,
                                               (list, tuple)) else [labels]
    if len(grid) == 1:
        xd = [xdata] if isinstance(xdata[0],
                                   (list, tuple, np.ndarray)) else [[xdata]]
        yd = [ydata] if isinstance(ydata[0],
                                   (list, tuple, np.ndarray)) else [[ydata]]
        if sdata is not None:
            sd = [sdata] if isinstance(
                sdata[0], (list, tuple, np.ndarray)) else [[sdata]]
        if fcolors is not None:
            fcls = [fcolors] if isinstance(
                fcolors, (list, tuple, np.ndarray)) else [[fcolors]]
            ecls = [ecolors] if isinstance(
                ecolors, (list, tuple, np.ndarray)) else [[ecolors]]

        axs = []
        for i, ax in enumerate(grid):
            if fcolors is not None:
                if sdata is not None:
                    pts = [ax.scatter(xx, yy, c=cc, ec=ec, s=ss, **sc_kwargs)
                           for xx, yy, cc, ec, ss in zip(
                               xd[i], yd[i], fcls[i], ecls[i], sd[i])]
                else:
                    pts = [ax.scatter(xx, yy, c=cc, ec=ec, **sc_kwargs)
                           for xx, yy, cc, ec in zip(
                               xd[i], yd[i], fcls[i], ecls[i])]
            else:
                if sdata is not None:
                    pts = [ax.scatter(xx, yy, s=ss, **sc_kwargs)
                           for xx, yy, ss in zip(xd[i], yd[i], sd[i])]
            if labels is not None:
                [pt.set_label(lbl) for pt, lbl in zip(pts, labels[i])]
                ax.legend(fontsize=lbl_fontsize)

            if axis_type in ('logx', 'logxy'):
                ax.set_xscale('log')
            if axis_type in ('logy', 'logxy'):
                ax.set_yscale('log')

            axs.append(ax)

    else:
        msg = "*** ERROR *** \n x data must be in a list when multiple figures"
        assert isinstance(xdata, list), msg
        msg = "*** ERROR *** \n y data must be in a list when multiple figures"
        assert isinstance(ydata, list), msg
        if labels is not None:
            msg = """*** ERROR *** \n Labels must be in a list when """
            """multiple figures"""
            assert isinstance(labels, list), msg
        if fcolors is not None:
            fcls = fcolors if isinstance(
                fcolors, (list, tuple, np.ndarray)) else [fcolors]
            ecls = ecolors if isinstance(
                ecolors, (list, tuple, np.ndarray)) else [ecolors]

        axs = []
        for i, ax in enumerate(grid):
            xd = xdata[i]
            yd = ydata[i]
            if isinstance(yd[0], (list, tuple, np.ndarray)):
                if fcolors is not None:
                    if sdata is not None:
                        pts = [ax.scatter(xx, yy, s=ss, c=cc, ec=ec,
                                          **sc_kwargs)
                               for xx, yy, ss, cc, ec in zip(
                                   xd, yd, sdata[i], fcls[i], ecls[i])]
                    else:
                        pts = [ax.scatter(xx, yy, c=cc, ec=ec, **sc_kwargs)
                               for xx, yy, cc, ec in zip(
                                   xd, yd, fcls[i], ecls[i])]
                else:
                    if sdata is not None:
                        pts = [ax.scatter(xx, yy, s=ss, **sc_kwargs)
                               for xx, yy, ss in zip(xd, yd, sdata[i])]
                    else:
                        pts = [ax.scatter(xx, yy, **sc_kwargs)
                               for xx, yy in zip(xd, yd)]
                if labels is not None:
                    [pt.set_label(lbl) for pt, lbl in zip(pts, dlabels[i])]

            else:
                if fcolors is not None:
                    if sdata is not None:
                        pts = ax.scatter(xd, yd, s=sdata[i], c=fcls[i],
                                         ec=ecls[i], **sc_kwargs)
                    else:
                        pts = ax.scatter(xd, yd, c=fcls[i], ec=ecls[i],
                                         **sc_kwargs)
                else:
                    if sdata is not None:
                        pts = ax.scatter(xd, yd, s=sdata[i], **sc_kwargs)
                    else:
                        pts = ax.scatter(xd, yd, **sc_kwargs)

                if labels is not None:
                    pts.set_label(dlabels[i])

            if labels is not None:
                ax.legend(fontsize=lbl_fontsize)

            if axis_type in ('logx', 'logxy'):
                ax.set_xscale('log')
            if axis_type in ('logy', 'logxy'):
                ax.set_yscale('log')

            axs.append(ax)

    return axs


def make_raster_plot(data, grid=None, clevs=None, norm=None, cmap='viridis',
                     **rs_kwargs):
    """
    Create a raster plot

    Parameters
    ----------
        grid: AxesGrid object
            returned from the 'image_grid_setup' function
        data: List
            List with 2D array(s) of data
        cmap: string/list
            String or list with strings of predefined Matplotlib colormaps.
            Defaults to 'viridis'
        clevs: Iterable data structure
            Consisting of lists with defined contour levels; e.g.
            (np.arange(1,10,2), [0,2,4,6,8]), [np.arange(100,step=5)]*3
        norm: BoundaryNorm object
            Object generated from matplotlib.colors.BoundaryNorm function.
            Generate a colormap index based on discrete intervals.
        **rs_kwargs: keyword arguments
            arguments (key=value) that can be used in pyplot.imshow
            See matplotlib.org for more information

    Returns
    -------
        axs: Axes objects
            The axes objects created for each plot
        rasters: Plot objects
            The raster plot objects created for each plot
    """

    # Single or multiple 2d data arrays
    iter_data = data if isinstance(data, (list, tuple)) else [data]

    igrid = image_grid_setup(fshape=get_nrow_ncol(len(iter_data)))[1]\
        if grid is None else grid

    # Number of grids
    nplots = len(iter_data)

    # Make sure clevs is provided
    if clevs is None:
        if len(grid.cbar_axes) > 1 and all([cax.get_axes_locator() for cax in
                                            grid.cbar_axes]):
            clevs = list(map(gen_clevels, iter_data, [15]*len(iter_data)))
        else:
            clevs = gen_clevels(np.stack(iter_data), 15)

    # Replace None in list of clevs with generated levels
    elif hasattr(clevs, '__iter__') and any([a is None for a in clevs]):
        clevs = [gen_clevels(iter_data[i], 15) if cl is None else cl
                 for i, cl in enumerate(clevs)]

    # Create iterators of input arguments
    iter_clevs = clevs if hasattr(clevs[0], '__iter__') else [clevs for i in
                                                              range(nplots)]
    iter_cmap = cmap if isinstance(cmap, (list, tuple)) else [cmap for i
                                                              in range(nplots)]
    if norm is None:
        iter_norm = []
        for cl, cm in zip(iter_clevs, iter_cmap):
            if isinstance(cm, str):
                ncolrs = mpl.cm.get_cmap(cm).N
            else:
                ncolrs = cm.N
            iter_norm.append(mpl.colors.BoundaryNorm(cl, ncolrs))
    else:
        iter_norm = norm if hasattr(norm, '__iter__') else [norm for i in
                                                            range(nplots)]
    rasters = []
    axs = []
    for g, ax in enumerate(igrid):

        rasters.append(ax.imshow(iter_data[g], cmap=iter_cmap[g],
                                 norm=iter_norm[g], **rs_kwargs))
        axs.append(ax)

    return axs, rasters


def make_line_plot(grid, ydata, xdata=None, labels=None,
                   lbl_fontsize='x-large', axis_type='linear', **lp_kwargs):
    """
    Create a line plot

    Parameters
    ----------
        grid: Axis object
            returned from the 'fig_grid_setup' function
        ydata: array/list
            Required. 1D array or list of 1D arrays with data for y axis
        xdata: array/list
            Optional. 1D array or list of 1D arrays with data for x axis
        labels: string/list
            String or list of strings with line labels
        lbl_fontsize: string/float
            Fontsize for legend labels
        axis_type: str
            Linear or log axes: 'linear' (defualt), 'logx'/'logy'/'logxy' (log
            x, y or both axes).
        **lp_kwargs: keyword arguments
            arguments (key=value) that can be used in pyplot.line
            See matplotlib.org for more information

    Returns
    -------
        axs: Axes objects
            The axes objects created for each plot
    """

    if len(grid) == 1:

        yd = [ydata] if isinstance(
            ydata[0], (list, tuple, range, np.ndarray)) else [[ydata]]
        if xdata is not None:
            xd = [xdata] if isinstance(
                xdata[0], (list, tuple, range, np.ndarray)) else [[xdata]]

        if labels is not None:
            dlabels = [[labels]] if not isinstance(labels,
                                                   (list, tuple)) else [labels]

        axs = []
        for i, ax in enumerate(grid):
            if xdata is not None:
                lines = [ax.plot(xx, yy, **lp_kwargs)[0]
                         for xx, yy in zip(xd[i], yd[i])]
            else:
                lines = [ax.plot(yy, **lp_kwargs)[0] for yy in yd[i]]

            if labels is not None:
                [line.set_label(lbl) for line, lbl in zip(lines, dlabels[i])]
                ax.legend(fontsize=lbl_fontsize)

            if np.any([np.any(np.array(dd) > 0.0) for dd in yd[i]]) and\
               np.any([np.any(np.array(dd) < 0.0) for dd in yd[i]]):
                ax.axhline(color='k', lw=2, ls='--', alpha=.7)

            if axis_type in ('logx', 'logxy'):
                ax.set_xscale('log')
            if axis_type in ('logy', 'logxy'):
                ax.set_yscale('log')

            axs.append(ax)

    else:
        msg = "*** ERROR *** \n y data must be in a list when multiple figures"
        assert isinstance(ydata, list), msg

        if xdata is not None:
            msg = """*** ERROR *** \n x data must be in a list """
            """when multiple figures"""
            assert isinstance(xdata, list), msg

        if labels is not None:
            msg = """*** ERROR *** \n Labels must be in a list when """
            """multiple figures"""
            assert isinstance(labels, list), msg

        axs = []
        for i, ax in enumerate(grid):
            yd = ydata[i]
            if isinstance(yd[0], (list, tuple, range, np.ndarray)):
                if xdata is not None:
                    xd = xdata[i]
                    lines = [ax.plot(xx, yy, **lp_kwargs)[0]
                             for xx, yy in zip(xd, yd)]
                else:
                    lines = [ax.plot(yy, **lp_kwargs)[0] for yy in yd]

                if labels is not None:
                    [line.set_label(lbl)
                     for line, lbl in zip(lines, labels[i])]
                    ax.legend(fontsize=lbl_fontsize)

                if np.any([np.any(np.array(dd) > 0.0) for dd in yd]) and\
                   np.any([np.any(np.array(dd) < 0.0) for dd in yd]):
                    ax.axhline(color='k', lw=2, ls='--', alpha=.7)
            else:
                if xdata is not None:
                    lines = ax.plot(xdata[i], yd, **lp_kwargs)
                else:
                    lines = ax.plot(yd, **lp_kwargs)

                if labels is not None:
                    [line.set_label(labels[i]) for line in lines]
                    ax.legend(fontsize=lbl_fontsize)

                if np.any(yd > 0.0) and np.any(yd < 0.0):
                    ax.axhline(color='k', lw=2, ls='--', alpha=.7)

            if axis_type in ('logx', 'logxy'):
                ax.set_xscale('log')
            if axis_type in ('logy', 'logxy'):
                ax.set_yscale('log')

            axs.append(ax)

    return axs


def make_box_plot(grid, data, labels=None, leg_labels=None, grouped=False,
                  box_colors=None, **box_kwargs):
    """
    Create a box plot

    Parameters
    ----------
        grid: AxesGrid object
            returned from the 'fig_grid_setup' function
        data: List/Array
            1d array or list of 1d arrays with data for boxplot
        labels: str/list
            String or a list of strings with xtick labels (for each box/group
            of boxes)
        leg_labels: str/list
            String or a list of strings with legend labels (mostly used for
            grouped boxplots).
        grouped: boolean
            Whether to plot grouped boxplot. If True, input data must be a
            dictionary. See _grouped_boxplot function for more info.
        box_colors: array/list
            Optional list of colors to be used for the boxes.
        **box_kwargs: keyword arguments
            arguments (key=value) that can be used in pyplot.boxplot
            See matplotlib.org for more information

    Returns
    -------
        axs: list
            Axes objects for each plot
        bps: list
            Each item in list is a dictionary mapping each component of the
            boxplot to a list of the `.Line2D` instances created.
    """

    if len(grid) == 1:
        ldata = [data] if not isinstance(data, (list, tuple)) else data
        idata = [ldata] if len(ldata) > 1 else ldata
    else:
        idata = data

    axs = []
    bps = []
    for g, ax in enumerate(grid):
        if labels is not None:
            msg = "*** ERROR *** \n 'labels' must be given in a list!"
            assert isinstance(labels, list), msg
            lbls = labels[g] if len(grid) > 1 else labels
        else:
            lbls = labels
        if leg_labels is not None:
            msg = "*** ERROR *** \n 'leg_labels' must be given in a list!"
            assert isinstance(leg_labels, list), msg
            lg_lbls = leg_labels[g] if len(grid) > 1 else leg_labels
        else:
            lg_lbls = leg_labels

        # data to plot
        bdata = idata[g]

        if grouped:
            assert isinstance(bdata, dict),\
                "\n{}\n".format("**Error**\nIf plotting grouped boxplot "
                                "(grouped=True), then input data must "
                                "be a dictionary.")
            bp = _grouped_boxplot(
                ax, bdata, group_names=lbls, leg_labels=lg_lbls,
                box_colors=box_colors, **box_kwargs)
        else:
            bp = ax.boxplot(bdata, labels=lbls, patch_artist=True,
                            **box_kwargs)
            _decorate_box(ax, bp, colors=box_colors)
            if leg_labels is not None:
                cols = [bx.get_facecolor() for bx in bp['boxes']]
                h = custom_legend(cols, lg_lbls)
                ax.legend(handles=h, fontsize='large')

        axs.append(ax)
        bps.append(bp)

    return axs, bps


def _grouped_boxplot(ax, data, group_names=None, leg_labels=None,
                     box_colors=None, box_width=0.6, box_spacing=1.0,
                     **box_kwargs):
    """
    Draws a grouped boxplot.

    Parameters
    ----------
        data: dictionary
            A dictionary of length equal to the number of the groups.
            If group_names are not provided, the group names will be taken from
            the keys of the dictionary (data.keys()). The key values should
            be a list of arrays. The length of the list should be equal to
            the number of subgroups.
        group_names:
            The group names, should be the same as data.keys(),
            but can be ordered.
    """

    nsubgroups = np.array([len(v) for v in data.values()], dtype=object)
    # assert len(np.unique(nsubgroups)) == 1,\
    #     "\n{}\n".format("Number of subgroups for each property "
    #                     "differ!")
    # nsubgroups = nsubgroups[0]

    bps = []
    cpos = 1
    label_pos = []
    for k, ngrp in zip(data, nsubgroups):
        d = data[k]
        pos = np.arange(ngrp) + cpos
        label_pos.append(pos.mean())
        bp = ax.boxplot(d, positions=pos, widths=box_width, patch_artist=True,
                        **box_kwargs)
        _decorate_box(ax, bp, colors=box_colors)
        cpos += ngrp + box_spacing
        bps.append(bp)
    if 'vert' in box_kwargs:
        if box_kwargs['vert'] is False:
            ax.set_ylim(0, cpos-1)
            ax.set_yticks(label_pos)
            if group_names is not None:
                ax.set_yticklabels(group_names)
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
            #     ax.set_yticklabels(data.keys())
        else:
            ax.set_xlim(0, cpos-1)
            ax.set_xticks(label_pos)
            if group_names is not None:
                ax.set_xticklabels(group_names)
            else:
                plt.setp(ax.get_xticklabels(), visible=False)
            #     ax.set_xticklabels(data.keys())
    else:
        ax.set_xlim(0, cpos-1)
        ax.set_xticks(label_pos)
        if group_names is not None:
            ax.set_xticklabels(group_names)
        else:
            plt.setp(ax.get_xticklabels(), visible=False)
        #     ax.set_xticklabels(data.keys())

    if leg_labels is not None:
        cols = [bx.get_facecolor() for bx in bp['boxes']]
        h = custom_legend(cols, leg_labels)
        ax.legend(handles=h, fontsize='large')

    return bps


def _decorate_box(ax, bp, colors=None):
    """
    Configuration of the boxes in a boxplot

    Parameters
    ----------
        ax: Axes handle
        bp: boxplot object; i.e. returned from pyplot.boxplot call
        colors: List/array of colors for the boxes
    """

    from itertools import cycle

    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    bxcolors = [mpl.colors.to_rgba(c, alpha=.8) for c in colors]
    facecolors = [mpl.colors.to_rgba(c, alpha=.5) for c in colors]

    # fill in each box with a color
    [box.set_facecolor(color=(c)) for box, c in
        zip(bp['boxes'], cycle(facecolors))]
    [box.set(edgecolor=c, lw=3) for box, c in
        zip(bp['boxes'], cycle(bxcolors))]

    # change color of whiskers
    for (w1, w2), c in zip(zip(bp['whiskers'][0::2], bp['whiskers'][1::2]),
                           cycle(bxcolors)):
        w1.set(color=c, linestyle='-', lw=2)
        w2.set(color=c, linestyle='-', lw=2)

    # draw a black line for the median
    [m.set(color=c, linewidth=2.5) for m, c in zip(bp['medians'],
                                                   cycle(bxcolors))]

    # Set fliers
    for flier in bp['fliers']:
        flier.set(marker='.', markersize=8,
                  mfc='k', mew=0.0, alpha=0.3)


def custom_legend(colors, labels, linestyles=None):
    """
    Creates a list of matplotlib Patch objects that can be passed to
    the legend(...) function to create a custom legend.

    Parameters
    ----------
        colors: list
            A list of colors, one for each entry in the legend.
            You can also include a linestyle, for example: 'k--'
        labels: list
            A list of labels, one for each entry in the legend.
    """

    if linestyles is not None:
        assert len(linestyles) == len(colors),\
            "Length of linestyles must match length of colors."

    h = []
    for k, (c, l) in enumerate(zip(colors, labels)):
        clr = c
        ls = 'solid'
        if linestyles is not None:
            ls = linestyles[k]
        patch = mpl.patches.Patch(color=clr, label=l, linestyle=ls)
        h.append(patch)
    return h


def gen_clevels(data, nsteps, robust=None):
    """
    Create contour levels based on min and max of input data

    Parameters
    ----------
        data: array
            data array
        nsteps: int
            number of levels to be produced
        robust: None or string
            If to soften the max/min limits due to extreme values;
            "top"/"bottom"/"both", which end(s) to soften.
    """

    dmin = float('{:.2g}'.format(np.nanmin(data)))
    dmin = 0.0 if abs(dmin-0.0) < 1e-10 else dmin
    dmin = np.nanpercentile(data, 2) if robust in ('bottom', 'both') else dmin
    dmax = float('{:.2g}'.format(np.nanmax(data)))
    dmax = 0.0 if abs(dmax-0.0) < 1e-10 else dmax
    dmax = np.nanpercentile(data, 98) if robust in ('top', 'both') else dmax
    frac = (dmax - dmin)/nsteps
    # if frac > 1.0:
    #     dstep = np.round(frac)
    # else:
    #     dstep = np.round(frac, decimals=1)
    dstep = float('{:.2g}'.format(frac))

    return np.arange(dmin, dmax+dstep, dstep)


def map_setup(grid, lats, lons, proj='stere', lon_0=None, lat_0=None,
              lat_1=None, res='l', zoom='crnrs', crnr_vals=None,
              zoom_geom=[None, None], **mkwargs):
    """
    Create a basemap object to be used in the overlaying of 2d plots.

    Parameters
    ----------
        grid: list
            List with axes objects for each plot in figure
        lats: array(s)
            array or list of arrays with latitudes
        lons: array(s)
            array or list of arrays with longitudes
        proj: string
            the projection to be used (so far only two proj's:
            sterographic and lambert conformal)
        lat_0/lon_0: float
            centre latitude and longitude location of map
        lat_1: float
            first standard parallell. if proj='lcc', lat_1 must be given.
        res: string
            resolution (low 'l', intermediate 'i' or high 'h')
        zoom: string
            "cornrs" or "geom", how to set the extent of map
        crnr_vals: 1D list/array
            List of values for corners given as [llcrnrlat, llcrnrlon,
            urcrnrlat, urcrnrlon]
        zoom_geom: 1D list/array
            Integers setting map geometry [width, height] in meters.
        **mkwargs: key word arguments
            Additional arguments provided to Basemap call. See
            http://matplotlib.org/basemap/# for more info.

    Returns
    -------
        m: Basemap object
            map object
        coords: list
            list containing x,y arrays of the transformed longitudes
            to map coordinates
    """

    from mpl_toolkits.basemap import Basemap

    def _get_mapobj(ax, lats, lons, _lon0, _lat0, lat1):
        # Make sure lons are +/- 180 deg.
        if lons.max() > 180:
            lons[lons > 180] -= 360.

        # If lon/lat is 1D; create 2D meshgrid
        lon, lat = np.meshgrid(lons, lats)\
            if lats.ndim == 1 else (lons, lats)

        # Calculate domain mid point if not given
        idx = tuple([np.int(i/2) for i in lat.shape])
        lat0 = lat[idx] if _lat0 is None else _lat0
        lon0 = lon[idx] if _lon0 is None else _lon0

        if zoom == 'crnrs':
            if crnr_vals is None:
                corners = [lat[0, 0], lon[0, 0], lat[-1, -1], lon[-1, -1]]
            else:
                corners = [crnr_vals[i] for i in range(4)]
        else:
            corners = [None for i in range(4)]
            assert zoom_geom, '''\nError! zoom_geom argument is not set
            correctly.\nSet to [width/height] in meters, and try again.'''

        m = Basemap(projection=proj, ax=ax,
                    lat_0=lat0, lon_0=lon0, lat_1=lat1,
                    width=zoom_geom[0], height=zoom_geom[1],
                    llcrnrlat=(corners[0]), urcrnrlat=(corners[2]),
                    llcrnrlon=(corners[1]), urcrnrlon=(corners[3]),
                    rsphere=(6378137.00, 6356752.3142), resolution=res,
                    **mkwargs)
        return m, m(lon, lat)

    ngrid = grid.ngrids

    if isinstance(lats, list):
        if len(lats) != ngrid:
            grid = grid[:len(lats)]
            ngrid = len(grid)
            print("\n{}\n{}\n\n".format(
                "***** WARNING *****",
                "Number of lat/lon arrays provided does not "
                "match number of plots (axes instances from "
                "grid setup). Map objects created only for "
                "provided lat/lon arrays"))
        llats = lats
        llons = lons
    else:
        llats = [lats]*ngrid
        llons = [lons]*ngrid

    # Lat 0/1, Lon 0
    llon0 = lon_0 if isinstance(lon_0, list) else [lon_0]*ngrid
    llat0 = lat_0 if isinstance(lat_0, list) else [lat_0]*ngrid
    if proj == 'lcc':
        _llat1 = lat_1 if isinstance(lat_1, list) else [lat_1]*ngrid
        llat1 = np.copy(llat0) if np.isin(None, _llat1) else _llat1
    else:
        llat1 = [None]*ngrid

    ms, crds = zip(*[
        _get_mapobj(ax, llats[i], llons[i], llon0[i], llat0[i], llat1[i])
                     for i, ax in enumerate(grid)])
    m_arr, coords = (ms[0], crds[0]) if len(grid) == 1 else (ms, crds)

    return m_arr, coords


def make_map_plot(data, grid=None, map_obj=None, coords=None,
                  lats=None, lons=None, cmap=None, clevs=None,
                  robust=None, norm=None, mesh=False, filled=True,
                  **map_kwargs):
    """
    Producing  map plots

    Parameters
    ----------
        data: array/list/tuple
            Array or list/tuple of 2D data array(s) to plot
        grid: AxesGrid object
            returned from the 'image_grid_setup' function
            If not provided it is generated using number of data arrays as
            input.
        map_obj: tuple
            List with Basemap objects returned from 'map_setup' function.
            If not provided, it is generated within this function.
        coords: tuple
            List with arrays of map coordinates; returned from
            'map_setup' function
        lats, lons: arrays
            If map_obj is not provided, latitudes and longitudes need to be
            provided to setup a Basemap object.
        cmap: string/list
            String or list with strings of predefined Matplotlib colormaps.
            For filled contour plots it defaults to 'viridis'.
        clevs: Iterable data structure
            Consisting of lists with defined contour levels; e.g.
            (np.arange(1,10,2), [0,2,4,6,8]), [np.arange(100,step=5)]*3
        robust: string
            See gen_clevs function for info
        norm: BoundaryNorm object
            Object generated from matplotlib.colors.BoundaryNorm function.
            Generate a colormap index based on discrete intervals.
        mesh: boolean
            Whether to plot data as mesh. If false (default), contour plot is
            made.
        filled: boolean
            Whether to color fill between contours or not. Defaults to True
        **map_kwargs: keyword arguments
            arguments (key=value) that can be used in pyplot.contour(f)
            See
            http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.contour
    Returns
    -------
        mplots: List
            List with map plot instances
    """

    # Single or multiple 2d data arrays
    iter_data = data if isinstance(data, (list, tuple)) else [data]

    igrid = image_grid_setup(fshape=get_nrow_ncol(len(iter_data)))[1]\
        if grid is None else grid
    if map_obj is None:
        assert lats is not None, "Lats and lons must be provided"
        imap_obj, icrds = map_setup(igrid, lats, lons)
    else:
        imap_obj, icrds = (map_obj, coords)

    # Number of grids
    nplots = len(iter_data)

    # Make sure clevs is provided
    if clevs is None:
        if len(grid.cbar_axes) > 1 and all([cax.get_axes_locator() for cax in
                                            grid.cbar_axes]):
            clevs = list(map(gen_clevels, iter_data, [15]*len(iter_data),
                         [robust]*len(iter_data)))
        else:
            clevs = gen_clevels(np.stack(iter_data), 15, robust)

    # Replace None in list of clevs with generated levels
    elif hasattr(clevs, '__iter__') and any([a is None for a in clevs]):
        clevs = [gen_clevels(iter_data[i], 15, robust) if cl is None else cl
                 for i, cl in enumerate(clevs)]

    # Create iterators of input arguments
    iter_clevs = clevs if hasattr(clevs[0], '__iter__') else [clevs for i in
                                                              range(nplots)]
    if cmap is None:
        if filled:
            cmap = 'viridis'
        else:
            pass

    iter_cmap = cmap if isinstance(cmap, (list, tuple)) else [cmap for i
                                                              in range(nplots)]
    if norm is None:
        if cmap is None:
            iter_norm = [None]*nplots
        else:
            iter_norm = []
            for cl, cm in zip(iter_clevs, iter_cmap):
                if cm is None:
                    iter_norm.append(None)
                else:
                    if isinstance(cm, str):
                        ncolrs = mpl.cm.get_cmap(cm).N
                    else:
                        ncolrs = cm.N
                    iter_norm.append(mpl.colors.BoundaryNorm(cl, ncolrs))
    else:
        iter_norm = norm if hasattr(norm, '__iter__') else [norm for i in
                                                            range(nplots)]
    iter_x = [icrds[0]]*nplots if not isinstance(icrds[0], tuple) else\
        [icrds[i][0] for i in range(len(icrds))]
    iter_y = [icrds[1]]*nplots if not isinstance(icrds[0], tuple) else\
        [icrds[i][1] for i in range(len(icrds))]
    iter_m = imap_obj if isinstance(imap_obj, (list, tuple)) else\
        [imap_obj]*nplots

    # !! NB It would be nice to use map or starmap but don't know how to handle
    # **kwargs.
    # Ex.
    # from itertools import starmap as smap
    # from itertools import cycle
    # iter_kwargs = [{'alpha': .6, 'lw': 2} for i in range(nplots)]
    #
    # arg_iter = zip(grid, iter_m, iter_x, iter_y, data, iter_clevs, iter_cmap,
    #               iter_norm, cycle([filled]), **iter_kwargs)
    # plots = smap(plot_map, arg_iter)
    if nplots != igrid.ngrids:
        ll = list(set(range(nplots)).symmetric_difference(
            set(range(igrid.ngrids))))
        [igrid[i].remove() for i in ll]
        igrid = igrid[:nplots]
        print("\n{}\n{}\n\n".format(
            "***** WARNING *****",
            "Mismatch in number of plots "
            "(axes instances from grid setup) and number"
            " of 2D data arrays provided. Excess grids are removed"))

    mplots = []
    for i in range(nplots):
        mplots.append(plot_map(iter_m[i], iter_x[i], iter_y[i],
                               iter_data[i], iter_clevs[i], iter_cmap[i],
                               iter_norm[i], mesh, filled, **map_kwargs))

    out_mplots = mplots.pop() if len(mplots) == 1 else mplots
    return out_mplots


def plot_map(m, x, y, data, clevs, cmap, norm, mesh, filled, **map_kwargs):
    """
    Producing a map plot

    Parameters
    ----------
        m: Basemap object
            Handle for the map object returned from 'map_setup' function
        data: numpy array
            2D data array to plot
        x,y: numpy arrays
            Arrays of map coordinates; returned from 'map_setup' function
        clevs: List/array
            Contour levels
        cmap: string
            Color map. See http://matplotlib.org/users/colormaps.html for more
            information.
        norm: Boundary norm object
            Normalize data to [0,1] to use for mapping colors
        mesh: boolean
            Whether to plot data as mesh. If false (default), contour plot is
            made.
        filled: Boolean
            Whether to color fill between contours or not. Defaults to True
        **map_kwargs: keyword arguments
            arguments (key=value) that can be used in pyplot.contour(f)
            See
            http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.contour

    Returns
    -------
        cs: Contour plot object
    """

    m.drawmapboundary(fill_color='aqua')
    try:
        # m.drawcountries(linewidth=1.0, color="#272727")
        m.drawcountries(linewidth=1.7, color="#272727")
    except ValueError:
        pass
    try:
        # m.drawcoastlines(linewidth=0.8, color='#272727')
        m.drawcoastlines(linewidth=1.3, color='#272727')
    except ValueError:
        pass

    if mesh:
        cs = m.pcolormesh(x, y, data,
                          cmap=cmap,
                          norm=norm,
                          **map_kwargs)
    else:
        if filled:
            cs = m.contourf(x, y, data,
                            levels=clevs,
                            norm=norm,
                            cmap=cmap,
                            **map_kwargs)
        else:
            if cmap is not None:
                map_kwargs['colors'] = None
                cs = m.contour(x, y, data,
                               levels=clevs,
                               norm=norm,
                               cmap=cmap,
                               **map_kwargs)
            else:
                cs = m.contour(x, y, data,
                               norm=norm,
                               levels=clevs,
                               **map_kwargs)

    return cs


def image_colorbar(cs, cbaxs, title=None, labelspacing=1,
                   labelsize='x-large', formatter='{:.2f}', **cbar_kwargs):
    """
    Add colobar to map plot

    Parameters
    ----------
        cs: Plot object
            Such as an image (imshow) or a contour set (with contourf)
        cbaxs: cbar axis object/list
            Colorbar axis object or list with axes objects for each
            plot in figure
        title: list
            list of strings with colorbar titles
        labelspacing: int
            Label spacing; the integer value represents the number of
            steps between each label. 1 show each label (default), 2 every
            second, etc.
        labelsize: str/int
            Size of labels; integer value or a string (e.g. 'large')
        **cbar_kwargs: keyword arguments
            arguments (key=value) that can be used in pyplot.colobar
            See
            http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.colorbar
    """

    cs = cs if isinstance(cs, list) else [cs]
    nplots = len(cs)

    # Tick label format
    fmt = formatter if isinstance(formatter, list) else [formatter]*nplots

    if nplots != cbaxs.ngrids:
        ll = list(set(range(nplots)).symmetric_difference(
            set(range(cbaxs.ngrids))))
        [cbaxs[i].cax.remove() for i in ll]
        cbaxs = cbaxs[:nplots]

    if title is not None:
        cb_title = [title]*nplots \
                if not isinstance(title, list) else title

    cbs = []
    for i, ax, ft in zip(range(nplots), cbaxs, fmt):
        try:
            lvls = cs[i].levels
            # lvls = [round(i, 2) for i in lvls]
        except AttributeError:
            lvls = cs[i].norm.boundaries
            # lvls = [round(i, 2) for i in lvls]

        # Set format of tick labels
        tlbls = [ft.format(i) for i in lvls]

        if 'ticks' not in cbar_kwargs:
            ticks = np.linspace(lvls[0], lvls[-1], len(lvls))
        else:
            ticks = cbar_kwargs['ticks']
        cb = ax.cax.colorbar(cs[i],
                             spacing='uniform', **cbar_kwargs)
        if ax.cax.orientation in ['right', 'left']:
            ax.cax.set_yticks(ticks[::labelspacing])
            ax.cax.set_yticklabels(tlbls[::labelspacing])
        else:
            ax.cax.set_xticks(ticks[::labelspacing])
            ax.cax.set_xticklabels(tlbls[::labelspacing])

        ax.cax.tick_params(labelsize=labelsize, length=0)

        if title is not None:
            axis = ax.cax.axis[ax.cax.orientation]
            axis.label.set_text(cb_title[i])
            axis.label.set_size(labelsize)

        cbs.append(cb)

    return cbs
