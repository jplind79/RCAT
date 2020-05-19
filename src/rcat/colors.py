"""
This module provides access to predefined (Matplotlib) or self-produced
colors and color maps.

@author Petter Lind
@date   2015-05-18
"""

import sys
from matplotlib import colors
import matplotlib.pyplot as plt
import palettable

# Self-produced colormaps
#
prct_diff = ["#006400", "#3CB371", "#8220F0", "#000096", "#0000CD", "#4169E1",
             "#1E90FF", "#00BFFF", "#A0D2FF", "#D2F5FF", "#FFFFC8", "#FFE132",
             "#FFAA00", "#FF6E00", "#FF0000", "#C80000", "#A02323", "#FF69B4",
             "#E4BD9A", "#BD885A"]
prct_diff = colors.ListedColormap(prct_diff)
prct_diff.set_over("#a14f07")
prct_diff.set_under("#42f942")
prct_diff_r = colors.ListedColormap(prct_diff.colors[::-1])

myWhYlOr = ["#DFEDED", "#E9F0D2", "#F0F2B7", "#F6F59A", "#FAF87C", "#FDFB5A",
            "#FEFE29", "#FFF500", "#FFE700", "#FFD900", "#FFCB00", "#FFBD00",
            "#FFAE00", "#FF9F00", "#FF8F00", "#FF7D00", "#FF6B00",
            "#FF5500", "#FF3A00", "#FF0000"]

prec_diff = ["#8B2500", "#983900", "#A64D00", "#B46100", "#C27500", "#CF8B0D",
             "#DAA543", "#E4BE78", "#EFD8AE", "#F9F2E4", "#E4EEE4", "#AECEAE",
             "#78AD78", "#438C43", "#0D6C0D", "#285428", "#5D3F5D", "#932A93",
             "#C915C9", "#FF00FF"]
prec_diff = colors.ListedColormap(prec_diff)
prec_diff.set_over("#fe8181")
prec_diff.set_under("#e0301e")

prec_diff_wzero = ["#8B2500", "#983900", "#A64D00", "#B46100", "#C27500",
                   "#CF8B0D", "#DAA543", "#E4BE78", "#EFD8AE", "#F9F2E4",
                   "#FFFFFF", "#FFFFFF", "#E4EEE4", "#AECEAE",
                   "#78AD78", "#438C43", "#0D6C0D", "#285428", "#5D3F5D",
                   "#5B7AD7", "#2D50B9", "#233E90", "#142352"]
prec_diff_wzero = colors.ListedColormap(prec_diff_wzero)
prec_diff_wzero.set_over("#142352")
prec_diff_wzero.set_under("#e0301e")

myGrBuPu_long = ["White", "#edfac2", "#cdffcd", "#99f0b2", "#53bd9f",
                 "#32a696", "#3296b4", "#0570b0", "#05508c", "#0a1f96",
                 "#2c0246", "#6a2c5a"]
myGrBuPu_long = colors.ListedColormap(myGrBuPu_long)
myGrBuPu_long.set_over("#ff00ff")
myGrBuPu_long.set_under("White")

myGrBuPu = ["#edfac2", "#cdffcd", "#99f0b2", "#53bd9f", "#32a696", "#3296b4",
            "#0570b0", "#05508c", "#0a1f96", "#2c0246", "#6a2c5a", "#ff99ac",
            "#ff4d6e", "#ffcccc", "#ffffcc"]
myGrBuPu = colors.ListedColormap(myGrBuPu)
myGrBuPu.set_over("#ff00ff")
myGrBuPu.set_under("White")

topography = ["#79B2DE", "#ACD0A5", "#94BF8B", "#A8C68F", "#BDCC96",
              "#D1D7AB", "#E1E4B5", "#EFEBC0", "#E8E1B6", "#DED6A3",
              "#D3CA9D", "#CAB982", "#C3A76B", "#B9985A", "#AA8753",
              "#AC9A7C", "#BAAE9A", "#CAC3B8", "#E0DED8"]


# Palettable colors (more info: https://jiffyclub.github.io/palettable/)
#
# Greens
greens_seq = palettable.colorbrewer.get_map('Greens',
                                            'sequential', 8).mpl_colormap

# Greys
greys_seq = palettable.colorbrewer.get_map('Greys',
                                           'sequential', 8).mpl_colormap

# Diverging brown-blue-green
BrBg_div = palettable.colorbrewer.get_map('BrBg', 'diverging', 9).mpl_colormap

# ??
unknown = ["#DFEDED", "#E9F0D2", "#F0F2B7", "#F6F59A", "#FAF87C", "#FDFB5A",
           "#FEFE29", "#FFF500", "#FFE700", "#FFD900", "#FFCB00", "#FFBD00",
           "#FFAE00", "#FF9F00", "#FF8F00", "#FF7D00", "#FF6B00", "#FF5500",
           "#FF3A00", "#FF0000"]
unknown = colors.ListedColormap(unknown)


# Good for line plots
set3 = palettable.colorbrewer.get_map('Set3', 'qualitative', 11).mpl_colors
set2 = palettable.colorbrewer.get_map('Set2', 'qualitative', 8).mpl_colors
set1 = palettable.colorbrewer.get_map('Set1', 'qualitative', 9).mpl_colors

# Misc
almost_black = '#262626'
magenta = "#FF00FF"
green = "#00EE00"

color_dict =\
        {
            'topography': topography,
            'prec_diff': prec_diff,
            'prec_diff_wzero': prec_diff_wzero,
            'myGrBuPu': myGrBuPu,
            'myGrBuPu_long': myGrBuPu_long,
            'prct_diff': prct_diff,
            'prct_diff_r': prct_diff_r,
            'myWhYlOr': myWhYlOr,
            'set1': set1,
            'set2': set2,
            'set3': set3,
            'unknown': unknown
        }
single_color_dict = \
        {
            'green': green,
            'magenta': magenta,
            'almost_black': almost_black
        }


def getcolormap(cmap_name, custom=False):
    '''
    Function to retrieve colormap, either customized (custom=True) or available
    through Matplotlib predefined colormaps.

    Parameters
    ----------
        cmap_name: string
            String, giving name of colormap to be retrieved.
        custom: Boolean
            Logical indicating self-produced (custom=True) or Matplotlib
            colormap.

    Returns
    -------
        cmap: Matplotlib colormap object
    '''

    if custom:
        cmap = color_dict[cmap_name]
    else:
        msg = "Error retrieving colormap: {}.\nMake sure it exists in "
        "Matplotlib's predefined colormaps, or change accordingly.".\
            format(cmap_name)
        try:
            cmap = plt.cm.get_cmap(cmap_name)
        except ValueError:
            print(msg)
            sys.exit

    return cmap


def getsinglecolor(color_name):
    '''
    Function to retrieve single custom color.

    Parameters
    ----------
        color_name: string
            String, giving name of color to be retrieved.

    Returns
    -------
        color: Matplotlib color object
    '''

    try:
        color = single_color_dict[color_name]
    except ValueError:
        print("Color {} does not exist in single color dictionary.".
              format(color))
        sys.exit

    return color


def norm_colors(bounds, ncolors, clip=False):
    """
    In addition to min and max of levels, this function takes as arguments
    boundaries between which data is to be mapped. The colors are then
    linearly distributed between these 'bounds'.
    """
    return palettable.palette.BoundaryNorm(boundaries=bounds, ncolors=ncolors,
                                           clip=clip)
