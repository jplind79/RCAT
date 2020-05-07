.. _polygons_howto:

Polygons in RCAT
================
-- How to plot them and create new ones
---------------------------------------

Polygons are used in RCAT to extract or select data and statistics for specified
sub-regions or areas. These consist of text files containing information of
latitudes and longitudes for the area and are stored under
*<path-to-rcat>/src/polygons*. The :doc:`geosfuncs <geosfuncs>` module use these
polygons to do the extraction (with the *mask_region* function) and it also has
a number of different help functions to create new polygons (for RCAT or
elsewhere) and plot them conveniently.

The following tutorial will describe some of this functionality along with a few
examples.


Plot a polygon
..............

The :doc:`geosfuncs <geosfuncs>` module has a function called *plot_polygon*
which allows you to plot one of the existing polygons on map. There are a couple
of ways to apply the function -- either from within python (or in a script)
where the module is imported and plotting function calls can be made, or the
function call can be made directly from command line::

    >> import geosfuncs as gf
    >> gf.plot_polygon?
    Signature: gf.plot_polygon(polygon, savefig=False, figpath=None)
    Docstring:
    Plot polygon on map.

    Parameters
    ----------
    polygon: string or list
        Name of polygon as defined by poly_dict dictionary in 'polygons'
        function, or list with polygon coordinates [[lon1, lat1], [lon2, lat2],
        ..., [lon1, lat1]].
    savefig: boolean
        If True, figure is saved to 'figpath' location ('figpath' must be set!).
        If false, figure is displayed on screen.
    figpath: string
        Path to folder for saved polygon figure.

The *plot_polygon* function takes a polygon name as input chosen from the list
of existing polygons in <path-to-rcat>/src/polygons. [Note that the polygon name
is the text file name without '.txt' and underscores replaced by white space.]
Also new polygons can be plotted if providing a list with polygon coordinates
(not possible when executing function from command line!).  The keyword
arguments allows to save the polygon plot, if not it is just displayed.

If you do not know which polygons are available, you can easily print them::

    >> gf.polygons(poly_print=True)

    Available polygons/regions:

        Alps
        Black Sea
        British Isles
        Central Europe
        Denmark
        East Europe
        Fenno-Scandinavia
        Finland
        France
        Germany
        Iberian peninsula
        Mediterranean Sea
        Netherlands
        Norway
        South-East Europe
        Spain
        Sweden
        Switzerland
        United Kingdom
        West Europe

To plot the polygon of British Isles::

    >> gf.plot_polygon('British Isles')

This function call can be made directly from the command line. Run the 
*geosfuncs.py* script (make sure it is executable) providing
appropriate arguments:

.. code-block:: bash

    ./geosfuncs.py --help
    ./geosfuncs.py -p plot -a "British Isles" --save True --figpath $HOME


Create a new polygon
....................

New regions/polygons can be created using the *create_polygon* function, either
from within python or from the command line. The call involves a set of
instructions where the user is continuously prompted for information and
actions. 

From command line:

.. code-block:: bash

    ./geosfuncs.py -p create

The creation part is made by clicking pointer on a displayed map. If you want to
save selected polygon to RCAT, make sure to provide correct folder path and an
appropriate polygon name. Once saved it will automatically be ready for RCAT --
check for example by printing available polygons:

.. code-block:: bash

    ./geosfuncs.py -p printareas




