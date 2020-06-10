Utilities
=========

Atmospheric physics
-------------------

Routines for calculations of various physical properties

.. autosummary::
   :toctree: ./generated/

    rcat.utils.atmosphys.rh2sh
    rcat.utils.atmosphys.td2sh
    rcat.utils.atmosphys.sh2td
    rcat.utils.atmosphys.es
    rcat.utils.atmosphys.e
    rcat.utils.atmosphys.td
    rcat.utils.atmosphys.wind2uv
    rcat.utils.atmosphys.uv2wind
    rcat.utils.atmosphys.calc_vaisala

IO handling
-----------

This module provides routines representing tools to read and write NetCDF
files.

.. autosummary::
   :toctree: ./generated/

    rcat.utils.file_io.ncdump
    rcat.utils.file_io.openFile
    rcat.utils.file_io.getDimensions
    rcat.utils.file_io.getParams
    rcat.utils.file_io.fracday2datetime
    rcat.utils.file_io.write2netcdf

Grid applications
-----------------

Routines to remap data given source and target grids.

.. autosummary::
   :toctree: ./generated/

    rcat.utils.grids.fnCellCorners
    rcat.utils.grids.calc_vertices
    rcat.utils.grids.fnRemapConOperator
    rcat.utils.grids.fnRemapCon
    rcat.utils.grids.add_matrix_NaNs

Config reader module
--------------------

Creates and return a dictionary built from a config file.

.. autosummary::
   :toctree: ./generated/

    rcat.utils.ini_reader.get_config_dict

.. _polygons:

Polygons
--------

Mask polygons
^^^^^^^^^^^^^

Routine for Masking Data with Polygons.

.. autosummary::
   :toctree: ./generated/

    rcat.utils.polygons.polygons
    rcat.utils.polygons.mask_region
    rcat.utils.polygons.create_polygon
    rcat.utils.polygons.plot_polygon
    rcat.utils.polygons.topo_mask
    rcat.utils.polygons.find_geo_indices

Draw polygon
^^^^^^^^^^^^

Draw a simple polygon using matplotlib with mouse event handling.

.. autosummary::
   :toctree: ./generated/

    rcat.utils.draw_polygon.Canvas.set_location
    rcat.utils.draw_polygon.Canvas.update_path

