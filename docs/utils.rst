Utilities
=========

Atmospheric physics
-------------------

Routines for calculations of various physical properties

.. autosummary::
   :toctree: ./generated/

    rcatool.utils.atmosphys.rh2sh
    rcatool.utils.atmosphys.sh2rh
    rcatool.utils.atmosphys.td2sh
    rcatool.utils.atmosphys.sh2td
    rcatool.utils.atmosphys.calc_e_from_w
    rcatool.utils.atmosphys.calc_e_from_sh
    rcatool.utils.atmosphys.calc_es
    rcatool.utils.atmosphys.calc_ws
    rcatool.utils.atmosphys.td
    rcatool.utils.atmosphys.wind2uv
    rcatool.utils.atmosphys.uv2wind
    rcatool.utils.atmosphys.brunt_vaisala_frequency
    rcatool.utils.atmosphys.lifted_condensation_temperature
    rcatool.utils.atmosphys.theta_equivalent
    rcatool.utils.atmosphys.theta_pseudoequiv

IO handling
-----------

This module provides routines representing tools to read and write NetCDF
files.

.. autosummary::
   :toctree: ./generated/

    rcatool.utils.file_io.ncdump
    rcatool.utils.file_io.openFile
    rcatool.utils.file_io.getDimensions
    rcatool.utils.file_io.getParams
    rcatool.utils.file_io.fracday2datetime
    rcatool.utils.file_io.write2netcdf

Grid applications
-----------------

Routines to remap data given source and target grids.

.. autosummary::
   :toctree: ./generated/

    rcatool.utils.grids.fnCellCorners
    rcatool.utils.grids.calc_vertices
    rcatool.utils.grids.fnRemapConOperator
    rcatool.utils.grids.fnRemapCon
    rcatool.utils.grids.add_matrix_NaNs

Config reader module
--------------------

Creates and return a dictionary built from a config file.

.. autosummary::
   :toctree: ./generated/

    rcatool.utils.ini_reader.get_config_dict

.. _polygons:

Polygons
--------

Mask polygons
^^^^^^^^^^^^^

Routine for Masking Data with Polygons.

.. autosummary::
   :toctree: ./generated/

    rcatool.utils.polygons.polygons
    rcatool.utils.polygons.mask_region
    rcatool.utils.polygons.create_polygon
    rcatool.utils.polygons.plot_polygon
    rcatool.utils.polygons.topo_mask
    rcatool.utils.polygons.find_geo_indices

Draw polygon
^^^^^^^^^^^^

Draw a simple polygon using matplotlib with mouse event handling.

.. autosummary::
   :toctree: ./generated/

    rcatool.utils.draw_polygon.Canvas.set_location
    rcatool.utils.draw_polygon.Canvas.update_path

