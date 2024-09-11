.. _howto:

How-To Guides
=============

The purpose of this section is to provide a new user with an overview of
RCAT -- an easy-to-follow guide of RCAT applications, in order to quickly
and easily get started with the tool. To this end you will find below
a few rather simple instructions or 'recipes' that will allow you to do some
basic analysis of model output and make simple visualizations of the results.


Preparation
-----------

To get started with the application examples it is expected that the RCAT
environment has been installed. If not, go ahead and :doc:`do so </install>`
before continuation. The main configuration and setup of RCAT is done in
*<path-to-RCAT>/config/config_main.ini*. It is therefore encouraged to read through the
:doc:`configuration </config>` and :doc:`statistics </statistics>` sections before (or
along with) going through the examples below.


.. note:: The Use Cases (and some parts of the source code) are in a few apects very
        specific to the HPC system at the National Supercomputer Centre (NSC)
        in Sweden; for example, available observation data sets and folder
        structure of model output. In future updates of RCAT, we strive to make
        it more general and flexible.

Use Cases
---------

#. :doc:`The Annual and Seasonal Cycles </usecase1>`

#. :doc:`PDF's on different time scales </usecase2>`

#. :doc:`Diurnal Variations </usecase3>`


RCAT polygons
-------------

* :doc:`How to create and plot polygons </polygons-howto>`

.. toctree::
    :maxdepth: 0
    :hidden:

    usecase1
    usecase2
    usecase3
    polygons-howto

