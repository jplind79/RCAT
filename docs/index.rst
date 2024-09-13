.. RCAT documentation master file, created by
   sphinx-quickstart on Wed Dec  4 11:21:06 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the RCAT documentation!
==================================

The Regional Climate Analysis Tool (RCAT) is an analysis tool originally
developed for evaluation of regional climate models, but may be used to analyze
output from most types of weather and climate models. It is purely written in
Python where the aim is for a modular code in a functional style.

The purpose is to have an efficient and user-friendly tool to both facilitate
quick standard evaluations in model development processes, but also to perform
more in-depth climate data analysis.

The tool is adapted to new demands in regional climate modeling, where the
produced output data volumes become very large and analysis is typically
carried out on HPC systems.

:doc:`Tutorials </tutorials>` **- Start here**

Instructions for new RCAT users on how to install the software and making your
first plot.

:doc:`How-to guides </howto>`

Hands on guides with code examples.

:doc:`Development </development>`

:doc:`Reference </api>`

Reference material (APIs)

.. toctree::
   :maxdepth: 1
   :hidden:
   :includehidden:

   tutorials
   howto
   development
   api
   release
