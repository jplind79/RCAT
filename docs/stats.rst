.. _statistics_functions:

Statistics
==========


Arithmetic routines
-------------------

Functions for various arithmetic calculations.

.. autosummary::
   :toctree: ./generated/

    rcatool.stats.arithmetics.run_mean


ASoP
----

Analyzing Scales of Precipitation.

.. autosummary::
   :toctree: ./generated/

    rcatool.stats.ASoP.asop
    rcatool.stats.ASoP.bins_calc

Bootstrapping
-------------

Routines for bootstrap calculations. 

.. autosummary::
   :toctree: ./generated/

    rcatool.stats.bootstrap.block_bootstr

Climate indices
---------------

Routines for various climate index calculations. 

.. autosummary::
   :toctree: ./generated/

    rcatool.stats.climateindex.hotdays_calc
    rcatool.stats.climateindex.extr_hotdays_calc
    rcatool.stats.climateindex.tropnights_calc
    rcatool.stats.climateindex.ehi
    rcatool.stats.climateindex.cdd
    rcatool.stats.climateindex.Rxx
    rcatool.stats.climateindex.RRpX
    rcatool.stats.climateindex.RRtX
    rcatool.stats.climateindex.SDII

Convolution
------------

This module includes functions to perform convolution, for example image
smoothing, using scipy's convolution routines.

.. autosummary::
   :toctree: ./generated/

    rcatool.stats.convolve.kernel_gen
    rcatool.stats.convolve.filtering
    rcatool.stats.convolve.fft_prep
    rcatool.stats.convolve.convolve_fft


Probability distributions
-------------------------

.. autosummary::
   :toctree: ./generated/

    rcatool.stats.pdf.freq_int_dist
    rcatool.stats.pdf.prob_of_exceed
    rcatool.stats.pdf.perkins_skill_score


SAL module
----------

Routines for calculation of SAL statistics.

.. autosummary::
   :toctree: ./generated/
    
    rcatool.stats.sal.A_stat
    rcatool.stats.sal.S_stat
    rcatool.stats.sal.L_stat
    rcatool.stats.sal.threshold
    rcatool.stats.sal.distfunc
    rcatool.stats.sal.remove_large_objects
    rcatool.stats.sal.sal_calc
    rcatool.stats.sal.write_to_disk
    rcatool.stats.sal.run_sal_analysis

