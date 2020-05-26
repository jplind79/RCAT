Statistics
==========


Arithmetic routines
-------------------

Functions for various arithmetic calculations.

.. autosummary::
   :toctree: ./generated/

    rcat.stats.arithmetics.run_mean


ASoP
----

Analyzing Scales of Precipitation.

.. autosummary::
   :toctree: ./generated/

    rcat.stats.ASoP.asop
    rcat.stats.ASoP.bins_calc

Bootstrapping
-------------

Routines for bootstrap calculations. 

.. autosummary::
   :toctree: ./generated/

    rcat.stats.bootstrap.block_bootstr

Climate indices
---------------

Routines for various climate index calculations. 

.. autosummary::
   :toctree: ./generated/

    rcat.stats.climateindex.hotdays_calc
    rcat.stats.climateindex.extr_hotdays_calc
    rcat.stats.climateindex.tropnights_calc
    rcat.stats.climateindex.ehi
    rcat.stats.climateindex.cdd_calc
    rcat.stats.climateindex.Rxx
    rcat.stats.climateindex.RRpX
    rcat.stats.climateindex.RRtX
    rcat.stats.climateindex.SDII

Convolution
------------

This module includes functions to perform convolution, for example image
smoothing, using scipy's convolution routines.

.. autosummary::
   :toctree: ./generated/

    rcat.stats.convolve.kernel_gen
    rcat.stats.convolve.convolve2Dfunc
    rcat.stats.convolve.fft_prep
    rcat.stats.convolve.convolve_fft


Probability distributions
-------------------------

.. autosummary::
   :toctree: ./generated/

    rcat.stats.pdf.freq_int_dist
    rcat.stats.pdf.prob_of_exceed
    rcat.stats.pdf.perkins_skill_score


SAL module
----------

Routines for calculation of SAL statistics.

.. autosummary::
   :toctree: ./generated/
    
    rcat.stats.sal.A_stat
    rcat.stats.sal.S_stat
    rcat.stats.sal.L_stat
    rcat.stats.sal.threshold
    rcat.stats.sal.distfunc
    rcat.stats.sal.remove_large_objects
    rcat.stats.sal.sal_calc
    rcat.stats.sal.write_to_disk
    rcat.stats.sal.run_sal_analysis

