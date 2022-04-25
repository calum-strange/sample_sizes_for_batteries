# sample_sizes_for_batteries

This repository contains the data and code required to replicate the results of the paper 'Automatic method for the estimation of li-ion degradation test sample sizes
required to understand cell-to-cell variability' - Calum Strange, Michael Allerhand, Philipp Dechent and Goncalo dos Reis.

The data for the paper can be found in the Data folder and contains subsampled data from 

  * Baumhofer-2014 48 cells, Sanyo/Panasonic UR18650E, NMC/graphite, 1.85 Ah
  * Dechent-2020 22 cells, Samsung INR18650-35E, NCA/graphite, 3.5 Ah
  * Dechent-2017 21 cells, Samsung NR18650-15 L1, NMC/graphite, 1.5 Ah
  * Severson-2019 67 out of 124 cells, A123 APR18650 M1 A, LFP/graphite, 1.1 Ah
  * Attia-2020 45 cells, A123 APR18650 M1 A, LFP/graphite, 1.1 Ah

functions.py contains the functions used for the project.
plotting.py produces the plots found in the paper.
variance_estimation.py estimates the number of cells needed for each dataset to capture the standard deviation.

Further details can be found in the accompanying paper.
