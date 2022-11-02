### Provisional pipeline

## Convert raw files
partition_brainsight.py:
The brainsight files need to be modified so that they can be read into MNE

convert.py:
Convert raw data into MNE, apply the brainsight coordinates, filter.

## Independent component analysis
do_ica.py:
Do ICA on raw files and remove ocular components

## Isolate theta bursts
mem_theta_zoom.py
Identify bursts of theta activity and epochs them
