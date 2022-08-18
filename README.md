# Principal Component Analysis for the WS22 datasets

This repository contains a custom Python script designed to perform dimensionality reduction analysis 
for the molecular geometries stored in the datasets of the WS22 database hosted in the ZENODO repository
(https://doi.org/10.5281/zenodo.6985377). 

The script works in three steps:

First, a built-in function is used to convert the Cartesian coordinates of the molecular geometries into 
an atom-atom pairwise distance descriptor of size $$N_{atoms} * (N_{atoms} - 1)/2$$. Then, this descriptor 
is scaled by using the MinMax approach. Finally, the rescaled data is passed as input to a PCA method that 
is used to project the high-dimensional descriptor into a compact 2D representation for visualization 
purposes.

## Requirements

To run this script, the following packages should be installed:

- python3 (tested with version 3.8.6)
- glob
- numpy
- pandas
- sklearn

## How to use

The script can be executed from the terminal as follows:

```
python dimred.py
```

The output of the script is a zipped csv file containing two columns storing the calculated principal
components for each molecular dataset together with an additional column with the corresponding labels
for the molecular conformations taken from the original datasets.
