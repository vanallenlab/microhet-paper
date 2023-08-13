#!/bin/bash

conda create --name deepcell python=3.8
pip install deepcell
# adding jupyter kernel
conda install ipykernel 
ipython kernel install --name=deepcell --user 
pip install imagecodecs 

conda install openjdk==8.0.152
pip install python-bioformats


# ark toolkit
pip install ark-analysis --use-deprecated=backtrack-on-build-failures

# install dask
conda install dask -c conda-forge
conda install dask distributed -c conda-forge

pip install parse

