#! /bin/bash

conda create --name dev
conda activate dev
conda install -c conda-forge pyvips

conda install ipykernel
ipython kernel install --name=dev --user

conda install numpy

conda install pytorch torchvision torchaudio cpuonly -c pytorch



# above does python 3.1
conda create --name dev2 python=3.8
conda activate dev2
conda install -c conda-forge pyvips numpy ipykernel
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install openjdk==8.0.152
pip install pathml

ipython kernel install --name=dev2 --user


# clone airlab repo and install
cd airlab
pip install -e .

# install legacy fork of PFMM
pip install git+https://github.com/jmnyman/PathFlow-MixMatch


# downgrade to opencv 3.X
## https://stackoverflow.com/questions/54734538/opencv-assertion-failed-215assertion-failed-npoints-0-depth-cv-32
pip uninstall opencv-python
pip install opencv-python==3.4.9.31